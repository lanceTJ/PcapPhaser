import os
import numpy as np
from scapy.all import PcapReader, IP, TCP
from collections import defaultdict
from numba import njit

class FeatureExtractor:
    """
    Base class for extracting packet-level features from PCAP files.
    Handles flow separation based on CICFlowMeter standards and truncation.
    """
    def __init__(self):
        pass

    def extract_features(self, pcap_path: str, feature_type: str, config: dict) -> dict:
        """
        Extract features for all flows in the PCAP.
        :param pcap_path: Path to PCAP file.
        :param feature_type: Type of feature (e.g., 'packet_length').
        :param config: Dict with 'pss' section containing 'max_flow_length' (default 10000) and 'timeout_sec' (default 64).
        :return: Dict {flow_id: np.array(feature_seq)}.
        """
        max_flow_length = config.get('pss', {}).get('max_flow_length', 10000)
        timeout_sec = config.get('pss', {}).get('timeout_sec', 64)
        
        # Use defaultdict(list) where each value is a list of flows (each flow is a list of features and timestamps)
        flow_dict = defaultdict(list)  # key: tuple_str, value: list of {'features': [], 'timestamps': []}
        
        with PcapReader(pcap_path) as reader:
            for pkt in reader:
                if IP not in pkt:
                    continue  # Skip non-IP packets
                
                src_ip = pkt[IP].src
                dst_ip = pkt[IP].dst
                proto = pkt[IP].proto
                sport = pkt.sport if hasattr(pkt, 'sport') else 0
                dport = pkt.dport if hasattr(pkt, 'dport') else 0
                timestamp = float(pkt.time)
                
                # Five-tuple key
                tuple_key = f"{src_ip}-{dst_ip}-{sport}-{dport}-{proto}"
                
                # If no flows yet for this tuple, start a new one
                if not flow_dict[tuple_key]:
                    flow_dict[tuple_key].append({'features': [], 'timestamps': [], 'first_ts': timestamp})
                
                current_flow = flow_dict[tuple_key][-1]
                prev_ts = current_flow['timestamps'][-1] if current_flow['timestamps'] else timestamp
                prev_pkt = None  # For features needing previous packet
                
                # Check for new flow conditions
                start_new_flow = False
                if len(current_flow['features']) >= max_flow_length:
                    start_new_flow = True  # Truncate: start new from current packet
                elif (timestamp - prev_ts) > timeout_sec:
                    start_new_flow = True  # Timeout: start new from current packet
                
                # Handle TCP FIN/RST: add current to old, start new for next
                is_fin_rst = False
                if TCP in pkt:
                    flags = pkt[TCP].flags
                    if flags & (0x01 | 0x04):  # FIN (0x01) or RST (0x04)
                        is_fin_rst = True
                
                if start_new_flow:
                    # Start new flow and add current packet to it
                    flow_dict[tuple_key].append({'features': [], 'timestamps': [], 'first_ts': timestamp})
                    current_flow = flow_dict[tuple_key][-1]
                
                # Compute feature value (subclass-specific)
                if current_flow['features']:  # If not first packet in flow
                    prev_pkt = {'len': len(current_flow['features'][-1]), 'ts': prev_ts}  # Simplified prev_pkt
                feature_value = self.compute_feature(pkt, prev_pkt)
                
                # Append to current flow
                current_flow['features'].append(feature_value)
                current_flow['timestamps'].append(timestamp)
                
                if is_fin_rst and not start_new_flow:
                    # After adding current, start new for next packet
                    flow_dict[tuple_key].append({'features': [], 'timestamps': [], 'first_ts': None})  # first_ts set on next pkt
        
        # Flatten to {unique_flow_id: np.array(features)}
        result = {}
        for tuple_key, flows in flow_dict.items():
            for idx, flow in enumerate(flows):
                if not flow['features']:
                    continue  # Skip empty flows
                first_ts = flow['first_ts']
                flow_id = f"{tuple_key}-{idx}-{int(first_ts * 1000)}"  # ms timestamp
                result[flow_id] = np.array(flow['features'])
        
        # Persist to file
        output_dir = os.path.join('feature_matrix', feature_type)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, os.path.basename(pcap_path) + '.npz')
        np.savez(output_path, **result)  # Save dict as compressed NPZ
        
        return result

    def compute_feature(self, pkt, prev_pkt):
        """
        To be overridden by subclasses.
        :param pkt: Current Scapy packet.
        :param prev_pkt: Dict with info from previous packet (or None).
        :return: Float or int feature value.
        """
        raise NotImplementedError

class PacketLengthExtractor(FeatureExtractor):
    """
    Extracts packet length feature.
    """
    def compute_feature(self, pkt, prev_pkt):
        return len(pkt)

class IATExtractor(FeatureExtractor):
    """
    Extracts inter-arrival time (IAT). Uses Numba for diff computation post-flow.
    Note: Since IAT needs sequence, override extract_features to compute after collecting timestamps.
    """
    def extract_features(self, pcap_path: str, feature_type: str, config: dict) -> dict:
        # First collect timestamps per flow using base class logic, but set features to timestamps temporarily
        temp_extractor = FeatureExtractor()
        temp_result = temp_extractor.extract_features(pcap_path, feature_type, config)
        
        # Now compute IAT for each flow's timestamps
        for flow_id, ts_array in temp_result.items():
            temp_result[flow_id] = self.compute_iat(np.array(ts_array))
        
        # Persist as in base
        output_dir = os.path.join('feature_matrix', feature_type)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, os.path.basename(pcap_path) + '.npz')
        np.savez(output_path, **temp_result)
        
        return temp_result

    @njit
    def compute_iat(self, ts_array: np.ndarray) -> np.ndarray:
        if len(ts_array) < 2:
            return np.array([0.0])
        iat = np.diff(ts_array)
        return np.concatenate(([0.0], iat))  # Prepend 0 for first packet

    def compute_feature(self, pkt, prev_pkt):
        # Temporarily return timestamp for collection
        return float(pkt.time)

class DirectionExtractor(FeatureExtractor):
    """
    Extracts direction: 1 for forward (src to dst), -1 for backward.
    Assumes first packet defines forward direction.
    """
    def compute_feature(self, pkt, prev_pkt):
        # Need flow context; simplified: assume src is client if sport > dport, else -1
        if pkt.sport > pkt.dport:
            return 1
        else:
            return -1

class UpDownRateExtractor(FeatureExtractor):
    """
    Extracts up/down rate: len(pkt) / (IAT + epsilon), signed by direction.
    Requires IAT, so similar to IATExtractor, compute post-flow.
    """
    def extract_features(self, pcap_path: str, feature_type: str, config: dict) -> dict:
        # Collect lengths, timestamps, directions
        temp_extractor = FeatureExtractor()
        temp_result = temp_extractor.extract_features(pcap_path, feature_type, config)
        
        # For each flow, compute rates
        epsilon = 1e-9
        for flow_id, data_array in temp_result.items():
            # Assume data_array temporarily holds [len, ts, direction] per packet
            # But base compute_feature needs override; for demo, assume collected as structured array
            # Simplified: re-parse or adjust base for multi-value
            pass  # Implement similar to IAT, but compute len / (diff(ts) + eps) * sign(dir)
        
        # Persistence as above
        return temp_result

    def compute_feature(self, pkt, prev_pkt):
        # Temporarily collect multiple: return [len(pkt), float(pkt.time), self.get_direction(pkt)]
        pass  # Adjust base to handle list of values per packet if needed

# Usage example (not part of class):
# config = {'pss': {'max_flow_length': 10000, 'timeout_sec': 64}}
# extractor = PacketLengthExtractor()
# features = extractor.extract_features('path/to/pcap', 'packet_length', config)