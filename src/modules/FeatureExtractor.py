import os
import numpy as np
from typing import Union, List, Dict
from scapy.all import PcapReader, IP, TCP
from collections import defaultdict
from numba import njit

class FeatureExtractor:
    """
    Class for extracting packet-level features from PCAP files.
    Handles flow separation based on CICFlowMeter standards and truncation.
    Supports features: packet_length, inter_arrival_time, direction, up_down_rate.
    Supports extracting multiple feature types in one pass.
    """
    def extract_features(self, pcap_path: str, feature_type: Union[str, List[str]], config: dict) -> Dict[str, dict]:
        """
        Extract features for all flows in the PCAP, supporting single or multiple feature types.
        :param pcap_path: Path to PCAP file.
        :param feature_type: Single string or list of strings for feature types.
        :param config: Dict with 'pss' section containing 'max_flow_length' (default 10000) and 'timeout_sec' (default 64).
        :return: Dict {feature_type: {flow_id: np.array(feature_seq)}}, simplified if single type.
        """
        if isinstance(feature_type, str):
            feature_types = [feature_type]
        else:
            feature_types = feature_type
        
        supported_features = {'packet_length', 'inter_arrival_time', 'direction', 'up_down_rate'}
        if not set(feature_types).issubset(supported_features):
            raise ValueError(f"Unsupported features: {set(feature_types) - supported_features}")
        
        max_flow_length = config.get('pss', {}).get('max_flow_length', 10000)
        timeout_sec = config.get('pss', {}).get('timeout_sec', 64)
        
        # defaultdict(list) where each value is list of flows (each flow is dict with shared data: timestamps, lengths, directions, is_super_flow)
        flow_dict = defaultdict(list)  # key: tuple_str
        
        needs_lengths = any(ft in {'packet_length', 'up_down_rate'} for ft in feature_types)
        needs_directions = any(ft in {'direction', 'up_down_rate'} for ft in feature_types)
        
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
                pkt_len = len(pkt)
                
                # Five-tuple key
                tuple_key = f"{src_ip}-{dst_ip}-{sport}-{dport}-{proto}"
                
                # If no flows for this tuple, start new
                if not flow_dict[tuple_key]:
                    new_flow = {
                        'timestamps': [],
                        'is_super_flow': False,
                        'first_ts': timestamp
                    }
                    if needs_lengths:
                        new_flow['lengths'] = []
                    if needs_directions:
                        new_flow['directions'] = []
                    flow_dict[tuple_key].append(new_flow)
                
                current_flow = flow_dict[tuple_key][-1]
                prev_ts = current_flow['timestamps'][-1] if current_flow['timestamps'] else timestamp
                
                # Check for new flow conditions
                start_new_flow = False
                if (timestamp - prev_ts) > timeout_sec:
                    start_new_flow = True
                
                # Handle TCP FIN/RST
                is_fin_rst = False
                if TCP in pkt:
                    flags = pkt[TCP].flags
                    if flags & (0x01 | 0x04):  # FIN or RST
                        is_fin_rst = True
                
                # Check if current flow is super flow (use lengths size as proxy for feature length)
                length_key = 'lengths' if needs_lengths else 'timestamps'  # Fallback to timestamps
                current_len = len(current_flow.get(length_key, current_flow['timestamps']))
                if current_len >= max_flow_length and not current_flow['is_super_flow']:
                    current_flow['is_super_flow'] = True
                
                is_super_flow = current_flow['is_super_flow']
                
                if start_new_flow:
                    # Start new flow from current packet
                    new_flow = {
                        'timestamps': [],
                        'is_super_flow': False,
                        'first_ts': timestamp
                    }
                    if needs_lengths:
                        new_flow['lengths'] = []
                    if needs_directions:
                        new_flow['directions'] = []
                    flow_dict[tuple_key].append(new_flow)
                    current_flow = flow_dict[tuple_key][-1]
                    is_super_flow = False
                
                # Always update timestamp for timeout checks
                current_flow['timestamps'].append(timestamp)
                
                if is_super_flow:
                    # In super flow, do not update other data, but update timestamp (already done)
                    # If FIN/RST, start new flow (do not add current to new)
                    if is_fin_rst:
                        new_flow = {
                            'timestamps': [],
                            'is_super_flow': False,
                            'first_ts': None
                        }
                        if needs_lengths:
                            new_flow['lengths'] = []
                        if needs_directions:
                            new_flow['directions'] = []
                        flow_dict[tuple_key].append(new_flow)
                    continue  # Skip other updates
                
                # Update shared data based on needs
                if needs_lengths:
                    current_flow['lengths'].append(pkt_len)
                if needs_directions:
                    direction = 1 if sport > dport else -1  # Simplified assumption
                    current_flow['directions'].append(direction)
                
                # If FIN/RST and not super, start new after adding current
                if is_fin_rst and not is_super_flow:
                    new_flow = {
                        'timestamps': [],
                        'is_super_flow': False,
                        'first_ts': None
                    }
                    if needs_lengths:
                        new_flow['lengths'] = []
                    if needs_directions:
                        new_flow['directions'] = []
                    flow_dict[tuple_key].append(new_flow)
        
        # Post-process: generate features for each type
        all_results = {}
        for ft in feature_types:
            result = {}
            for tuple_key, flows in flow_dict.items():
                for idx, flow in enumerate(flows):
                    if not flow['timestamps']:
                        continue  # Skip empty
                    first_ts = flow['first_ts'] or flow['timestamps'][0] if flow['timestamps'] else 0
                    flow_id = f"{tuple_key}-{idx}-{int(first_ts * 1000)}"
                    
                    if ft == 'packet_length':
                        features = np.array(flow.get('lengths', []))
                    elif ft == 'direction':
                        features = np.array(flow.get('directions', []))
                    elif ft == 'inter_arrival_time':
                        ts_array = np.array(flow['timestamps'])
                        features = self.compute_iat(ts_array)
                    elif ft == 'up_down_rate':
                        ts_array = np.array(flow['timestamps'])
                        iat = self.compute_iat(ts_array)
                        lengths = np.array(flow.get('lengths', []))
                        directions = np.array(flow.get('directions', []))
                        epsilon = 1e-9
                        features = (lengths / (iat + epsilon)) * directions
                    
                    if len(features) > 0:
                        result[flow_id] = features
            
            all_results[ft] = result
            
            # Persist with writing flag
            output_dir = os.path.join('feature_matrix', ft)
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, os.path.basename(pcap_path) + '.npz')
            writing_flag = output_path + '.writing'
            open(writing_flag, 'w').close()  # Create empty writing flag file
            try:
                np.savez(output_path, **result)  # Save dict as compressed NPZ
            finally:
                os.remove(writing_flag)  # Remove flag after write, even if error
        
        if len(feature_types) == 1:
            return all_results[feature_types[0]]  # Simplify output if single type
        return all_results

    @njit
    def compute_iat(self, ts_array: np.ndarray) -> np.ndarray:
        """
        Compute IAT with Numba.
        """
        if len(ts_array) < 2:
            return np.array([0.0])
        iat = np.diff(ts_array)
        return np.concatenate(([0.0], iat))

# Usage example:
# config = {'pss': {'max_flow_length': 10000, 'timeout_sec': 64}}
# extractor = FeatureExtractor()
# features = extractor.extract_features('path/to/pcap.pcap', ['packet_length', 'inter_arrival_time'], config)
# # features = {'packet_length': {...}, 'inter_arrival_time': {...}}

if __name__ == '__main__':
    # Config for testing: set small max_flow_length to trigger truncation
    config = {
        'pss': {
            'max_flow_length': 5,  # Small value to test super flow truncation
            'timeout_sec': 64
        }
    }

    # Instantiate and call with all feature types
    extractor = FeatureExtractor()
    results = extractor.extract_features('/mnt/raid/luohaoran/cicids2018/SaP/phased_dataset_gen/tests/test.pcap', ['packet_length', 'inter_arrival_time', 'direction', 'up_down_rate'], config)

    # Print results for verification: for each feature, show flow IDs and sequence lengths
    for ft, res in results.items():
        print(f"Feature: {ft}")
        for flow_id, seq in res.items():
            print(f"  Flow ID: {flow_id}, Sequence length: {len(seq)}")