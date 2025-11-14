import argparse
import sys
import os
import numpy as np
from typing import Union, List, Dict
from scapy.all import PcapReader, IP, TCP
from collections import defaultdict
from numba import njit

@njit
def compute_iat(ts_array: np.ndarray) -> np.ndarray:
    """
    Compute IAT with Numba.
    """
    if len(ts_array) < 2:
        return np.array([0.0])
    iat = np.diff(ts_array)
    return np.concatenate((np.array([0.0]), iat))

class FeatureExtractor:
    """
    Class for extracting packet-level features from PCAP files.
    Handles flow separation based on CICFlowMeter standards and truncation.
    Supports features: packet_length, inter_arrival_time, direction, up_down_rate.
    Supports extracting multiple feature types in one pass.
    """
    def extract_features(self, pcap_path: str, feature_type: Union[str, List[str]], config: dict, output_base_dir: str = 'feature_matrix', store: bool = True) -> Dict[str, dict]:
        """
        Extract features for all flows in the PCAP, supporting single or multiple feature types.
        :param pcap_path: Path to PCAP file.
        :param feature_type: Single string or list of strings for feature types.
        :param config: Dict with 'pss' section containing 'max_flow_length' (default 1000), 'min_flow_length' (default 3) and 'timeout_sec' (default 64).
        :param output_base_dir: Base directory for output (default 'feature_matrix').
        :return: Dict {feature_type: {flow_id: np.array(feature_seq)}}, simplified if single type.
        """
        feature_types = [feature_type] if isinstance(feature_type, str) else feature_type
        supported_features = {'packet_length', 'inter_arrival_time', 'direction', 'up_down_rate'}
        if not set(feature_types).issubset(supported_features):
            raise ValueError(f"Unsupported features: {set(feature_types) - supported_features}")
        
        max_flow_length = config.get('pss', {}).get('max_flow_length', 1000)
        min_flow_length = config.get('pss', {}).get('min_flow_length', 3)
        timeout_sec = config.get('pss', {}).get('timeout_sec', 64)
        
        # Determine data needs based on features
        needs_lengths = any(ft in {'packet_length', 'up_down_rate'} for ft in feature_types)
        needs_directions = any(ft in {'direction', 'up_down_rate'} for ft in feature_types)
        needs_timestamps = any(ft in {'inter_arrival_time', 'up_down_rate'} for ft in feature_types)
        
        # Process PCAP and build flows
        flow_dict = self._process_pcap_and_build_flows(pcap_path, max_flow_length, timeout_sec, needs_lengths, needs_directions)
        
        # Post-process and generate results for each feature
        all_results = {}
        for ft in feature_types:
            result = self._post_process_flows_for_feature(flow_dict, ft, needs_lengths, needs_directions, min_flow_length)
            all_results[ft] = result
            print(f'{len(result)} flow records\' {ft} were writed to file {os.path.join(output_base_dir, ft)}')
            # If needed, save the result
            if store:
                self._save_feature_data(result, pcap_path, ft, output_base_dir)
        
        return all_results[feature_types[0]] if len(feature_types) == 1 else all_results

    def _process_pcap_and_build_flows(self, pcap_path: str, max_flow_length: int, timeout_sec: float, needs_lengths: bool, needs_directions: bool) -> defaultdict:
        """Process PCAP file iteratively and build flow dictionary with bidirectional aggregation."""
        flow_dict = defaultdict(list)
        with PcapReader(pcap_path) as reader:
            for pkt in reader:
                if IP not in pkt:
                    continue
                src_ip, dst_ip = pkt[IP].src, pkt[IP].dst
                proto = pkt[IP].proto
                sport = pkt.sport if hasattr(pkt, 'sport') else 0
                dport = pkt.dport if hasattr(pkt, 'dport') else 0

                if sport == 0 or dport == 0:
                    continue  # Skip non-TCP/UDP packets

                timestamp = float(pkt.time)
                pkt_len = len(pkt)
                # Normalize key for bidirectional: sort IPs, min-max ports
                ips = sorted([src_ip, dst_ip])
                ports = sorted([sport, dport])
                tuple_key = f"{ips[0]}-{ips[1]}-{ports[0]}-{ports[1]}-{proto}"
                
                # Initialize new flow if none exists
                if not flow_dict[tuple_key]:
                    self._init_flow(flow_dict[tuple_key], timestamp, needs_lengths, needs_directions, src_ip, sport)  # Pass src for direction init
                
                current_flow = flow_dict[tuple_key][-1]
                prev_ts = current_flow['last_check_ts'] if current_flow['last_check_ts'] is not None else timestamp
                
                # Check conditions and update flow
                self._update_flow(current_flow, pkt, timestamp, pkt_len, prev_ts, max_flow_length, timeout_sec, needs_lengths, needs_directions, flow_dict[tuple_key], src_ip, sport)
        
        return flow_dict

    def _init_flow(self, flows_list: list, timestamp: float, needs_lengths: bool, needs_directions: bool, init_src_ip: str, init_sport: int):
        """Initialize a new flow dictionary with required fields, including initial direction reference."""
        new_flow = {
            'timestamps': [],
            'is_super_flow': False,
            'first_ts': timestamp,
            'last_check_ts': timestamp if timestamp is not None else None,
            'init_src_ip': init_src_ip,  # For determining forward direction
            'init_sport': init_sport
        }
        if needs_lengths:
            new_flow['lengths'] = []
        if needs_directions:
            new_flow['directions'] = []
        flows_list.append(new_flow)

    def _update_flow(self, current_flow: dict, pkt, timestamp: float, pkt_len: int, prev_ts: float, max_flow_length: int, timeout_sec: float, needs_lengths: bool, needs_directions: bool, flows_list: list, curr_src_ip: str, curr_sport: int):
        """Update current flow with packet data, handling super flow and new flow starts, with bidirectional direction."""
        start_new_flow = (timestamp - prev_ts) > timeout_sec
        is_fin_rst = False
        if TCP in pkt:
            flags = pkt[TCP].flags
            if flags & (0x01 | 0x04):  # FIN or RST
                is_fin_rst = True
        
        # Check super flow using lengths size as proxy (or timestamps if no lengths)
        length_key = 'lengths' if needs_lengths else 'timestamps'
        current_len = len(current_flow.get(length_key, current_flow['timestamps']))
        if current_len >= max_flow_length and not current_flow['is_super_flow']:
            current_flow['is_super_flow'] = True
        
        is_super_flow = current_flow['is_super_flow']
        
        if start_new_flow:
            self._init_flow(flows_list, timestamp, needs_lengths, needs_directions, curr_src_ip, curr_sport)
            current_flow = flows_list[-1]
            is_super_flow = False
        
        # Always update last_check_ts for future checks
        current_flow['last_check_ts'] = timestamp
        
        if is_super_flow:
            if is_fin_rst:
                self._init_flow(flows_list, timestamp, needs_lengths, needs_directions, curr_src_ip, curr_sport)  # Start new from current pkt if FIN in super
            return  # Skip data updates
        
        # Update timestamps and other data if not super flow
        current_flow['timestamps'].append(timestamp)
        if needs_lengths:
            current_flow['lengths'].append(pkt_len)
        if needs_directions:
            # Determine direction relative to initial packet
            is_forward = (curr_src_ip == current_flow['init_src_ip'] and curr_sport == current_flow['init_sport'])
            direction = 1 if is_forward else -1
            current_flow['directions'].append(direction)
        
        if is_fin_rst and not is_super_flow:
            self._init_flow(flows_list, None, needs_lengths, needs_directions, None, None)  # None for new flow init

    def _post_process_flows_for_feature(self, flow_dict: defaultdict, ft: str, needs_lengths: bool, needs_directions: bool, min_flow_length: int) -> dict:
        """Post-process flows to compute feature-specific sequences, filtering short flows."""
        result = {}
        for tuple_key, flows in flow_dict.items():
            for idx, flow in enumerate(flows):
                if not flow['timestamps'] or len(flow['timestamps']) < min_flow_length:
                    continue  # Skip empty or short flows
                first_ts = flow['first_ts'] or flow['timestamps'][0] if flow['timestamps'] else 0
                flow_id = f"{tuple_key}-{idx}-{int(first_ts * 1000)}"
                
                if ft == 'packet_length':
                    features = np.array(flow.get('lengths', []))
                elif ft == 'direction':
                    features = np.array(flow.get('directions', []))
                elif ft == 'inter_arrival_time':
                    ts_array = np.array(flow['timestamps'])
                    features = compute_iat(ts_array)
                elif ft == 'up_down_rate':
                    ts_array = np.array(flow['timestamps'])
                    iat = compute_iat(ts_array)
                    lengths = np.array(flow.get('lengths', []))
                    directions = np.array(flow.get('directions', []))
                    epsilon = 1e-6  # To avoid division by zero
                    features = (lengths / (iat + epsilon)) * directions 
                    features[0] = 0 # First packet IAT is zero, so rate is set to 0
                
                if len(features) > 0:
                    result[flow_id] = features
        return result

    def _save_feature_data(self, result: dict, pcap_path: str, ft: str, output_base_dir: str) -> str:
        """Save feature data to .npz with writing flag for integrity."""
        output_dir = os.path.join(output_base_dir, ft)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, os.path.basename(pcap_path) + '.npz')
        writing_flag = output_path + '.writing'
        success = False
        open(writing_flag, 'w').close()  # Create writing flag
        try:
            np.savez(output_path, **result)
            success = True
        finally:
            if success:
                os.remove(writing_flag)  # Remove flag after save
        return output_path

# Usage example:
# config = {'pss': {'max_flow_length': 1000, 'min_flow_length': 3, 'timeout_sec': 64}}
# extractor = FeatureExtractor()
# features = extractor.extract_features('path/to/pcap.pcap', ['packet_length', 'inter_arrival_time'], config, output_base_dir='custom_matrix')
# # features = {'packet_length': {...}, 'inter_arrival_time': {...}}

if __name__ == '__main__':
    # Config for testing: set small max_flow_length to trigger truncation
    config = {
        'pss': {
            'max_flow_length': 50,  # Small value to test super flow truncation
            'min_flow_length': 3,
            'timeout_sec': 64
        }
    }

    # parse command-line parameters and optionally run immediately

    parser = argparse.ArgumentParser(description='Extract features from a PCAP file.')
    parser.add_argument('-p', '--pcap', type=str, help='Path to PCAP file (if omitted, script will continue to the hard-coded call below).')
    parser.add_argument('-f', '--features', type=str, default='packet_length,inter_arrival_time,direction,up_down_rate',
                        help='Comma-separated list of feature types (packet_length, inter_arrival_time, direction, up_down_rate).')
    parser.add_argument('-o', '--output', type=str, default='workspace/test/feature_matrix',
                        help='Output base directory for feature matrices.')
    parser.add_argument('--max_flow_length', type=int, default=config['pss']['max_flow_length'],
                        help='Maximum flow length (super flow threshold).')
    parser.add_argument('--min_flow_length', type=int, default=config['pss']['min_flow_length'],
                        help='Minimum flow length to keep.')
    parser.add_argument('--timeout_sec', type=float, default=config['pss']['timeout_sec'],
                        help='Flow timeout in seconds.')
    parser.add_argument('--run', action='store_true', help='If set and --pcap provided, run extraction now and exit (skips the hard-coded call below).')

    args = parser.parse_args()

    # apply CLI config overrides
    config['pss']['max_flow_length'] = args.max_flow_length
    config['pss']['min_flow_length'] = args.min_flow_length
    config['pss']['timeout_sec'] = args.timeout_sec

    feature_list = [ft.strip() for ft in args.features.split(',') if ft.strip()]

    if args.pcap and args.run:
        extractor = FeatureExtractor()
        results = extractor.extract_features(args.pcap, feature_list, config, args.output)
        # print brief summary
        if isinstance(results, dict):
            for ft, res in (results.items() if isinstance(results, dict) and any(isinstance(v, dict) for v in results.values()) else [(feature_list[0], results)]):
                print(f'Feature "{ft}": {len(res)} flows extracted, saved under {os.path.join(args.output, ft)}')
        sys.exit(0)

        # Print results for verification: for each feature, show flow IDs and sequence lengths
        # for ft, res in results.items():
        #     print(f"Feature: {ft}")
        #     for flow_id, seq in res.items():
        #         print(f"  Flow ID: {flow_id}, Sequence length: {len(seq)}")