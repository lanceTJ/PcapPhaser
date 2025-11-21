import argparse
import sys
import os
import json
from typing import Dict, List
from collections import defaultdict
from scapy.all import rdpcap, wrpcap, PcapReader, IP, TCP, UDP  # For pcap reading/writing; assume available or simulate via code tool

class PhaseReconstructor:
    """
    Class for reconstructing phased pcap files from original pcap and phase marks.
    Outputs per-phase pcap files containing packets for that phase across all flows.
    Supports saving with integrity flag.
    """
    def __init__(self, config: dict = None):
        """
        :param config: Dict with 'pss' section containing optional 'num_phases' (int, default 4),
                       'max_flow_length' (int, default 1000), 'min_flow_length' (int, default 3),
                       'timeout_sec' (float, default 600).
        """
        D_num_phases = 4
        D_max_flow_length = 1000
        D_min_flow_length = 3
        D_timeout_sec = 600
        if config is not None:
            pss = config.get('pss', {})
            self.num_phases = pss.get('num_phases', D_num_phases)
            self.max_flow_length = pss.get('max_flow_length', D_max_flow_length)
            self.min_flow_length = pss.get('min_flow_length', D_min_flow_length)
            self.timeout_sec = pss.get('timeout_sec', D_timeout_sec)
        else:
            self.num_phases = D_num_phases
            self.max_flow_length = D_max_flow_length
            self.min_flow_length = D_min_flow_length
            self.timeout_sec = D_timeout_sec

    def reconstruct_phases(self, phase_marks: Dict[str, List[int]], original_pcap_path: str, output_base_dir: str = 'datasets/feature_set_n/phased_pcap', store_file_base: str = 'default', store: bool = True) -> Dict[int, str]:
        """
        Reconstruct phased pcaps from original pcap and marks.
        :param phase_marks: Dict {flow_id: [int, ...]} with 1-based phase end marks from PhaseDivider.
        :param original_pcap_path: Path to original pcap file.
        :param output_base_dir: Base directory for output.
        :param store_file_base: Base name for storing phased pcaps (default 'default').
        :param store: Whether to store the results to disk (default True).
        :return: Dict {phase_num: output_path} for each phased pcap.
        """
        # Process PCAP and build flows with splitting logic
        flow_dict = self._process_pcap_and_build_flows(original_pcap_path, self.max_flow_length, self.timeout_sec)

        # Build flow_packets {flow_id: [pkt]} with unified float ts format
        flow_packets = {}
        for tuple_key, flows in flow_dict.items():
            for i, flow in enumerate(flows, start=0):
                if not flow['timestamps']:
                    continue  # Skip empty
                first_ts = flow['first_ts'] or flow['timestamps'][0]
                std_time = f'{first_ts:.6f}'[:-3].replace('.', '')  # e.g., 1622547800123456
                flow_id = f"{tuple_key}-{i}-{std_time}"
                pkts = flow['packets']
                flow_packets[flow_id] = pkts

        # Prepare per-phase packet lists
        phase_lists = defaultdict(list)  # {phase_num: [pkt, ...]}

        for flow_id, pkts in flow_packets.items():
            if flow_id in phase_marks:
                marks = phase_marks[flow_id]
                starts = [0] + marks
                ends = marks + [len(pkts)]
                for ph in range(1, self.num_phases + 1):
                    start_idx = starts[ph - 1]
                    end_idx = ends[ph - 1]
                    phase_lists[ph].extend(pkts[start_idx:end_idx])
            else:
                # Short flows or no marks: all in phase 1
                phase_lists[1].extend(pkts)

        # Save phased pcaps if store
        output_paths = {}
        if store:
            os.makedirs(output_base_dir, exist_ok=True)
            for ph in range(1, self.num_phases + 1):
                output_path = os.path.join(output_base_dir, f'phase_{ph}', f'p_{ph}_{store_file_base}.pcap')
                output_paths[ph] = self._save_phased_pcap(phase_lists[ph], output_path)

        print(f'Reconstructed {len(output_paths)} phased pcaps from {original_pcap_path} under {output_base_dir}')
        return output_paths

    def _process_pcap_and_build_flows(self, pcap_path: str, max_flow_length: int, timeout_sec: float) -> defaultdict:
        """Process PCAP file iteratively and build flow dictionary with bidirectional aggregation and packets."""
        flow_dict = defaultdict(list)
        needs_lengths = False
        needs_directions = False
        with PcapReader(pcap_path) as reader:
            for pkt in reader:
                if IP not in pkt:
                    continue
                src_ip, dst_ip = pkt[IP].src, pkt[IP].dst
                proto = pkt[IP].proto
                if proto not in (6, 17):  # TCP/UDP only
                    continue
                sport = pkt.sport if hasattr(pkt, 'sport') else 0
                dport = pkt.dport if hasattr(pkt, 'dport') else 0
                if sport == 0 or dport == 0:
                    continue
                timestamp = float(pkt.time)
                pkt_len = len(pkt)
                # Normalize key: sort IPs and swap ports accordingly, then sort ports independently to match FeatureExtractor
                if src_ip > dst_ip:
                    src_ip, dst_ip = dst_ip, src_ip
                    sport, dport = dport, sport
                ports = sorted([sport, dport])
                tuple_key = f"{src_ip}-{dst_ip}-{ports[0]}-{ports[1]}-{proto}"

                # Initialize new flow if none
                if not flow_dict[tuple_key]:
                    self._init_flow(flow_dict[tuple_key], timestamp, needs_lengths, needs_directions, src_ip, sport)

                current_flow = flow_dict[tuple_key][-1]
                prev_ts = current_flow['last_check_ts'] if current_flow['last_check_ts'] is not None else timestamp

                # Check conditions and update flow
                self._update_flow(current_flow, pkt, timestamp, pkt_len, prev_ts, max_flow_length, timeout_sec, needs_lengths, needs_directions, flow_dict[tuple_key], src_ip, sport)

        return flow_dict

    def _init_flow(self, flows_list: list, timestamp: float, needs_lengths: bool, needs_directions: bool, init_src_ip: str, init_sport: int):
        """Initialize a new flow dictionary with required fields, including packets."""
        new_flow = {
            'timestamps': [],
            'packets': [],
            'is_super_flow': False,
            'first_ts': timestamp,
            'last_check_ts': timestamp if timestamp is not None else None,
        }
        if needs_directions:
            new_flow['init_src_ip'] = init_src_ip
            new_flow['init_sport'] = init_sport
        if needs_lengths:
            new_flow['lengths'] = []
        if needs_directions:
            new_flow['directions'] = []
        flows_list.append(new_flow)

    def _update_flow(self, current_flow: dict, pkt, timestamp: float, pkt_len: int, prev_ts: float, max_flow_length: int, timeout_sec: float, needs_lengths: bool, needs_directions: bool, flows_list: list, curr_src_ip: str, curr_sport: int):
        """Update current flow with packet data, handling super flow, new flow starts, and FIN bug fix."""
        start_new_flow = (timestamp - prev_ts) > timeout_sec
        is_fin_rst = False
        if TCP in pkt:
            flags = pkt[TCP].flags
            if flags & (0x01 | 0x04):  # FIN or RST
                is_fin_rst = True

        # Check for super flow using timestamps len (since no lengths needed)
        length_key = 'timestamps'  # Always use timestamps for recon
        current_len = len(current_flow[length_key])
        if current_len >= max_flow_length and not current_flow['is_super_flow']:
            current_flow['is_super_flow'] = True

        is_super_flow = current_flow['is_super_flow']

        if start_new_flow:
            self._init_flow(flows_list, timestamp, needs_lengths, needs_directions, curr_src_ip, curr_sport)
            current_flow = flows_list[-1]
            is_super_flow = False

        # Always update last_check_ts
        current_flow['last_check_ts'] = timestamp

        # If super flow and not FIN, skip adding
        if is_super_flow and not is_fin_rst:
            return

        # Add data (for FIN in super, add to current)
        current_flow['timestamps'].append(timestamp)
        current_flow['packets'].append(pkt)
        if needs_lengths:
            current_flow['lengths'].append(pkt_len)
        if needs_directions:
            is_forward = (curr_src_ip == current_flow.get('init_src_ip') and curr_sport == current_flow.get('init_sport'))
            direction = 1 if is_forward else -1
            current_flow['directions'].append(direction)

        # After adding, if FIN/RST, start new flow
        if is_fin_rst:
            self._init_flow(flows_list, None, needs_lengths, needs_directions, None, None)
            # No need to set current_flow, next pkt will use the old until new condition

    def _save_phased_pcap(self, pkts: List, output_path: str) -> str:
        """
        Save list of packets to pcap with writing flag for integrity.
        """
        writing_flag = output_path + '.writing'
        success = False
        open(writing_flag, 'w').close()  # Create writing flag
        try:
            wrpcap(output_path, pkts)
            success = True
        finally:
            if success:
                os.remove(writing_flag)  # Remove flag after save
        return output_path

# Usage example:
# from PhaseDivider import PhaseDivider
# config = {'pss': {'num_phases': 4}}
# divider = PhaseDivider(config)
# marks = divider.divide_phases(fused_data)
# recon = PhaseReconstructor(config)
# phased_paths = recon.reconstruct_phases(marks, 'path/to/original.pcap')

if __name__ == '__main__':
    # Config for testing
    config = {'pss': {'num_phases': 4, 'max_flow_length': 1000, 'min_flow_length': 3, 'timeout_sec': 600}}

    parser = argparse.ArgumentParser(description='Reconstruct phased pcaps from marks and original pcap.')
    parser.add_argument('-p', '--pcap', type=str, required=True, help='Path to original pcap file.')
    parser.add_argument('-m', '--marks_json', type=str, required=True, help='Path to phase_marks .json from PhaseDivider.')
    parser.add_argument('-o', '--output', type=str, default='workspace/test/datasets/feature_set_1/phased_pcap', help='Output base directory.')
    parser.add_argument('--num_phases', type=int, default=config['pss']['num_phases'], help='Number of phases (for config update).')
    parser.add_argument('--run', action='store_true', help='Run reconstruction now if inputs provided.')

    import time
    start_time = time.time()
    print(f'------Starting reconstruction at {time.ctime(start_time)}')
    args = parser.parse_args()
    if args.pcap and args.marks_json and args.run:
        with open(args.marks_json, 'r') as f:
            phase_marks = json.load(f)
        config['pss']['num_phases'] = args.num_phases  # Update config
        recon = PhaseReconstructor(config)
        basename = os.path.basename(args.pcap)[:-5]  # Remove .pcap
        results = recon.reconstruct_phases(phase_marks, args.pcap, args.output, basename, store=True)
        print(f'Reconstructed {len(results)} phased pcaps, saved under {args.output}')
        sys.exit(0)
    end_time = time.time()
    print(f'-------Total time: {end_time - start_time:.2f} seconds')