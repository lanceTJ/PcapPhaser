# 
import numpy as np
from numba import njit
import pickle
import os
from scapy.all import sniff, IP, TCP, UDP  # For streaming PCAP parsing
from utils import load_config

# Numba-accelerated function to update U (mean) and M (cumulative second moment) matrices
@njit
def update_matrices(feature_seq, long_threshold):
    """
    Update U and M matrices using Welford's method for variance computation.
    Truncate sequence to long_threshold to avoid O(n^2) explosion.
    """
    n = min(len(feature_seq), long_threshold)
    U = np.zeros((n, n), dtype=np.float64)
    M = np.zeros((n, n), dtype=np.float64)
    for t in range(n):
        U[t, t] = feature_seq[t]
        M[t, t] = 0.0
        for s in range(t - 1, -1, -1):
            delta = feature_seq[t] - U[s, t - 1]
            seg_len = t - s + 1
            U[s, t] = U[s, t - 1] + delta / seg_len
            M[s, t] = M[s, t - 1] + delta * (feature_seq[t] - U[s, t])
    return U, M

# Feature extraction example (packet length); can be swapped via parameter
def extract_packet_length(pkt):
    """Extract full packet length including header and payload."""
    return len(pkt)

# Main class for SingleFeatureMatrixBuilder module
class SingleFeatureMatrixBuilder:
    def __init__(self, config_path='config.ini', feature_extractor=extract_packet_length):
        self.config = load_config(config_path)  # Load thresholds from config
        self.feature_extractor = feature_extractor

    def process_pcap(self, pcap_path, output_dir, feature_name):
        """
        Process PCAP streamingly: group packets into flows with sub-flows based on FIN/RST and timeout.
        Extract feature sequence, update matrices if length >= short_threshold, persist in batches.
        """
        from collections import defaultdict
        flows = defaultdict(list)  # key: 5-tuple, value: list of flow_dicts {'feature_seq': [], 'last_time': None}
        terminated_flows = {}  # Batch for matrices {flow_id_sub_id: {'U': U, 'M': M}}

        # Streaming packet processing
        def process_packet(pkt):
            key = self.get_flow_key(pkt)
            if key is None:
                return
            current_time = float(pkt.time)
            feature_value = self.feature_extractor(pkt)

            # If no sub-flows yet, create first one
            if not flows[key]:
                flows[key].append({'feature_seq': [], 'last_time': None})

            last_flow = flows[key][-1]
            if last_flow['last_time'] is not None:
                delta_time = current_time - last_flow['last_time']
                if delta_time > self.config['timeout']:
                    # Timeout: create new sub-flow and add current packet to it
                    flows[key].append({'feature_seq': [feature_value], 'last_time': current_time})
                    return

            # Add to last sub-flow and update last_time
            last_flow['feature_seq'].append(feature_value)
            last_flow['last_time'] = current_time

            # Check TCP termination (FIN or RST): add to current, then create new empty sub-flow
            if TCP in pkt and (pkt[TCP].flags.F or pkt[TCP].flags.R):
                flows[key].append({'feature_seq': [], 'last_time': None})  # New empty sub-flow for potential continuation

        sniff(offline=pcap_path, prn=process_packet, store=False)

        # Post-process: terminate all sub-flows, discard incomplete timeouts or shorts
        for key in list(flows.keys()):
            for sub_id, sub_flow in enumerate(flows[key]):
                seq = sub_flow['feature_seq']
                if len(seq) < self.config['short_threshold']:
                    continue  # Discard short sub-flows
                # Check if this sub-flow is timed out (only for unfinished ones)
                if sub_flow['last_time'] is not None and (float(os.path.getmtime(pcap_path)) - sub_flow['last_time'] > self.config['timeout']):
                    continue  # Discard timed-out unfinished sub-flows (simulated end time via file mtime)
                U, M = update_matrices(np.array(seq), self.config['long_threshold'])
                terminated_flows[f"{key}_sub{sub_id + 1}"] = {'U': U, 'M': M}
                # Batch persist
                if len(terminated_flows) >= self.config['batch_size']:
                    self.persist_batch(terminated_flows, output_dir, feature_name)
                    terminated_flows.clear()

        # Persist any remaining batch
        if terminated_flows:
            self.persist_batch(terminated_flows, output_dir, feature_name)

    def persist_batch(self, batch, output_dir, feature_name):
        """Persist U and M matrices to subdirs as pickle files."""
        u_dir = os.path.join(output_dir, feature_name, 'U')
        m_dir = os.path.join(output_dir, feature_name, 'M')
        os.makedirs(u_dir, exist_ok=True)
        os.makedirs(m_dir, exist_ok=True)
        for flow_id, matrices in batch.items():
            with open(os.path.join(u_dir, f"{flow_id}.pkl"), 'wb') as f:
                pickle.dump(matrices['U'], f)
            with open(os.path.join(m_dir, f"{flow_id}.pkl"), 'wb') as f:
                pickle.dump(matrices['M'], f)

    @staticmethod
    def get_flow_key(pkt):
        """Generate normalized 5-tuple flow key for bidirectional flows."""
        if IP not in pkt:
            return None
        src_ip, dst_ip = pkt[IP].src, pkt[IP].dst
        proto = pkt[IP].proto
        if TCP in pkt:
            sport, dport = pkt[TCP].sport, pkt[TCP].dport
        elif UDP in pkt:
            sport, dport = pkt[UDP].sport, pkt[UDP].dport
        else:
            return None
        if src_ip > dst_ip:
            return f"{dst_ip}-{src_ip}-{dport}-{sport}-{proto}"
        return f"{src_ip}-{dst_ip}-{sport}-{dport}-{proto}"

# Usage example
if __name__ == "__main__":
    builder = SingleFeatureMatrixBuilder()
    builder.process_pcap('input.pcap', 'feature_matrix', 'packet_length')