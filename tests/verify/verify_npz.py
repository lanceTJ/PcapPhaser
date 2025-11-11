import os
import numpy as np
import argparse

# Parse command-line arguments for flexibility
parser = argparse.ArgumentParser(description="Verify and print contents of .npz files from feature_matrix.")
parser.add_argument('--feature_type', type=str, required=True, help="Feature type (e.g., 'packet_length')")
parser.add_argument('--pcap_name', type=str, default='test.npz', help="PCAP name with .npz extension (default: test.npz)")
parser.add_argument('--base_dir', type=str, default='feature_matrix', help="Base directory for feature_matrix (default: feature_matrix)")
args = parser.parse_args()

# Construct the full path to the .npz file
npz_path = os.path.join(args.base_dir, args.feature_type, args.pcap_name)

# Check if the file exists
if not os.path.exists(npz_path):
    print(f"Error: File {npz_path} does not exist.")
    exit(1)

# Load the .npz file (allow_pickle=True since it's a dict)
data = np.load(npz_path, allow_pickle=True)

# Print summary for each flow_id
print(f"Loaded {npz_path} with {len(data)} flows.")
for flow_id in data:
    seq = data[flow_id]
    seq_len = len(seq)
    print(f"Flow ID: {flow_id}")
    print(f"  Sequence length: {seq_len}")
    if seq_len <= 10:
        print(f"  Full sequence: {seq}")
    else:
        print(f"  First 5 elements: {seq[:5]}")
        print(f"  Last 5 elements: {seq[-5:]}")
    print("-" * 40)



# python verify_npz.py --feature_type packet_length --pcap_name test.pcap.npz