import os
import numpy as np
import argparse

# Parse command-line arguments for flexibility
parser = argparse.ArgumentParser(description="Verify and print contents of .npz files from feature_matrix matrices.")
parser.add_argument('--feature_type', type=str, required=True, help="Feature type (e.g., 'packet_length')")
parser.add_argument('--npz_name', type=str, default='packet_length_matrices.npz', help="Matrices .npz name (default: packet_length_matrices.npz)")
parser.add_argument('--base_dir', type=str, default='feature_matrix', help="Base directory for feature_matrix (default: feature_matrix)")
parser.add_argument('--display_matrix_size', type=int, default=5, help="Display full matrix if size <= this value (default: 5, over 10 is not recommended)")
args = parser.parse_args()

# Construct the full path to the .npz file
npz_path = os.path.join(args.base_dir, args.feature_type, args.npz_name)

# Displayed matrix size threshold
display_size = args.display_matrix_size

# Check if the file exists
if not os.path.exists(npz_path):
    print(f"Error: File {npz_path} does not exist.")
    exit(1)

# Load the .npz file (allow_pickle=True since it's a nested dict)
data = np.load(npz_path, allow_pickle=True)

# Print summary for each flow_id
print(f"Loaded {npz_path} with {len(data)} flows.")
for flow_id in data:
    mats = data[flow_id].item()  # Get the inner dict {'U': array, 'M': array, 'J': array}
    print(f"Flow ID: {flow_id}")
    for mat_name in ['U', 'M', 'J']:
        mat = mats[mat_name]
        n = mat.shape[0]
        print(f"  Matrix {mat_name} shape: {mat.shape}")
        if n <= 5:
            print(f"  Full {mat_name} matrix:\n{mat}")
        else:
            print(f"  First {display_size}x{display_size} of {mat_name}:\n{mat[:display_size, :display_size]}")
            print(f"  Last {display_size}x{display_size} of {mat_name}:\n{mat[-display_size:, -display_size:]}")
            # Verify upper triangular for J (should be zero below diagonal)
            if mat_name == 'J':
                below_diag = mat[np.tril_indices(n, k=-1)]
                if np.allclose(below_diag, 0):
                    print("  J is upper triangular as expected.")
                else:
                    print("  Warning: J has non-zero values below diagonal.")
    print("-" * 40)