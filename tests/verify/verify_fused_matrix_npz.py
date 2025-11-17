import os
import numpy as np
import argparse

# Parse command-line arguments for flexibility
parser = argparse.ArgumentParser(description="Verify and print contents of .npz files from fused matrices.")
parser.add_argument('--feature_set', type=str, required=True, help="Feature set (e.g., 'feature_set_1')")
parser.add_argument('--npz_name', type=str, default='merged_matrix_fused.npz', help="Fused .npz name (default: merged_matrix_fused.npz)")
parser.add_argument('--base_dir', type=str, default='datasets', help="Base directory for datasets (default: datasets)")
parser.add_argument('--display_matrix_size', type=int, default=5, help="Display full matrix if size <= this value (default: 5, over 10 is not recommended)")
args = parser.parse_args()

# Construct the full path to the .npz file
npz_path = os.path.join(args.base_dir, args.feature_set, 'merged_matrix', args.npz_name)

# Displayed matrix size threshold
display_size = args.display_matrix_size

# Check if the file exists
if not os.path.exists(npz_path):
    print(f"Error: File {npz_path} does not exist.")
    exit(1)

# Load the .npz file (allow_pickle=True since it's a dict of arrays)
data = np.load(npz_path, allow_pickle=True)

# Print summary for each flow_id
print(f"Loaded {npz_path} with {len(data)} flows.")
for flow_id in data:
    fused_j = data[flow_id]  # Directly the fused J array
    n = fused_j.shape[0]
    print(f"Flow ID: {flow_id}")
    print(f"  Fused J shape: {fused_j.shape}")
    if n <= 5:
        print(f"  Full fused J matrix:\n{fused_j}")
    else:
        print(f"  First {display_size}x{display_size} of fused J:\n{fused_j[:display_size, :display_size]}")
        print(f"  Last {display_size}x{display_size} of fused J:\n{fused_j[-display_size:, -display_size:]}")
    # Verify upper triangular (should be zero below diagonal)
    below_diag = fused_j[np.tril_indices(n, k=-1)]
    if np.allclose(below_diag, 0):
        print("  Fused J is upper triangular as expected.")
    else:
        print("  Warning: Fused J has non-zero values below diagonal.")
    # Basic regularization check: diagonal should be positive if lambda added
    diag = np.diag(fused_j)
    if np.all(diag > 0):
        print("  Diagonal values are positive (regularization likely applied).")
    else:
        print("  Warning: Some diagonal values <= 0 (check regularization).")
    print("-" * 40)