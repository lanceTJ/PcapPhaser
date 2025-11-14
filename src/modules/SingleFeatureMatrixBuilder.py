import argparse
import sys
import os
import numpy as np
from typing import Dict
from collections import defaultdict
from numba import njit, prange

@njit
def welford_update(existing_count: int, existing_mean: float, existing_m2: float, new_value: float) -> tuple:
    """
    Welford's online update for mean and M2 (sum of squared differences).
    Returns updated count, mean, M2.
    """
    count = existing_count + 1
    delta = new_value - existing_mean
    mean = existing_mean + delta / count
    delta2 = new_value - mean
    m2 = existing_m2 + delta * delta2
    return count, mean, m2

@njit(parallel=True)
def compute_matrices(seq: np.ndarray, max_len: int) -> tuple:
    """
    Compute upper triangular matrices U (mean), M (M2), J (normalized variance) for sequence using Welford.
    Limits to max_len if longer.
    """
    n = min(len(seq), max_len)
    U = np.zeros((n, n), dtype=np.float64)
    M = np.zeros((n, n), dtype=np.float64)
    J = np.zeros((n, n), dtype=np.float64)
    
    for s in prange(n):
        count, mean, m2 = 0, 0.0, 0.0
        for t in range(s, n):
            count, mean, m2 = welford_update(count, mean, m2, seq[t])
            U[s, t] = mean
            M[s, t] = m2
            if t > s:
                J[s, t] = m2 / (t - s)  # Normalized variance as dissimilarity score
    
    return U, M, J

class SingleFeatureMatrixBuilder:
    """
    Class for building single-feature matrices from feature sequences.
    Computes U (mean), M (M2), J (dissimilarity) using Welford's method with Numba acceleration.
    Supports saving to .npz with integrity flag.
    """
    def build_matrices(self, feature_data: Dict[str, np.ndarray], feature_type: str, config: dict, output_base_dir: str = 'feature_matrix') -> Dict[str, dict]:
        """
        Build matrices for all flows in the feature data.
        :param feature_data: Dict {flow_id: np.array(seq)} from FeatureExtractor.
        :param feature_type: String for feature type (e.g., 'packet_length').
        :param config: Dict with 'pss' section containing 'max_flow_length' (default 1000).
        :param output_base_dir: Base directory for output (default 'feature_matrix').
        :return: Dict {flow_id: {'U': np.array, 'M': np.array, 'J': np.array}}.
        """
        max_flow_length = config.get('pss', {}).get('max_flow_length', 1000)
        
        results = {}
        for flow_id, seq in feature_data.items():
            if len(seq) < 2:
                continue  # Skip too short sequences
            U, M, J = compute_matrices(seq, max_flow_length)
            results[flow_id] = {'U': U, 'M': M, 'J': J}
        
        print(f'{len(results)} flow matrices for {feature_type} were computed and saved to {os.path.join(output_base_dir, feature_type)}')
        self._save_matrices(results, feature_type, output_base_dir)
        
        return results

    def _save_matrices(self, results: Dict[str, dict], feature_type: str, output_base_dir: str) -> str:
        """
        Save matrices to .npz with writing flag for integrity.
        """
        output_dir = os.path.join(output_base_dir, feature_type)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'{feature_type}_matrices.npz')
        writing_flag = output_path + '.writing'
        success = False
        open(writing_flag, 'w').close()  # Create writing flag
        try:
            save_dict = defaultdict(dict)
            for flow_id, mats in results.items():
                for mat_name, mat in mats.items():
                    save_dict[flow_id][mat_name] = mat
            np.savez(output_path, **save_dict)
            success = True
        finally:
            if success:
                os.remove(writing_flag)  # Remove flag after save
        return output_path

# Usage example:
# from FeatureExtractor import FeatureExtractor
# config = {'pss': {'max_flow_length': 1000}}
# extractor = FeatureExtractor()
# feature_data = extractor.extract_features('path/to/pcap.pcap', 'packet_length', config)['packet_length']
# builder = SingleFeatureMatrixBuilder()
# matrices = builder.build_matrices(feature_data, 'packet_length', config)

if __name__ == '__main__':
    # Config for testing
    config = {'pss': {'max_flow_length': 50}}  # Small value to test truncation

    parser = argparse.ArgumentParser(description='Build matrices from feature data.')
    parser.add_argument('-f', '--feature_type', type=str, required=True, help='Feature type (e.g., packet_length).')
    parser.add_argument('-i', '--input_npz', type=str, required=True, help='Path to input .npz from FeatureExtractor.')
    parser.add_argument('-o', '--output', type=str, default='workspace/test/feature_matrix', help='Output base directory.')
    parser.add_argument('--max_flow_length', type=int, default=config['pss']['max_flow_length'], help='Max sequence length for matrices.')
    parser.add_argument('--run', action='store_true', help='Run building now if input provided.')

    args = parser.parse_args()
    config['pss']['max_flow_length'] = args.max_flow_length

    if args.input_npz and args.run:
        feature_data = np.load(args.input_npz, allow_pickle=True)
        feature_data = {k: v for k, v in feature_data.items()}  # Convert to dict
        builder = SingleFeatureMatrixBuilder()
        results = builder.build_matrices(feature_data, args.feature_type, config, args.output)
        print(f'Feature "{args.feature_type}": {len(results)} flow matrices built, saved under {os.path.join(args.output, args.feature_type)}')
        sys.exit(0)