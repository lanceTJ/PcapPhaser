import argparse
import sys
import os
import ast
import numpy as np
from typing import Dict
from collections import defaultdict
from numba import njit, prange

@njit(parallel=True, fastmath=True)
def compute_matrices(seq: np.ndarray, max_len: int, lambda_value: float = 1e-3) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Compute U (mean), M (M2), J (dissimilarity score) matrices using Welford's method.
    J[s,t] = 1 / ( (M[s,t] / (t-s)) + lambda_value ) for t > s; for t == s, J = 0.0 (undefined in formula).
    Only upper triangular (t >= s) is computed; lower is zero.
    """
    n = min(len(seq), max_len)
    U = np.zeros((n, n), dtype=np.float64)
    M = np.zeros((n, n), dtype=np.float64)
    J = np.zeros((n, n), dtype=np.float64)

    for s in prange(n):
        count = 0
        mean = 0.0
        m2 = 0.0

        for t in range(s, n):
            x = seq[t]
            count += 1
            delta = x - mean
            mean += delta / count
            delta2 = x - mean
            m2 += delta * delta2

            U[s, t] = mean
            M[s, t] = m2
            if t > s:
                variance = m2 / (t - s) if (t - s) > 0 else 0.0
                J[s, t] = 1.0 / (variance + lambda_value)
            else:
                J[s, t] = 0.0  # Set diagonal to 0 as per output

    return U, M, J

class SingleFeatureMatrixBuilder:
    """
    Class for building single-feature matrices from feature sequences.
    Computes U (mean), M (M2), J (dissimilarity) using Welford's method with Numba acceleration.
    Supports saving to .npz with integrity flag.
    """

    def __init__(self, config: dict = None):
        """
        :param config: Dict with 'pss' section containing optional 'allowed_feature_names' (list of str), 'lambda_dict' (dict of str to float), and 'max_flow_length' (int, default 1000).
        """
        # Read allowed_feature_names, lambda_dict and max_flow_length from config if provided
        D_allowed_feature_names = {'packet_length', 'inter_arrival_time', 'up_down_rate', 'direction'}
        D_lambda_dict = {'packet_length': 1e-3, 'inter_arrival_time': 1e-3, 'up_down_rate': 1e-3, 'direction': 1e-3}
        D_max_flow_length = 1000
        if config is not None:
            allowed_names = config.get('pss', {}).get('allowed_feature_names', D_allowed_feature_names)
            if isinstance(allowed_names, str):
                allowed_names = [ft.strip() for ft in allowed_names.split(',')]
            self.allowed_feature_names = allowed_names
            lambda_dict_str = config.get('pss', {}).get('lambda_dict', str(D_lambda_dict))
            self.lambda_dict = ast.literal_eval(lambda_dict_str)
            max_flow_length_str = config.get('pss', {}).get('max_flow_length', '1000')
            self.max_flow_length = int(max_flow_length_str)
        else:
            self.allowed_feature_names = D_allowed_feature_names
            self.lambda_dict = D_lambda_dict
            self.max_flow_length = D_max_flow_length

    def build_matrices(self, feature_data: Dict[str, np.ndarray], feature_type: str, output_base_dir: str = 'feature_matrix', store_file_name: str = 'default_feature_matrix_filename', store: bool = True) -> Dict[str, dict]:
        """
        Build matrices for all flows in the feature data.
        :param feature_data: Dict {flow_id: np.array(seq)} from FeatureExtractor.
        :param feature_type: String for feature type (e.g., 'packet_length').
        :param output_base_dir: Base directory for output (default 'feature_matrix').
        :param store_file_name: File name for storing the matrices (default 'default_feature_matrix_filename').
        :param store: Whether to store the results to disk (default True).
        :return: Dict {flow_id: {'U': np.array, 'M': np.array, 'J': np.array}}.
        """
        results = {}

        # Check if feature_type is allowed
        if self.allowed_feature_names and feature_type not in self.allowed_feature_names:
            print(f'Feature type "{feature_type}" is not in allowed_feature_names. Skipping matrix building.')
            return {}
        
        for flow_id, seq in feature_data.items():
            if len(seq) < 2:
                continue  # Skip too short sequences
            U, M, J = compute_matrices(seq, self.max_flow_length, self.lambda_dict.get(feature_type, 1e-3))
            results[flow_id] = {'U': U, 'M': M, 'J': J}
        
        print(f'{len(results)} flow matrices for {feature_type} were computed and saved to {os.path.join(output_base_dir, feature_type)}')
        if store:
            self._save_matrices(results, feature_type, output_base_dir, store_file_name)
        
        return results

    def _save_matrices(self, results: Dict[str, dict], feature_type: str, output_base_dir: str, store_file_name: str) -> str:
        """
        Save matrices to .npz with writing flag for integrity.
        """
        output_dir = os.path.join(output_base_dir, feature_type)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'{store_file_name}_matrices.npz')
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
# matrices = builder.build_matrices(feature_data, 'packet_length')

if __name__ == '__main__':
    # Config for testing
    config = {'pss': {'max_flow_length': 50, 
                      'allowed_feature_names': ['packet_length', 'inter_arrival_time', 'up_down_rate', 'direction'],
                      'lambda_dict': {'packet_length': 0.5, 'inter_arrival_time': 0.3, 'up_down_rate': 0.1, 'direction': 0.1}
                      }}  # Small value to test truncation

    parser = argparse.ArgumentParser(description='Build matrices from feature data.')
    parser.add_argument('-f', '--feature_type', type=str, required=True, help='Feature type (e.g., packet_length).')
    parser.add_argument('-i', '--input_npz', type=str, required=True, help='Path to input .npz from FeatureExtractor.')
    parser.add_argument('-o', '--output', type=str, default='workspace/test/feature_matrix', help='Output base directory.')
    parser.add_argument('--max_flow_length', type=int, default=config['pss']['max_flow_length'], help='Max sequence length for matrices.')
    parser.add_argument('--run', action='store_true', help='Run building now if input provided.')

    args = parser.parse_args()
    if args.input_npz and args.run:
        feature_data = np.load(args.input_npz, allow_pickle=True)
        feature_data = {k: v for k, v in feature_data.items()}  # Convert to dict
        builder = SingleFeatureMatrixBuilder(config)
        results = builder.build_matrices(feature_data, args.feature_type, args.output, args.input_npz[:-4], store=True)
        print(f'Feature "{args.feature_type}": {len(results)} flow matrices built, saved under {os.path.join(args.output, args.feature_type)}')
        sys.exit(0)