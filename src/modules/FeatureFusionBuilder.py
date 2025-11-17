import argparse
import sys
import os
import numpy as np
from typing import Dict, List
from collections import defaultdict
from numba import njit, prange

@njit(fastmath=True)  # Removed parallel=True to avoid List ndim error in Numba parfor analysis
def fuse_matrices(js_list: List[np.ndarray], weights: np.ndarray) -> np.ndarray:
    """
    Fuse multiple J matrices with normalization and weighted sum using Numba.
    Normalizes each J to [0,1] before weighting.
    Assumes all Js have the same shape.
    """
    num_mats = len(js_list)
    shape = js_list[0].shape
    fused_j = np.zeros(shape, dtype=np.float64)
    
    for i in range(num_mats):  # Use serial loop instead of prange for compatibility with List[np.ndarray]
        j_mat = js_list[i]
        j_min = np.min(j_mat)
        j_max = np.max(j_mat)
        if j_max - j_min > 1e-6:  # Avoid division by zero
            norm_j = (j_mat - j_min) / (j_max - j_min)
        else:
            norm_j = np.zeros_like(j_mat)
        fused_j += norm_j * weights[i]
    
    return fused_j

class FeatureFusion:
    """
    Class for fusing multiple single-feature J matrices into a single fused J matrix.
    Performs normalization, weighted sum, and optional regularization.
    Supports saving to .npz with integrity flag.
    """
    def __init__(self, config: dict = None):
        """
        :param config: Dict with 'pss' section containing optional 'allowed_feature_names' (list of str), 
                       'feature_weights' (dict of str to float), 'regularization_lambda' (float, default 1e-3).
        """
        D_allowed_feature_names = {'packet_length', 'inter_arrival_time', 'up_down_rate', 'direction'}
        D_feature_weights = {'packet_length': 1.0, 'inter_arrival_time': 1.0, 'up_down_rate': 1.0, 'direction': 1.0}
        D_regularization_lambda = 1e-3
        if config is not None:
            self.allowed_feature_names = config.get('pss', {}).get('allowed_feature_names', D_allowed_feature_names)
            self.feature_weights = config.get('pss', {}).get('feature_weights', D_feature_weights)
            self.regularization_lambda = config.get('pss', {}).get('regularization_lambda', D_regularization_lambda)
        else:
            self.allowed_feature_names = D_allowed_feature_names
            self.feature_weights = D_feature_weights
            self.regularization_lambda = D_regularization_lambda

    def fuse_features(self, matrices_data: Dict[str, Dict[str, dict]], feature_types: List[str], output_base_dir: str = 'datasets/feature_set_n/merged_matrix', store_file_name: str = 'default_merged_matrix_filename', store: bool = True) -> Dict[str, np.ndarray]:
        """
        Fuse J matrices for all flows across multiple feature types.
        :param matrices_data: Dict {feature_type: {flow_id: {'U': np.array, 'M': np.array, 'J': np.array}}}.
        :param feature_types: List of str for feature types to fuse (e.g., ['packet_length', 'inter_arrival_time']).
        :param output_base_dir: Base directory for output.
        :param store_file_name: File name for storing the fused matrix (default 'default_merged_matrix_filename').
        :param store: Whether to store the results to disk (default True).
        :return: Dict {flow_id: fused_J np.array}.
        """
        results = {}

        # Check if feature_types are allowed
        if not set(feature_types).issubset(self.allowed_feature_names):
            print(f'Unsupported features: {set(feature_types) - set(self.allowed_feature_names)}. Skipping fusion.')
            return {}
        
        # Get weights array
        weights = np.array([self.feature_weights.get(ft, 1.0) for ft in feature_types])
        
        # Assume all flows are consistent across feature types
        flow_ids = list(matrices_data[feature_types[0]].keys())
        for flow_id in flow_ids:
            js_list = []
            for ft in feature_types:
                j_mat = matrices_data[ft][flow_id]['J']
                js_list.append(j_mat)
            
            fused_j = fuse_matrices(js_list, weights)
            # Apply regularization: add lambda to diagonal for stability
            np.fill_diagonal(fused_j, fused_j.diagonal() + self.regularization_lambda)
            results[flow_id] = fused_j
        
        print(f'{len(results)} fused matrices were computed and saved to {output_base_dir}')
        if store:
            self._save_fused_matrix(results, output_base_dir, store_file_name)
        
        return results

    def _save_fused_matrix(self, results: Dict[str, np.ndarray], output_base_dir: str, store_file_name: str) -> str:
        """
        Save fused J matrices to .npz with writing flag for integrity.
        """
        os.makedirs(output_base_dir, exist_ok=True)
        output_path = os.path.join(output_base_dir, f'{store_file_name}_fused.npz')
        writing_flag = output_path + '.writing'
        success = False
        open(writing_flag, 'w').close()  # Create writing flag
        try:
            np.savez(output_path, **results)
            success = True
        finally:
            if success:
                os.remove(writing_flag)  # Remove flag after save
        return output_path

# Usage example:
# from SingleFeatureMatrixBuilder import SingleFeatureMatrixBuilder
# config = {'pss': {'feature_weights': {'packet_length': 0.5, 'inter_arrival_time': 0.3}}}
# builder = SingleFeatureMatrixBuilder(config)
# matrices_pl = builder.build_matrices(feature_data_pl, 'packet_length')
# matrices_iat = builder.build_matrices(feature_data_iat, 'inter_arrival_time')
# matrices_data = {'packet_length': matrices_pl, 'inter_arrival_time': matrices_iat}
# fusion = FeatureFusion(config)
# fused = fusion.fuse_features(matrices_data, ['packet_length', 'inter_arrival_time'])

if __name__ == '__main__':
    # Config for testing
    config = {'pss': {'allowed_feature_names': ['packet_length', 'inter_arrival_time', 'up_down_rate', 'direction'],
                      'feature_weights': {'packet_length': 0.5, 'inter_arrival_time': 0.3, 'up_down_rate': 0.1, 'direction': 0.1},
                      'regularization_lambda': 1e-3}}

    parser = argparse.ArgumentParser(description='Fuse matrices from multiple features.')
    parser.add_argument('-f', '--feature_types', type=str, required=True, help='Comma-separated feature types (e.g., packet_length,inter_arrival_time).')
    parser.add_argument('-i', '--input_npz_paths', type=str, required=True, help='Comma-separated paths to input .npz files for each feature type.')
    parser.add_argument('-o', '--output', type=str, default='workspace/test/datasets/feature_set_1', help='Output base directory.')
    parser.add_argument('--run', action='store_true', help='Run fusion now if inputs provided.')

    args = parser.parse_args()
    if args.input_npz_paths and args.run:
        feature_types = [ft.strip() for ft in args.feature_types.split(',') if ft.strip()]
        input_paths = [path.strip() for path in args.input_npz_paths.split(',') if path.strip()]
        if len(feature_types) != len(input_paths):
            print('Mismatch between feature_types and input_npz_paths counts.')
            sys.exit(1)
        
        matrices_data = {}
        for ft, path in zip(feature_types, input_paths):
            data = np.load(path, allow_pickle=True)
            matrices_data[ft] = {k: v.item() for k, v in data.items()}  # Convert to dict with inner dicts
        
        fusion = FeatureFusion(config)
        results = fusion.fuse_features(matrices_data, feature_types, args.output, os.path.basename(input_paths[0])[:-4], store=True)
        print(f'Fused {len(results)} matrices, saved under {args.output}')
        sys.exit(0)