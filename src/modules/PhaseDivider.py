import argparse
import sys
import os
import numpy as np
import json
from typing import Dict, List
from numba import njit

@njit(fastmath=True)
def compute_marks(J: np.ndarray, p: int) -> np.ndarray:
    """
    Compute phase marks using dynamic programming to maximize sum of J[s,t] for p phases.
    Returns (p-1) 1-based end indices of the first (p-1) phases.
    """
    n = J.shape[0]
    marks = []
    if n < p:
        return np.array(marks, dtype=np.int32)  # Skip if too short for p phases
    dp = np.full((p + 1, n + 1), -np.inf, dtype=np.float64)
    prev = np.zeros((p + 1, n + 1), dtype=np.int32) - 1
    dp[0][0] = 0.0

    for t in range(1, n + 1):
        dp[1][t] = J[0, t - 1]
        prev[1][t] = 0

    for k in range(2, p + 1):
        for t in range(k, n + 1):
            for j in range(k - 1, t):
                score = dp[k - 1][j] + J[j, t - 1]
                if score > dp[k][t]:
                    dp[k][t] = score
                    prev[k][t] = j

    # Backtrack to get ends (1-based)
    current = n
    k = p
    while k > 1:
        start = prev[k][current]
        marks.append(start)  # Append end of previous phase (1-based: start == end+1 of prev, but since start is 0-based index, +1 not needed? Wait: start is 0-based start of current phase, so end of prev is start - 1 +1 (1-based)
        # Correction: start is 0-based, end_prev = start (1-based, since packet 1 is index 0, end at 10 means index 9, but 10 is start of next as 10 (index 9+1)
        # To get 1-based end: start
        current = start
        k -= 1
    marks.reverse()
    return np.array(marks, dtype=np.int32)

class PhaseDivider:
    """
    Class for dividing flows into phases based on fused J matrix using dynamic programming.
    Outputs phase marks as dict of lists (1-based end indices).
    Supports saving to .json with integrity flag.
    """
    def __init__(self, config: dict = None):
        """
        :param config: Dict with 'pss' section containing optional 'num_phases' (int, default 4).
        """
        D_num_phases = 4
        if config is not None:
            self.num_phases = config.get('pss', {}).get('num_phases', D_num_phases)
        else:
            self.num_phases = D_num_phases

    def divide_phases(self, fused_data: Dict[str, np.ndarray], output_base_dir: str = 'datasets/feature_set_n/phase_marks/p_phases', store_file_name: str = 'default_phase_marks', store: bool = True) -> Dict[str, List[int]]:
        """
        Divide all flows into phases based on fused J.
        :param fused_data: Dict {flow_id: fused_J np.array} from FeatureFusion.
        :param output_base_dir: Base directory for output.
        :param store_file_name: File name for storing the marks (default 'default_phase_marks').
        :param store: Whether to store the results to disk (default True).
        :return: Dict {flow_id: [int, ...]} with 1-based phase end marks.
        """
        results = {}

        for flow_id, fused_j in fused_data.items():
            if fused_j.shape[0] < self.num_phases:
                continue  # Skip too short for phases
            marks = compute_marks(fused_j, self.num_phases)
            results[flow_id] = marks.tolist()
        
        print(f'{len(results)} phase marks were computed and saved to {output_base_dir}')
        if store:
            self._save_marks(results, output_base_dir, store_file_name)
        
        return results

    def _save_marks(self, results: Dict[str, List[int]], output_base_dir: str, store_file_name: str) -> str:
        """
        Save phase marks to .json with writing flag for integrity.
        """
        os.makedirs(output_base_dir, exist_ok=True)
        output_path = os.path.join(output_base_dir, f'{store_file_name}_phase_marks.json')
        writing_flag = output_path + '.writing'
        success = False
        open(writing_flag, 'w').close()  # Create writing flag
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f)
            success = True
        finally:
            if success:
                os.remove(writing_flag)  # Remove flag after save
        return output_path

# Usage example:
# from FeatureFusion import FeatureFusion
# config = {'pss': {'num_phases': 4}}
# fusion = FeatureFusion(config)
# fused = fusion.fuse_features(matrices_data, feature_types)
# divider = PhaseDivider(config)
# marks = divider.divide_phases(fused)

if __name__ == '__main__':
    # Config for testing
    config = {'pss': {'num_phases': 4}}

    parser = argparse.ArgumentParser(description='Divide phases from fused matrices.')
    parser.add_argument('-i', '--input_npz', type=str, required=True, help='Path to input fused .npz from FeatureFusion.')
    parser.add_argument('-o', '--output', type=str, default='workspace/test/datasets/feature_set_1', help='Output base directory.')
    parser.add_argument('--num_phases', type=int, default=config['pss']['num_phases'], help='Number of phases to divide into.')
    parser.add_argument('--run', action='store_true', help='Run division now if input provided.')

    args = parser.parse_args()
    if args.input_npz and args.run:
        fused_data = np.load(args.input_npz, allow_pickle=True)
        fused_data = {k: v for k, v in fused_data.items()}  # Convert to dict
        config['pss']['num_phases'] = args.num_phases  # Update config
        divider = PhaseDivider(config)
        output_base_dir = os.path.join(args.output, f'{args.num_phases}_phase', 'phase_marks')
        results = divider.divide_phases(fused_data, output_base_dir, os.path.basename(args.input_npz)[:-4], store=True)
        print(f'Divided {len(results)} phase marks, saved under {output_base_dir}')
        sys.exit(0)