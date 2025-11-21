# src/modules/FeatureConcatenator.py
import argparse
import sys
import os
import pandas as pd
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List
from collections import defaultdict

class FeatureConcatenator:
    """
    Class for concatenating phase-level features from CICFlowMeter CSVs into a single per-original-flow feature vector.
    Handles short flow replication, feature masking, and integrity flags.
    """
    def __init__(self, config: dict = None):
        """
        Initialize with config, loading mask_features for ignoring specific columns during concatenation.
        """
        self.mask_features = config.get('concat', {}).get('mask_features', []) if config else []
        self.max_workers = config.get('concat', {}).get('max_workers', max(4, os.cpu_count() or 4)) if config else max(4, os.cpu_count() or 4)

    def concatenate_features(self,
                             phase_base_dir: str,
                             num_phases: int,
                             output_csv_path: str = None,
                             store: bool = True) -> str:
        """
        Concatenate features from all phase CSVs under a specific phase experiment directory.
        :param phase_base_dir: Path like 'datasets/feature_set_1/4_phase'
        :param num_phases: Number of phases
        :param output_csv_path: Path for output CSV (default: {phase_base_dir}/concatenated_features.csv)
        :param store: Whether to save the concatenated CSV (False for dry-run)
        :return: Path to concatenated CSV if stored, else None
        """
        cfm_features_root = os.path.join(phase_base_dir, 'cfm_features')
        if output_csv_path is None:
            output_csv_path = os.path.join(phase_base_dir, 'concatenated_features.csv')

        tasks = []
        for ph in range(1, num_phases + 1):
            input_dir = os.path.join(cfm_features_root, f'phase_{ph}')
            if not os.path.exists(input_dir):
                logging.warning(f"Phase {ph} features directory not exists: {input_dir}")
                continue
            tasks.append((ph, input_dir))

        if not tasks:
            print(f"No phase features found under {cfm_features_root}")
            return None

        print(f"Starting feature concatenation from {len(tasks)} phase directories using {self.max_workers} workers")

        # Load all phase data in parallel
        phase_data = {}  # {ph: list of pd.DataFrame}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._load_phase_csvs, ph, in_dir): ph
                for ph, in_dir in tasks
            }
            for future in as_completed(futures):
                ph = futures[future]
                phase_data[ph] = future.result()

        # Group rows by 5-tuple across all phases
        grouped = defaultdict(list)  # {(src_ip, dst_ip, src_port, dst_port, proto): [(timestamp, row_dict, ph), ...]}
        id_columns = ['Src IP', 'Dst IP', 'Src Port', 'Dst Port', 'Protocol', 'Timestamp']
        feature_columns = None  # To be set from first DF

        for ph, dfs in phase_data.items():
            for df in dfs:
                if feature_columns is None:
                    feature_columns = [col for col in df.columns if col not in id_columns + self.mask_features]
                for _, row in df.iterrows():
                    key = (row['Src IP'], row['Dst IP'], row['Src Port'], row['Dst Port'], row['Protocol'])
                    ts = row['Timestamp']
                    row_dict = {col: row[col] for col in feature_columns}
                    grouped[key].append((ts, row_dict, ph))

        # Process groups to create concatenated rows
        concatenated_rows = []
        for key, items in grouped.items():
            # Sort by timestamp
            items.sort(key=lambda x: x[0])
            sub_features = [item[1] for item in items]
            num_subs = len(sub_features)
            if num_subs == 1:
                # Short flow: replicate to all phases
                sub_features = sub_features * num_phases
            elif num_subs == num_phases:
                # Long flow: use as is
                pass
            else:
                logging.warning(f"Unexpected sub-flow count {num_subs} for key {key}, skipping")
                continue

            # Concatenate into flat dict with suffixed keys
            flat_row = {'Flow Key': '-'.join(map(str, key))}
            for ph in range(1, num_phases + 1):
                for feat, val in sub_features[ph - 1].items():
                    flat_row[f"{feat}_p{ph}"] = val
            concatenated_rows.append(flat_row)

        if not concatenated_rows:
            print("No features to concatenate")
            return None

        result_df = pd.DataFrame(concatenated_rows)

        if store:
            writing_flag = output_csv_path + '.writing'
            if os.path.exists(writing_flag):
                print("Already processing or failed, skipping save")
                return None
            open(writing_flag, 'w').close()
            try:
                result_df.to_csv(output_csv_path, index=False)
                print(f"Concatenated features saved to {output_csv_path}")
                return output_csv_path
            except Exception as e:
                print(f"Exception during save: {e}")
                return None
            finally:
                if os.path.exists(writing_flag):
                    os.remove(writing_flag)
        else:
            print("Dry-run completed, no file saved")
            return None

    def _load_phase_csvs(self, phase_num: int, input_dir: str) -> List[pd.DataFrame]:
        """
        Load all CSV files from a phase directory.
        """
        csv_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.csv')]
        dfs = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                dfs.append(df)
            except Exception as e:
                logging.warning(f"Failed to load {csv_file}: {e}")
        print(f"[Phase {phase_num}] Loaded {len(dfs)} CSV files")
        return dfs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Concatenate phase-level features from CFM CSVs')
    parser.add_argument('-d', '--dataset_dir', type=str, required=True,
                        help='Full path to phase dataset dir, e.g., datasets/feature_set_1/4_phase')
    parser.add_argument('--num_phases', type=int, required=True, help='Number of phases')
    parser.add_argument('--run', action='store_true', help='Execute concatenation now')

    args = parser.parse_args()
    if args.run:
        config = {'concat': {'max_workers': 8, 'mask_features': ['Flow ID', 'Timestamp']}}  # Example config
        concatenator = FeatureConcatenator(config)
        concatenator.concatenate_features(args.dataset_dir, args.num_phases, store=True)