# filename: FeatureConcatenator.py
import argparse
import sys
import os
import pandas as pd
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple
from collections import defaultdict

class FeatureConcatenator:
    """
    Class for concatenating phase-level features from CICFlowMeter CSVs into a single per-original-flow feature vector.
    Handles short flow replication, feature masking, and integrity flags.
    Supports per-original-pcap concatenation and output.
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
                             store: bool = True) -> Dict[str, str]:
        """
        Concatenate features from all phase CSVs under a specific phase experiment directory, grouped by original pcap basename.
        :param phase_base_dir: Path like 'datasets/feature_set_1/4_phase'
        :param num_phases: Number of phases
        :param store: Whether to save the concatenated CSVs (False for dry-run)
        :return: Dict {pcap_basename: output_csv_path}
        """
        cfm_features_root = os.path.join(phase_base_dir, 'cfm_features')
        concat_output_root = os.path.join(phase_base_dir, 'concat_csv')
        if store:
            os.makedirs(concat_output_root, exist_ok=True)

        tasks = []
        for ph in range(1, num_phases + 1):
            input_dir = os.path.join(cfm_features_root, f'phase_{ph}')
            if not os.path.exists(input_dir):
                logging.warning(f"Phase {ph} features directory not exists: {input_dir}")
                continue
            tasks.append((ph, input_dir))

        if not tasks:
            print(f"No phase features found under {cfm_features_root}")
            return {}

        print(f"Starting feature concatenation from {len(tasks)} phase directories using {self.max_workers} workers")

        # Load all phase data in parallel: {ph: List[Tuple[pd.DataFrame, pcap_basename]]}
        phase_data = {}  # {ph: List[Tuple[pd.DataFrame, str]]}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._load_phase_csvs, ph, in_dir): ph
                for ph, in_dir in tasks
            }
            for future in as_completed(futures):
                ph = futures[future]
                phase_data[ph] = future.result()

        # Group by pcap_basename across phases: {basename: {ph: df}}
        pcap_groups = defaultdict(dict)  # {basename: {ph: pd.DataFrame}}
        id_columns = ['Src IP', 'Dst IP', 'Src Port', 'Dst Port', 'Protocol', 'Timestamp']
        feature_columns = None

        for ph, df_tuples in phase_data.items():
            for df, basename in df_tuples:
                if feature_columns is None:
                    feature_columns = [col for col in df.columns if col not in id_columns + self.mask_features]
                pcap_groups[basename][ph] = df

        if not pcap_groups:
            print("No pcap groups found after loading")
            return {}

        # Process each pcap_group independently
        results = {}
        for basename, phase_dfs in pcap_groups.items():
            # Local grouped for this pcap
            grouped = defaultdict(list)  # {(src_ip, dst_ip, src_port, dst_port, proto): [(timestamp, row_dict, ph), ...]}

            for ph in range(1, num_phases + 1):
                df = phase_dfs.get(ph)
                if df is None:
                    continue  # Skip missing phases for this pcap
                for _, row in df.iterrows():
                    key = (row['Src IP'], row['Dst IP'], row['Src Port'], row['Dst Port'], row['Protocol'])
                    ts = row['Timestamp']
                    row_dict = {col: row[col] for col in feature_columns}
                    grouped[key].append((ts, row_dict, ph))

            # Process groups to create concatenated rows (same as original)
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
                    if num_subs < num_phases:
                        # Replicate first phase to fill
                        first_feat = sub_features[0]
                        sub_features.extend([first_feat] * (num_phases - num_subs))
                        logging.debug(f"Short flow with {num_subs} phases, replicated last phase to fill {num_phases}")
                    else:
                        first_feats = sub_features[:num_phases]
                        sub_features = first_feats
                        logging.debug(f"Long flow with {num_subs} phases, truncated to first {num_phases}")
                    continue

                # Concatenate into flat dict with suffixed keys
                flat_row = {'Flow Key': '-'.join(map(str, key))}
                for ph in range(1, num_phases + 1):
                    for feat, val in sub_features[ph - 1].items():
                        flat_row[f"{feat}_p{ph}"] = val
                concatenated_rows.append(flat_row)

            if not concatenated_rows:
                logging.warning(f"No features to concatenate for pcap {basename}")
                continue

            result_df = pd.DataFrame(concatenated_rows)
            output_csv_path = os.path.join(concat_output_root, f"{basename}_concat.csv")

            if store:
                writing_flag = output_csv_path + '.writing'
                if os.path.exists(writing_flag):
                    print(f"Already processing or failed for {basename}, skipping save")
                    continue
                open(writing_flag, 'w').close()
                try:
                    result_df.to_csv(output_csv_path, index=False)
                    print(f"Concatenated features for {basename} saved to {output_csv_path}")
                    results[basename] = output_csv_path
                except Exception as e:
                    print(f"Exception during save for {basename}: {e}")
                finally:
                    if os.path.exists(writing_flag):
                        os.remove(writing_flag)
            else:
                print(f"Dry-run completed for {basename}, no file saved")

        return results

    def _load_phase_csvs(self, phase_num: int, input_dir: str) -> List[Tuple[pd.DataFrame, str]]:
        """
        Load all CSV files from a phase directory, and extract original pcap basename from filename.
        e.g., from 'phase_1_pcap_file1.csv' extract 'pcap_file1'
        """
        csv_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.csv')]
        dfs_with_basenames = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                # Extract basename: remove 'p_n_' prefix and '.csv' suffix
                filename = os.path.basename(csv_file)
                basename = filename.split(f'p_{phase_num}_', 1)[-1].rsplit('.csv', 1)[0]
                dfs_with_basenames.append((df, basename))
            except Exception as e:
                logging.warning(f"Failed to load {csv_file}: {e}")
        print(f"[Phase {phase_num}] Loaded {len(dfs_with_basenames)} CSV files")
        return dfs_with_basenames

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