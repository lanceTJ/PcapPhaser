# filename: FeatureConcatenator.py
import argparse
import sys
import os
import pandas as pd
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple
from collections import defaultdict
from datetime import datetime  # For timestamp parsing and delta calculation

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
        self.timeout_sec = config.get('concat', {}).get('timeout_sec', 600.0) if config else 600.0  # Default timeout in seconds

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
            # Local grouped for this pcap: {key: {ph: [(ts, row_dict), ...]}}
            grouped = defaultdict(lambda: defaultdict(list))

            for ph in range(1, num_phases + 1):
                df = phase_dfs.get(ph)
                if df is None:
                    continue  # Skip missing phases for this pcap
                for _, row in df.iterrows():
                    key = (row['Src IP'], row['Dst IP'], row['Src Port'], row['Dst Port'], row['Protocol'])
                    ts = row['Timestamp']
                    row_dict = {col: row[col] for col in feature_columns}
                    grouped[key][ph].append((ts, row_dict))

            # Process groups to create concatenated rows
            concatenated_rows = []
            for key in grouped:
                phase_items = grouped[key]
                # Debug log: key and per-ph items without row_dict
                ph_ts_dict = {ph: [item[0] for item in phase_items[ph]] for ph in phase_items}
                logging.warning(f"Flow key={key}, per-phase timestamps: {ph_ts_dict}")

                # Aggregate subflows by phase
                sub_features = self._aggregate_subflows_by_phase(phase_items, num_phases)

                # Concatenate into flat dict with suffixed keys
                flat_row = {'Flow Key': '-'.join(map(str, key))}
                flat_row['Timestamp'] = min(item[0] for ph_items in phase_items.values() for item in ph_items) if phase_items else ''  # Earliest ts across all
                for ph in range(1, num_phases + 1):
                    if sub_features[ph - 1]:
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

    def _aggregate_subflows_by_phase(self, phase_items: Dict[int, List[Tuple[str, Dict]]], num_phases: int) -> List[Dict]:
        """
        Aggregate subflows per phase for a single key, starting from phase 1, checking timeouts between phases.
        :param phase_items: {ph: [(ts, row_dict), ...]} with sorted ts per ph
        :param num_phases: Number of phases
        :return: List of aggregated feat dict per phase (length num_phases)
        """
        sub_features = [{} for _ in range(num_phases)]  # Init empty dict per phase
        if 1 not in phase_items or not phase_items[1]:
            logging.debug("No phase 1 items, skipping aggregation")
            return sub_features  # Empty if no phase 1

        # Sort per phase (ensure, though already in loop)
        for ph in phase_items:
            phase_items[ph].sort(key=lambda x: x[0])

        # Start with phase 1: aggregate all items in phase 1 to single feat
        sub_features[0] = self._aggregate_phase_feats(phase_items[1])
        last_ts = datetime.fromisoformat(phase_items[1][-1][0])  # Last ts of phase 1 (assume isoformat)

        # For ph=2 to num_phases
        for ph in range(2, num_phases + 1):
            if ph not in phase_items or not phase_items[ph]:
                # No items: copy previous phase feat
                sub_features[ph - 1] = sub_features[ph - 2].copy()
                logging.debug(f"Phase {ph} no items, copied phase {ph-1} feat")
                continue

            # Check each item in current ph
            selected_items = []
            for ts_str, row_dict in phase_items[ph]:
                curr_ts = datetime.fromisoformat(ts_str)
                delta = (curr_ts - last_ts).total_seconds()
                if delta < self.timeout_sec:
                    selected_items.append((ts_str, row_dict))
                    last_ts = curr_ts  # Update last_ts to this item
                else:
                    logging.debug(f"Item in phase {ph} ts={ts_str} timeout delta={delta} > {self.timeout_sec}, stopping check")
                    break  # Since sorted, later items larger delta

            if selected_items:
                # Aggregate selected
                sub_features[ph - 1] = self._aggregate_phase_feats(selected_items)
                logging.debug(f"Phase {ph} selected {len(selected_items)} items, aggregated")
            else:
                # No valid: copy previous
                sub_features[ph - 1] = sub_features[ph - 2].copy()
                logging.debug(f"Phase {ph} no valid items after timeout check, copied phase {ph-1} feat")

        return sub_features

    def _aggregate_phase_feats(self, items: List[Tuple[str, Dict]]) -> Dict:
        """
        Aggregate multiple row_dict in a phase to single feat dict (sum/avg/max rules).
        :param items: [(ts, row_dict), ...]
        :return: Aggregated dict
        """
        if not items:
            return {}

        aggregated = {}
        for feat in items[0][1]:  # Assume all have same keys
            values = [item[1][feat] for item in items]
            if 'Total' in feat or 'Count' in feat or 'Length' in feat:
                aggregated[feat] = sum(values)  # Sum for counts/lengths
            elif 'Max' in feat or 'Duration' in feat:
                aggregated[feat] = max(values)  # Max for maxes/durations
            elif 'Min' in feat:
                aggregated[feat] = min(values)  # Min for mins
            else:
                aggregated[feat] = sum(values) / len(values)  # Avg for means/stds
        return aggregated

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
        config = {'concat': {'max_workers': 8, 'mask_features': ['Flow ID', 'Label']}}  # Example config
        concatenator = FeatureConcatenator(config)
        concatenator.concatenate_features(args.dataset_dir, args.num_phases, store=True)