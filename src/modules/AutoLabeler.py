# src/modules/AutoLabeler.py
import argparse
import sys
import os
import pandas as pd
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple
from collections import defaultdict
import yaml
from datetime import datetime

class AutoLabeler:
    """
    Class for automatically labeling concatenated features from CSVs using CIC-IDS2018 improved rules.
    Applies rules from YAML file, adds 'Label' column, and handles feature masking.
    Supports per-original-pcap labeling and output.
    """
    def __init__(self, config: dict = None):
        """
        Initialize with config, loading mask_features for ignoring specific columns during labeling.
        """
        self.mask_features = config.get('label', {}).get('mask_features', []) if config else []
        self.rules_file = config.get('label', {}).get('rules_file', 'cic_improved_2018_rules.yaml') if config else 'cic_improved_2018_rules.yaml'
        self.max_workers = config.get('label', {}).get('max_workers', max(4, os.cpu_count() or 4)) if config else max(4, os.cpu_count() or 4)
        self.rules = self._load_rules()

    def _load_rules(self) -> Dict:
        """
        Load labeling rules from YAML file.
        """
        if not os.path.exists(self.rules_file):
            raise FileNotFoundError(f"Rules file not found: {self.rules_file}")
        with open(self.rules_file, 'r') as f:
            return yaml.safe_load(f)

    def label_features(self,
                       phase_base_dir: str,
                       num_phases: int,
                       store: bool = True) -> Dict[str, str]:
        """
        Label concatenated features from concat CSVs under a specific phase experiment directory, grouped by original pcap basename.
        :param phase_base_dir: Path like 'datasets/feature_set_1/4_phase'
        :param num_phases: Number of phases (unused here but for consistency)
        :param store: Whether to save the labeled CSVs (False for dry-run)
        :return: Dict {pcap_basename: output_csv_path}
        """
        concat_input_root = os.path.join(phase_base_dir, 'concat_csv')
        labeled_output_root = os.path.join(phase_base_dir, 'labeled_csv')
        if store:
            os.makedirs(labeled_output_root, exist_ok=True)

        csv_files = [os.path.join(concat_input_root, f) for f in os.listdir(concat_input_root) if f.endswith('_concat.csv')]
        if not csv_files:
            print(f"No concatenated CSVs found under {concat_input_root}")
            return {}

        print(f"Starting feature labeling for {len(csv_files)} CSVs using {self.max_workers} workers")

        # Load all concat CSVs in parallel: List[Tuple[pd.DataFrame, basename]]
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._load_concat_csv, csv_file) for csv_file in csv_files]
            data = [future.result() for future in as_completed(futures)]

        if not data:
            print("No data found after loading")
            return {}

        # Process each basename independently
        results = {}
        for df, basename in data:
            # Drop masked features
            df = df.drop(columns=[col for col in self.mask_features if col in df.columns], errors='ignore')

            # Apply labels
            df['Label'] = self.rules['meta']['default_label']
            rules_list = self.rules.get('rules', [])
            for rule in sorted(rules_list, key=lambda r: r.get('priority', float('inf'))):
                matched = self._apply_rule(df, rule)
                if matched.any():
                    df.loc[matched, 'Label'] = rule['label']

            output_csv_path = os.path.join(labeled_output_root, f"{basename}_labeled.csv")

            if store:
                writing_flag = output_csv_path + '.writing'
                if os.path.exists(writing_flag):
                    print(f"Already processing or failed for {basename}, skipping save")
                    continue
                open(writing_flag, 'w').close()
                try:
                    df.to_csv(output_csv_path, index=False)
                    print(f"Labeled features for {basename} saved to {output_csv_path}")
                    results[basename] = output_csv_path
                except Exception as e:
                    print(f"Exception during save for {basename}: {e}")
                finally:
                    if os.path.exists(writing_flag):
                        os.remove(writing_flag)
            else:
                print(f"Dry-run completed for {basename}, no file saved")

        return results

    def _load_concat_csv(self, csv_file: str) -> Tuple[pd.DataFrame, str]:
        """
        Load a concatenated CSV file and extract original pcap basename from filename.
        e.g., from 'capEC2AMAZ-O4EL3NG-172.31.69.29.pcap_Flow_concat.csv' extract 'capEC2AMAZ-O4EL3NG-172.31.69.29.pcap_Flow'
        """
        try:
            df = pd.read_csv(csv_file)
            filename = os.path.basename(csv_file)
            basename = filename.rsplit('_concat.csv', 1)[0]
            return df, basename
        except Exception as e:
            logging.warning(f"Failed to load {csv_file}: {e}")
            return None, None

    def _apply_rule(self, df: pd.DataFrame, rule: Dict) -> pd.Series:
        """
        Apply a single rule to the DataFrame, returning a boolean mask of matching rows.
        Handles match conditions, extra_logic, and time_window.
        """
        matched = pd.Series(True, index=df.index)
        for cond in rule.get('match', []):
            field = cond['field']
            op = cond['op']
            value = cond['value']

            if field == 'time_start' or field == 'time_end':
                # Assume Timestamp is in df, convert to datetime
                df_ts = pd.to_datetime(df['Timestamp'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
                rule_time = datetime.fromisoformat(value)
                if op == '==':
                    matched &= (df_ts == rule_time)
                # Add more ops as needed (e.g., range, ranges)
            elif field in df.columns:
                col = df[field]
                if op == '==':
                    matched &= (col == value)
                elif op == '!=':
                    matched &= (col != value)
                elif op == '>':
                    matched &= (col > value)
                # Add other operators: in, not_in, contains, etc.
            # Handle special fields like zero_fwd, flow_duration_s, etc.

        # Handle extra_logic if present (e.g., any_of, all_of)
        extra = rule.get('extra_logic', {})
        if extra:
            # Implement logic for any_of/all_of (recursive or eval-based)
            pass  # Placeholder: add based on YAML structure

        return matched

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Label concatenated features using CIC-IDS2018 rules')
    parser.add_argument('-d', '--dataset_dir', type=str, required=True,
                        help='Full path to phase dataset dir, e.g., datasets/feature_set_1/4_phase')
    parser.add_argument('--num_phases', type=int, required=True, help='Number of phases')
    parser.add_argument('--run', action='store_true', help='Execute labeling now')

    args = parser.parse_args()
    if args.run:
        config = {'label': {'max_workers': 8, 'mask_features': ['Flow ID'], 'rules_file': '/home/lance/PcapPhaser/workspace/test/label_rules/cic_improved_2018_rule.yaml'}}  # Example config
        labeler = AutoLabeler(config)
        labeler.label_features(args.dataset_dir, args.num_phases, store=True)