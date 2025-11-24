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
from datetime import datetime, timedelta
import re

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
        self.local_offset = datetime.now() - datetime.utcnow()
        self.local_offset -= timedelta(microseconds=self.local_offset.microseconds)
        self._convert_rules_times_to_local()
        self.time_format = '%d/%m/%Y %I:%M:%S %p'  # Timestamp format from CICFlowMeter
    
    def _convert_rules_times_to_local(self):
        """
        Convert all time values in rules to local timezone by adding offset.
        Assumes original times in rules are in UTC.
        """
        for rule in self.rules.get('rules', []):
            for cond in rule.get('match', []):
                if cond['field'] in ['time_start', 'time_end']:
                    cond['value'] = self._convert_time_str(cond['value'])
                elif cond['field'] == 'time_window':
                    if cond['op'] == 'range':
                        cond['value'] = [self._convert_time_str(v) for v in cond['value']]
                    elif cond['op'] == 'ranges':
                        cond['value'] = [[self._convert_time_str(subv) for subv in sublist] for sublist in cond['value']]
        # self._print_bot_rules()

    def _print_bot_rules(self):
        """
        Debug function to print BOT-related rules (Botnet Ares) after time conversion.
        """
        bot_rules = [rule for rule in self.rules.get('rules', []) if rule.get('label', '').startswith('Botnet Ares')]
        print("Adjusted BOT-related rules after time conversion:")
        print(yaml.dump({'rules': bot_rules}, default_flow_style=False))

    def _convert_time_str(self, dt_str: str) -> str:
        """
        Parse ISO datetime string (UTC), add local offset, return as ISO string.
        Handles strings with or without microseconds.
        """
        naive_utc = datetime.fromisoformat(dt_str)
        naive_local = naive_utc + self.local_offset
        return naive_local.isoformat()

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
        :param num_phases: Number of phases
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
            # Drop masked features (but keep Timestamp for time matching)
            mask_cols = [col for col in self.mask_features if col in df.columns and col != 'Timestamp']
            df = df.drop(columns=mask_cols, errors='ignore')

            # Extract src_ip, dst_ip, src_port, dst_port, proto from Flow Key
            key_parts = df['Flow Key'].str.split('-', expand=True)
            df['src_ip'] = key_parts[0]
            df['dst_ip'] = key_parts[1]
            df['src_port'] = key_parts[2].astype(int)
            df['dst_port'] = key_parts[3].astype(int)
            df['proto'] = key_parts[4].apply(lambda x: int(x) if x.isdigit() else x)

            # Parse Timestamp
            df['flow_start'] = pd.to_datetime(df['Timestamp'], format=self.time_format, errors='coerce')
            df['flow_duration_s'] = df['Flow Duration_p1'] / 1000000.0
            df['flow_end'] = df['flow_start'] + pd.to_timedelta(df['flow_duration_s'], unit='s')

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
        Handles match conditions, extra_logic, and time_window with intersection logic.
        """
        matched = self._compute_cond_list(df, rule.get('match', []))

        extra = rule.get('extra_logic', {})
        if extra:
            matched &= self._compute_logic(df, extra)

        return matched

    def _compute_cond_list(self, df: pd.DataFrame, cond_list: List[Dict]) -> pd.Series:
        """
        Compute boolean mask for a list of conditions (AND logic).
        """
        matched = pd.Series(True, index=df.index)
        rule_start = None
        rule_end = None
        ranges = []

        for cond in cond_list:
            field = cond['field']
            op = cond['op']
            value = cond['value']

            if field == 'time_start':
                rule_start = datetime.fromisoformat(value)
                continue  # Special, handled after
            if field == 'time_end':
                rule_end = datetime.fromisoformat(value)
                continue  # Special, handled after
            if field == 'time_window':
                if op == 'range':
                    rule_start = datetime.fromisoformat(value[0])
                    rule_end = datetime.fromisoformat(value[1])
                elif op == 'ranges':
                    ranges = [(datetime.fromisoformat(r[0]), datetime.fromisoformat(r[1])) for r in value]
                continue  # Special, handled after

            # Special handling for 'any' wildcard (placeholder in rules)
            if field == 'any':
                if op == '==' and value is True:
                    continue  # Always true, no change to matched
                else:
                    raise ValueError(f"Unsupported op/value for 'any': {op}, {value}")

            # Special handling for 'any_ip'
            if field == 'any_ip':
                matched &= (df['src_ip'].isin(value) | df['dst_ip'].isin(value))
                continue

            # Get column or special value
            if field in df.columns:
                col = df[field]
            else:
                col = self._get_special_field(df, field)

            # Apply operator
            matched &= self._apply_op(col, op, value)

        # Handle time intersection
        if rule_start is not None and rule_end is not None:
            matched &= ~((df['flow_end'] < rule_start) | (df['flow_start'] > rule_end))
        elif ranges:
            any_intersect = pd.Series(False, index=df.index)
            for r_start, r_end in ranges:
                any_intersect |= ~((df['flow_end'] < r_start) | (df['flow_start'] > r_end))
            matched &= any_intersect

        return matched

    def _get_special_field(self, df: pd.DataFrame, field: str) -> pd.Series:
        """
        Get value for special fields like zero_fwd, flow_duration_s, etc.
        """
        if field == 'zero_fwd':
            return df['Total Length of Fwd Packet_p1'] == 0
        if field == 'zero_bwd':
            return df['Total Length of Bwd Packet_p1'] == 0
        if field == 'flow_duration_s':
            return df['Flow Duration_p1'] / 1000000.0
        if field == 'flow_duration':  # Handle rule field as seconds
            return df['Flow Duration_p1'] / 1000000.0
        if field in ['fwd_rst_flags', 'bwd_rst_flags']:  # Assume RST Flag Count for both (no separate in standard)
            return df['RST Flag Count_p1']
        if field == 'total_fwd_pkts':
            return df['Total Fwd Packet_p1']
        if field == 'total_fwd_len':
            return df['Total Length of Fwd Packet_p1']
        if field == 'total_bwd_len':
            return df['Total Length of Bwd Packet_p1']
        # Add more as needed
        raise ValueError(f"Unknown special field: {field}")

    def _apply_op(self, col: pd.Series, op: str, value: any) -> pd.Series:
        """
        Apply operator to column and value.
        """
        if op == '==':
            return col == value
        if op == '!=':
            return col != value
        if op == '>':
            return col > value
        if op == '>=':
            return col >= value
        if op == '<':
            return col < value
        if op == '<=':
            return col <= value
        if op == 'in':
            return col.isin(value)
        if op == 'not_in':
            return ~col.isin(value)
        if op == 'contains':
            return col.str.contains(value)
        if op == 'regex':
            return col.str.match(value)
        if op == 'isnull':
            return col.isnull()
        if op == 'notnull':
            return ~col.isnull()
        raise ValueError(f"Unknown operator: {op}")

    def _compute_logic(self, df: pd.DataFrame, logic: Dict) -> pd.Series:
        """
        Recursively compute extra_logic (any_of/all_of).
        """
        if 'any_of' in logic:
            matched = pd.Series(False, index=df.index)
            for sub in logic['any_of']:
                matched |= self._compute_logic(df, sub)
            return matched
        if 'all_of' in logic:
            matched = pd.Series(True, index=df.index)
            for sub in logic['all_of']:
                matched &= self._compute_cond_list(df, [sub])  
            return matched
        raise ValueError("Invalid logic structure")

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