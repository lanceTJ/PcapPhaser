# src/pipline_cic_ids_2018.py
import argparse
import sys
import os
import configparser
import numpy as np
import json
from typing import Dict, List

# Import all module classes
from modules.FeatureExtractor import FeatureExtractor
from modules.SingleFeatureMatrixBuilder import SingleFeatureMatrixBuilder
from modules.FeatureFusionBuilder import FeatureFusion
from modules.PhaseDivider import PhaseDivider
from modules.PhaseReconstructor import PhaseReconstructor
from modules.CFMRunner import CFMRunner
from modules.FeatureConcatenator import FeatureConcatenator
from modules.AutoLabeler import AutoLabeler
from modules.utils import load_config  # Assuming utils.py is in the same directory or path

def main():
    parser = argparse.ArgumentParser(description='Full pipeline for generating phased datasets from PCAP files using PcapPhaser modules.')
    parser.add_argument('--config', type=str, required=True, help='Path to config.ini file.')
    parser.add_argument('--input_dir', type=str, required=True, help='Path to input date directory, e.g., pcapdata/Friday-02-03-2018.')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to output date directory, e.g., workspace/Friday-02-03-2018')
    parser.add_argument('--dataset', type=str, default='dataset1', help='Dataset name for output directory under each date dir.')
    parser.add_argument('--run', action='store_true', help='Run the pipeline now.')

    args = parser.parse_args()
    if not args.run:
        print("Use --run to execute the pipeline.")
        sys.exit(0)

    # Load unified config
    config = load_config(args.config)  # Returns dict from utils.py
    # Extend config with sections for all modules
    full_config = configparser.ConfigParser()
    full_config.read(args.config)
    config = {section: dict(full_config.items(section)) for section in full_config.sections()}

    # Extract key params from config
    feature_types = config.get('pss', {}).get('allowed_feature_names', ['packet_length', 'inter_arrival_time', 'direction', 'up_down_rate'])
    if isinstance(feature_types, str):
        feature_types = [ft.strip() for ft in feature_types.split(',')]
    num_phases_raw = config.get('pss', {}).get('num_phases', '2,3,4')
    num_phases_clean = num_phases_raw.split(';')[0].strip()  # Remove inline comment if present
    num_phases_list = [int(p) for p in num_phases_clean.split(',')]
    feature_matrix_dir = 'feature_matrix'  # Global feature_matrix at root

    # Get pcap dir under input_dir
    pcap_dir = os.path.join(args.input_dir, 'pcap')
    if not os.path.exists(pcap_dir):
        print(f"PCAP directory not found: {pcap_dir}")
        sys.exit(1)

    # List all pcap files (no .pcap extension, but treat as pcap)
    pcap_files = [f for f in os.listdir(pcap_dir) if f.startswith('cap') and not f.endswith('.csv')]  # Assume all cap* are pcaps
    if not pcap_files:
        print(f"No PCAP files found in {pcap_dir}")
        sys.exit(1)

    # Dataset dir under the date dir
    dataset_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(dataset_dir, exist_ok=True)

    # Save config to dataset_dir for traceability
    with open(os.path.join(dataset_dir, 'config.ini'), 'w') as f:
        full_config.write(f)

    print(f"Processing {len(pcap_files)} PCAP files in {args.input_dir}")

    for pcap_file in pcap_files:
        pcap_path = os.path.join(pcap_dir, pcap_file)
        pcap_basename = pcap_file  # e.g., capEC2AMAZ-O4EL3NG-172.31.69.29

        # Step 1: Check if feature_matrix exists for this pcap; if not, extract and build matrices
        all_features_extracted = all(os.path.exists(os.path.join(feature_matrix_dir, ft, f'{pcap_basename}_matrices.npz')) for ft in feature_types)
        if not all_features_extracted:
            print(f"Feature matrices not found for {pcap_basename}. Running FeatureExtractor and SingleFeatureMatrixBuilder.")
            extractor = FeatureExtractor(config)
            features = extractor.extract_features(pcap_path, feature_types, feature_matrix_dir, store=True)

            matrices_data = {}
            for ft in feature_types:
                feature_data = features[ft] if isinstance(features, dict) else features
                builder = SingleFeatureMatrixBuilder(config)
                matrices = builder.build_matrices(feature_data, ft, feature_matrix_dir, pcap_basename, store=True)
                matrices_data[ft] = matrices
        else:
            print(f"Feature matrices found for {pcap_basename}. Skipping extraction and matrix building.")
            matrices_data = {}
            for ft in feature_types:
                npz_path = os.path.join(feature_matrix_dir, ft, f'{pcap_basename}_matrices.npz')
                data = np.load(npz_path, allow_pickle=True)
                matrices_data[ft] = {k: v.item() for k, v in data.items()}

        # Step 2: Fuse features if merged_matrix not exists
        merged_dir = os.path.join(dataset_dir, 'merged_matrix')
        merged_path = os.path.join(merged_dir, f'{pcap_basename}_matrices_fused.npz')
        if not os.path.exists(merged_path):
            print(f"Fusing features for {pcap_basename}.")
            fusion = FeatureFusion(config)
            fused_data = fusion.fuse_features(matrices_data, feature_types, merged_dir, pcap_basename, store=True)
        else:
            print(f"Merged matrix found for {pcap_basename}. Loading.")
            fused_data = np.load(merged_path, allow_pickle=True)
            fused_data = {k: v for k, v in fused_data.items()}

        # Step 3: For each num_phases, run the remaining pipeline
        for num_phases in num_phases_list:
            phase_base_dir = os.path.join(dataset_dir, f'{num_phases}_phase')
            os.makedirs(phase_base_dir, exist_ok=True)

            # Step 3.1: Divide phases if phase_marks not exists
            marks_dir = os.path.join(phase_base_dir, 'phase_marks')
            marks_path = os.path.join(marks_dir, f'{pcap_basename}_matrices_fused_phase_marks.json')
            if not os.path.exists(marks_path):
                print(f"Dividing phases for {num_phases} phases on {pcap_basename}.")
                config['pss']['num_phases'] = str(num_phases)  # Temporarily update for this run
                divider = PhaseDivider(config)
                phase_marks = divider.divide_phases(fused_data, marks_dir, pcap_basename, store=True)
            else:
                print(f"Phase marks found for {num_phases} phases on {pcap_basename}. Loading.")
                with open(marks_path, 'r') as f:
                    phase_marks = json.load(f)

            # Step 3.2: Reconstruct phased pcaps (output with .pcap extension)
            phased_pcap_root = os.path.join(phase_base_dir, 'phased_pcap')
            phased_pcap_basename = f'{pcap_basename}.pcap'  # Add .pcap for output
            if not all(os.path.exists(os.path.join(phased_pcap_root, f'phase_{ph}', f'p_{ph}_{phased_pcap_basename}')) for ph in range(1, num_phases + 1)):
                print(f"Reconstructing phased pcaps for {num_phases} phases on {pcap_basename}.")
                config['pss']['num_phases'] = str(num_phases)
                recon = PhaseReconstructor(config)
                recon.reconstruct_phases(phase_marks, pcap_path, phased_pcap_root, phased_pcap_basename, store=True)

            # Step 3.3: Run CFM on phased pcaps
            cfm_output = os.path.join(phase_base_dir, 'cfm_features')
            if not all(os.path.exists(os.path.join(cfm_output, f'phase_{ph}', f'p_{ph}_{phased_pcap_basename}_Flow.csv')) for ph in range(1, num_phases + 1)):
                print(f"Running CFMRunner for {num_phases} phases on {pcap_basename}.")
                runner = CFMRunner(config)
                runner.run_cfm_on_phased_pcaps(phase_base_dir, num_phases, store=True)

            # Step 3.4: Concatenate features
            concat_output = os.path.join(phase_base_dir, 'concat_csv')
            concat_path = os.path.join(concat_output, f'{phased_pcap_basename}_Flow_concat.csv')  # Match _Flow_concat.csv
            if not os.path.exists(concat_path):
                print(f"Running FeatureConcatenator for {num_phases} phases on {pcap_basename}.")
                concatenator = FeatureConcatenator(config)
                concatenator.concatenate_features(phase_base_dir, num_phases, store=True)

            # Step 3.5: Auto label
            labeled_output = os.path.join(phase_base_dir, 'labeled_csv')
            labeled_path = os.path.join(labeled_output, f'{phased_pcap_basename}_Flow_labeled.csv')  # Match _Flow_labeled.csv
            if not os.path.exists(labeled_path):
                print(f"Running AutoLabeler for {num_phases} phases on {pcap_basename}.")
                labeler = AutoLabeler(config)
                labeler.label_features(phase_base_dir, num_phases, store=True)

    print(f"Pipeline completed for {args.input_dir} under {dataset_dir}.")

if __name__ == '__main__':
    import time
    start_time = time.time()
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        end_time = time.time()
        print(f"Total execution time: {end_time - start_time} seconds")