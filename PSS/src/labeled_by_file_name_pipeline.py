import argparse
import sys
import os
import signal
import configparser
import traceback
import numpy as np
import json
import multiprocessing
import time
import copy
import pandas as pd  # Added for SimpleLabeler to handle CSV

# Import all module classes (excluding AutoLabeler)
from modules.FeatureExtractor import FeatureExtractor
from modules.SingleFeatureMatrixBuilder import SingleFeatureMatrixBuilder
from modules.FeatureFusionBuilder import FeatureFusion
from modules.PhaseDivider import PhaseDivider
from modules.PhaseReconstructor import PhaseReconstructor
from modules.CFMRunner import CFMRunner
from modules.FeatureConcatenator import FeatureConcatenator
from modules.utils import load_config

PCAP_TIMEOUT_SEC = 6 * 3600

def clean_incomplete_files(output_dir):
    """
    Recursively scan the output directory for .writing files and delete both the .writing file
    and the corresponding main file (without .writing suffix) if it exists.
    """
    deleted_files = []
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file.endswith(".writing"):
                writing_path = os.path.join(root, file)
                main_file = file[:-8]  # Remove '.writing' suffix
                main_path = os.path.join(root, main_file)
                if os.path.exists(main_path):
                    try:
                        os.remove(main_path)
                        deleted_files.append(main_path)
                    except OSError as e:
                        print(f"[Cleanup] Failed to delete main file {main_path}: {e}")
                try:
                    os.remove(writing_path)
                    deleted_files.append(writing_path)
                except OSError as e:
                    print(
                        f"[Cleanup] Failed to delete writing file {writing_path}: {e}"
                    )
    if deleted_files:
        print(f"[Cleanup] Deleted {len(deleted_files)} incomplete files: ")
        for file in deleted_files:
            print(f"[Cleanup] -->{file}<--")
    else:
        print("[Cleanup] No incomplete files found.")


class SimpleLabeler:
    """
    Simple labeler for datasets labeled by filename: adds 'Label' column to CSV based on PCAP filename.
    - If filename starts with 'Benign', label as 'Benign'.
    - Otherwise, label as 'Attack'.
    """

    def __init__(self, config: dict):
        self.config = config

    def label_features(
        self,
        phase_base_dir: str,
        num_phases: int,
        pcap_basename: str,
        store: bool = False,
    ) -> pd.DataFrame:
        """
        Load concatenated CSV, add 'Label' column, and optionally store to labeled_csv.
        """
        concat_output = os.path.join(phase_base_dir, "concat_csv")
        concat_path = os.path.join(concat_output, f"{pcap_basename}_Flow_concat.csv")

        if not os.path.exists(concat_path):
            raise FileNotFoundError(f"Concatenated CSV not found: {concat_path}")

        df = pd.read_csv(concat_path)

        # Determine label based on pcap_basename
        if "benign" in pcap_basename.lower():
            label = "Benign"
        else:
            label = "Malicious"

        # Add 'Label' column to every row
        df["Label"] = label

        if store:
            labeled_output = os.path.join(phase_base_dir, "labeled_csv")
            os.makedirs(labeled_output, exist_ok=True)
            labeled_path = os.path.join(
                labeled_output, f"{pcap_basename}_Flow_labeled.csv"
            )
            df.to_csv(labeled_path, index=False)
            print(f"[SimpleLabeler] Labeled CSV stored at {labeled_path}")

        return df


def process_pcap(
    pcap_file,
    args,
    config,
    feature_types,
    num_phases_list,
    feature_matrix_dir,
    dataset_dir,
    pcap_dir,
):
    """
    Process a single PCAP file through the entire pipeline.
    This function is called in parallel for each PCAP.
    """
    pcap_path = None
    try:
        pcap_path = os.path.join(pcap_dir, pcap_file)
        pcap_basename = pcap_file  # e.g., Benign-Device1.pcap (includes .pcap)

        # Step 1: Check if feature matrices exist; if not, extract and build
        all_features_extracted = all(
            os.path.exists(
                os.path.join(feature_matrix_dir, ft, f"{pcap_basename}_matrices.npz")
            )
            for ft in feature_types
        )
        if not all_features_extracted:
            print(
                f"[Process {os.getpid()}] Feature matrices not found for {pcap_basename}. Running FeatureExtractor and SingleFeatureMatrixBuilder."
            )
            extractor = FeatureExtractor(config)
            features = extractor.extract_features(
                pcap_path, feature_types, feature_matrix_dir, store=True
            )

            matrices_data = {}
            for ft in feature_types:
                feature_data = features[ft] if isinstance(features, dict) else features
                builder = SingleFeatureMatrixBuilder(config)
                matrices = builder.build_matrices(
                    feature_data, ft, feature_matrix_dir, pcap_basename, store=True
                )
                matrices_data[ft] = matrices
        else:
            print(
                f"[Process {os.getpid()}] Feature matrices found for {pcap_basename}. Skipping extraction and matrix building."
            )
            matrices_data = {}
            for ft in feature_types:
                npz_path = os.path.join(
                    feature_matrix_dir, ft, f"{pcap_basename}_matrices.npz"
                )
                data = np.load(npz_path, allow_pickle=True)
                matrices_data[ft] = {k: v.item() for k, v in data.items()}

        # Step 2: Fuse features if merged_matrix does not exist
        merged_dir = os.path.join(dataset_dir, "merged_matrix")
        os.makedirs(merged_dir, exist_ok=True)  # Ensure directory exists safely
        merged_path = os.path.join(merged_dir, f"{pcap_basename}_fused.npz")
        if not os.path.exists(merged_path):
            print(f"[Process {os.getpid()}] Fusing features for {pcap_basename}.")
            fusion = FeatureFusion(config)
            fused_data = fusion.fuse_features(
                matrices_data, feature_types, merged_dir, pcap_basename, store=True
            )
        else:
            print(
                f"[Process {os.getpid()}] Merged matrix found for {pcap_basename}. Loading."
            )
            fused_data = np.load(merged_path, allow_pickle=True)
            fused_data = {k: v for k, v in fused_data.items()}

        # Step 3: For each num_phases, run the remaining pipeline
        for num_phases in num_phases_list:
            if num_phases < 2:
                print(
                    f"[Process {os.getpid()}] Skipping invalid num_phases: {num_phases}"
                )
                continue
            phase_base_dir = os.path.join(dataset_dir, f"{num_phases}_phase")
            os.makedirs(phase_base_dir, exist_ok=True)

            # Step 3.1: Divide phases if phase_marks does not exist
            marks_dir = os.path.join(phase_base_dir, "phase_marks")
            os.makedirs(marks_dir, exist_ok=True)
            marks_path = os.path.join(marks_dir, f"{pcap_basename}_phase_marks.json")
            if not os.path.exists(marks_path):
                print(
                    f"[Process {os.getpid()}] Dividing phases for {num_phases} phases on {pcap_basename}."
                )
                local_config = copy.deepcopy(
                    config
                )  # Deep copy to avoid modifying shared config
                local_config["pss"][
                    "num_phases"
                ] = num_phases  # Temporarily update for this run
                divider = PhaseDivider(local_config)
                phase_marks = divider.divide_phases(
                    fused_data, marks_dir, pcap_basename, store=True
                )
            else:
                print(
                    f"[Process {os.getpid()}] Phase marks found for {num_phases} phases on {pcap_basename}. Loading."
                )
                with open(marks_path, "r") as f:
                    phase_marks = json.load(f)

            # Step 3.2: Reconstruct phased pcaps (output with .pcap extension)
            phased_pcap_root = os.path.join(phase_base_dir, "phased_pcap")
            os.makedirs(phased_pcap_root, exist_ok=True)
            phased_pcap_basename = pcap_basename  # Keep .pcap for output
            phase_paths = [
                os.path.join(
                    phased_pcap_root, f"phase_{ph}", f"p_{ph}_{phased_pcap_basename}"
                )
                for ph in range(1, num_phases + 1)
            ]
            if not all(os.path.exists(p) for p in phase_paths):
                print(
                    f"[Process {os.getpid()}] Reconstructing phased pcaps for {num_phases} phases on {pcap_basename}."
                )
                local_config = copy.deepcopy(config)
                local_config["pss"]["num_phases"] = num_phases
                recon = PhaseReconstructor(local_config)
                recon.reconstruct_phases(
                    phase_marks,
                    pcap_path,
                    phased_pcap_root,
                    phased_pcap_basename,
                    store=True,
                )
            else:
                print(
                    f"[Process {os.getpid()}] Phased pcaps found for {num_phases} phases on {pcap_basename}. Skipping reconstruction."
                )

            # Step 3.3: Run CFM on phased pcaps
            cfm_output = os.path.join(phase_base_dir, "cfm_features")
            os.makedirs(cfm_output, exist_ok=True)
            cfm_paths = [
                os.path.join(
                    cfm_output, f"phase_{ph}", f"p_{ph}_{phased_pcap_basename}_Flow.csv"
                )
                for ph in range(1, num_phases + 1)
            ]
            if not all(os.path.exists(p) for p in cfm_paths):
                print(
                    f"[Process {os.getpid()}] Running CFMRunner for {num_phases} phases on {pcap_basename}."
                )
                runner = CFMRunner(config)
                runner.run_cfm_on_single_basename(
                    phase_base_dir, num_phases, phased_pcap_basename, store=True
                )
            else:
                print(
                    f"[Process {os.getpid()}] CFM features found for {num_phases} phases on {pcap_basename}. Skipping CFM run."
                )

            # Step 3.4: Concatenate features + (optional) label by filename, directly output labeled CSV
            labeled_output = os.path.join(phase_base_dir, "labeled_csv")
            os.makedirs(labeled_output, exist_ok=True)
            labeled_path = os.path.join(
                labeled_output, f"{phased_pcap_basename}_Flow_labeled.csv"
            )

            if not os.path.exists(labeled_path):
                print(
                    f"[Process {os.getpid()}] Running FeatureConcatenator (label_by_file_name=True) for {num_phases} phases on {pcap_basename}."
                )
                concatenator = FeatureConcatenator(config)
                concatenator.concatenate_single_basename(
                    phase_base_dir,
                    num_phases,
                    f"{phased_pcap_basename}_Flow",
                    store=True,
                    label_by_file_name=True,
                )
            else:
                print(
                    f"[Process {os.getpid()}] Labeled features found for {num_phases} phases on {pcap_basename}. Skipping concat+label."
                )

        print(f"[Process {os.getpid()}] Completed processing for {pcap_basename}.")
    except Exception as e:
        import traceback
        print(f"[Process {os.getpid()}] Error processing {pcap_basename} ({pcap_path}): {e}", flush=True)
        traceback.print_exc()
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Full pipeline for generating phased datasets from PCAP files using PSS modules for datasets labeled by filename."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config.ini file."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to input date directory, e.g., pcapdata/2023-data.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to output date directory, e.g., workspace/2023-data",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="dataset1",
        help="Dataset name for output directory under each date dir.",
    )
    parser.add_argument("--run", action="store_true", help="Run the pipeline now.")

    args = parser.parse_args()

    if not args.run:
        print("Use --run to execute the pipeline.")
        sys.exit(0)

    # Load unified config
    config = load_config(args.config)  # Returns dict from utils.py
    print(f"[PipeLine] Configuration loaded: {config}")
    # Clean incomplete files before starting
    clean_incomplete_files(args.output_dir)
    # Extend config with sections for all modules
    full_config = configparser.ConfigParser()
    full_config.read(args.config)

    # Extract key params from config
    feature_types = config.get("pss", {}).get(
        "allowed_feature_names",
        ["packet_length", "inter_arrival_time", "up_down_rate"],
    )
    if isinstance(feature_types, str):
        feature_types = [ft.strip() for ft in feature_types.split(",")]
    num_phases_list = config.get("pss", {}).get("num_phases", [2, 3, 4])
    feature_matrix_dir = os.path.join(
        args.output_dir, "feature_matrix"
    )  # Global feature_matrix at output_dir
    os.makedirs(feature_matrix_dir, exist_ok=True)

    # Get pcap dir under input_dir
    candidate_pcap_dir = os.path.join(args.input_dir, "pcap")
    if os.path.exists(candidate_pcap_dir):
        pcap_dir = candidate_pcap_dir
        print(f"[PipeLine] Using PCAP directory: {pcap_dir}")
    else:
        pcap_dir = args.input_dir
        print(
            f"[PipeLine] 'pcap' subdir not found, falling back to input_dir: {pcap_dir}"
        )

    if not os.path.exists(pcap_dir):
        print(f"[PipeLine] PCAP directory not found: {pcap_dir}")
        sys.exit(1)

    # List all pcap files (ending with .pcap)
    pcap_files = [f for f in os.listdir(pcap_dir) if f.endswith(".pcap")]
    if not pcap_files:
        print(f"[PipeLine] No PCAP files found in {pcap_dir}")
        sys.exit(1)

    # Dataset dir under the date dir
    dataset_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(dataset_dir, exist_ok=True)

    # Save config to dataset_dir for traceability
    with open(os.path.join(dataset_dir, "config.ini"), "w") as f:
        full_config.write(f)

    print(f"[PipeLine] Processing {len(pcap_files)} PCAP files in {args.input_dir}")

    # Use multiprocessing Pool for parallel processing
    num_processes = config.get("pipeline", {}).get("pcap_workers", 20)

    def _ignore_sigint():
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    pool = multiprocessing.Pool(processes=num_processes, initializer=_ignore_sigint)

    results = []
    for pcap_file in pcap_files:
        ar = pool.apply_async(
            process_pcap,
            args=(pcap_file, args, config, feature_types, num_phases_list,
                feature_matrix_dir, dataset_dir, pcap_dir),
        )
        results.append((pcap_file, ar))

    ok = 0
    failed = 0
    failed_pcaps = []
    try:
        for pcap_file, ar in results:
            try:
                ar.get(timeout=PCAP_TIMEOUT_SEC)
                ok += 1
            except multiprocessing.TimeoutError:
                failed += 1
                failed_pcaps.append(pcap_file)
                print(f"[PipeLine] TIMEOUT on {pcap_file}. Terminating pool now.")
                pool.terminate()
                pool.join()
                raise
            except Exception:
                failed += 1
                failed_pcaps.append(pcap_file)
                print(f"[PipeLine] Worker crashed on {pcap_file}:\n{traceback.format_exc()}")
    finally:
        if pool._state not in ("TERMINATE", "CLOSE"):
            pool.close()
        pool.join()

    print(f"[PipeLine] Done. ok={ok} failed={failed}")
    if failed_pcaps:
        print("[PipeLine] Failed PCAP list:")
        for f in failed_pcaps:
            print(f"  - {f}")
        raise RuntimeError(
            f"PSS pipeline failed for {failed} of {len(pcap_files)} PCAP files"
        )

if __name__ == "__main__":
    start_time = time.time()
    try:
        main()
    except Exception as e:
        print(f"[PipeLine] An error occurred: {e}")
        raise e
    finally:
        end_time = time.time()
        print(f"[PipeLine] Total execution time: {end_time - start_time} seconds")
