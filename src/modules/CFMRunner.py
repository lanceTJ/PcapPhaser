# filename: CFMRunner.py
import argparse
import sys
import os
import subprocess
import shutil
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional

class CFMRunner:
    """
    Class for running CICFlowMeter (Java version) on phased pcaps to generate standard 80+ flow features.
    Fully compatible with .writing integrity flag and multi-phase dataset isolation.
    """
    def __init__(self, config: dict = None):
        """
        :param config: Dict with optional 'cfm' section:
                       'jar_path' (str): Path to cicflowmeter.jar
                       'java_cmd' (str): Java executable (default 'java')
                       'max_workers' (int): Parallel threads (default = os.cpu_count() or 4)
                       'timeout_min' (int): Per-file timeout in minutes (default 30)
        """
        default_jar = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'third_party', 'cicflowmeter', 'cicflowmeter.jar')
        if config is not None and 'cfm' in config:
            self.jar_path = config['cfm'].get('jar_path', default_jar)
            self.java_cmd = config['cfm'].get('java_cmd', 'java')
            self.max_workers = config['cfm'].get('max_workers', max(4, os.cpu_count() or 4))
            self.timeout_min = config['cfm'].get('timeout_min', 30)
        else:
            self.jar_path = default_jar
            self.java_cmd = 'java'
            self.max_workers = max(4, os.cpu_count() or 4)
            self.timeout_min = 30

        if not os.path.exists(self.jar_path):
            raise FileNotFoundError(f"CICFlowMeter JAR not found at {self.jar_path}")

    def run_cfm_on_phased_pcaps(self,
                                phase_base_dir: str,
                                num_phases: int,
                                store: bool = True) -> Dict[int, str]:
        """
        Run CICFlowMeter on all phased pcaps under a specific phase experiment directory.
        :param phase_base_dir: Path like 'datasets/feature_set_1/4_phase'
        :param num_phases: Number of phases (used to verify directory structure)
        :param store: Whether to execute CFM (False for dry-run)
        :return: Dict {phase_num: cfm_output_dir}
        """
        phased_pcap_root = os.path.join(phase_base_dir, 'phased_pcap')
        cfm_output_root = os.path.join(phase_base_dir, 'cfm_features')
        os.makedirs(cfm_output_root, exist_ok=True)

        tasks = []
        for ph in range(1, num_phases + 1):
            input_dir = os.path.join(phased_pcap_root, f'phase_{ph}')
            output_dir = os.path.join(cfm_output_root, f'phase_{ph}')
            os.makedirs(output_dir, exist_ok=True)
            if not os.path.exists(input_dir):
                logging.warning(f"Phase {ph} pcap directory not exists: {input_dir}")
                continue
            tasks.append((ph, input_dir, output_dir))

        if not tasks:
            print(f"No phased pcap found under {phased_pcap_root}")
            return {}

        print(f"Starting CICFlowMeter on {len(tasks)} phase directories using {self.max_workers} workers")
        results = {}
        if store:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self._process_single_phase, ph, in_dir, out_dir): ph
                    for ph, in_dir, out_dir in tasks
                }
                for future in as_completed(futures):
                    ph = futures[future]
                    success = future.result()
                    results[ph] = os.path.join(cfm_output_root, f'phase_{ph}') if success else None

        print(f"CFM processing completed for {phase_base_dir}")
        return results

    def _process_single_phase(self, phase_num: int, input_dir: str, output_dir: str) -> bool:
        """
        Process all .pcap files in one phase directory using CICFlowMeter.
        Uses .writing flag for integrity.
        """
        pcap_files = [f for f in os.listdir(input_dir) if f.endswith('.pcap')]
        if not pcap_files:
            logging.info(f"No pcap in phase {phase_num}, skip")
            return True

        writing_flag = os.path.join(output_dir, '.cfm_processing')
        if os.path.exists(writing_flag):
            print(f"Phase {phase_num} is already being processed or failed before, skip")
            return False

        open(writing_flag, 'w').close()
        success = False
        try:
            # CICFlowMeter command: batch mode
            cmd = [
                self.java_cmd, '-jar', self.jar_path,
                '-i', input_dir,
                '-o', output_dir,
                '-f', 'csv'
            ]
            print(f"Running CFM on phase {phase_num}: {' '.join(cmd)}")
            timeout_sec = self.timeout_min * 60
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)

            if result.returncode == 0:
                csv_count = len([f for f in os.listdir(output_dir) if f.endswith('.csv')])
                print(f"Phase {phase_num} completed, generated {csv_count} CSV files")
                success = True
            else:
                print(f"Phase {phase_num} CFM failed: {result.stderr[-500:]}")
        except subprocess.TimeoutExpired:
            print(f"Phase {phase_num} CFM timeout after {self.timeout_min} minutes")
        except Exception as e:
            print(f"Phase {phase_num} CFM exception: {e}")
        finally:
            if os.path.exists(writing_flag):
                os.remove(writing_flag)
        return success

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run CICFlowMeter on phased pcaps')
    parser.add_argument('-d', '--dataset_dir', type=str, required=True,
                        help='Full path to phase dataset dir, e.g., datasets/feature_set_1/4_phase')
    parser.add_argument('--num_phases', type=int, required=True, help='Number of phases')
    parser.add_argument('--run', action='store_true', help='Execute CFM now')

    args = parser.parse_args()
    if args.run:
        config = {'cfm': {'max_workers': 8, 'timeout_min': 60}}
        runner = CFMRunner(config)
        runner.run_cfm_on_phased_pcaps(args.dataset_dir, args.num_phases, store=True)