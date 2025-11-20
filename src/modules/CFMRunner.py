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
        Directly invoke Java with correct CLASSPATH including all lib/*.jar
        Uses Java wildcard support for lib/* to ensure all dependencies (including jnetpcap) are loaded
        """
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_file_dir, '..', '..'))
        dist_base = os.path.join(project_root, 'third_party', 'cicflowmeter', 'cfm_dist')

        dist_candidates = [d for d in os.listdir(dist_base) if d.startswith('CICFlowMeter')]
        if not dist_candidates:
            raise FileNotFoundError(f"No CICFlowMeter distribution found in {dist_base}")
        self.dist_dir = os.path.join(dist_base, sorted(dist_candidates)[-1])

        self.native_lib_dir = os.path.join(self.dist_dir, 'lib', 'native')
        self.lib_dir = os.path.join(self.dist_dir, 'lib')

        # Critical: Use lib/* wildcard - Java 6+ supports this directly
        self.classpath = os.path.join(self.lib_dir, '*')

        self.max_workers = config['cfm'].get('max_workers', max(4, os.cpu_count() or 4)) if config else max(4, os.cpu_count() or 4)
        self.timeout_min = config['cfm'].get('timeout_min', 30) if config else 30

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
        Execute CICFlowMeter using direct Java invocation with proper CLASSPATH wildcard
        This guarantees all jars including jnetpcap are loaded, no NoClassDefFoundError
        """
        writing_flag = os.path.join(output_dir, '.cfm_processing')
        if os.path.exists(writing_flag):
            print(f"[Phase {phase_num}] Already processing or failed, skipping")
            return False

        open(writing_flag, 'w').close()
        try:
            cmd = [
                'java',
                f'-Djava.library.path={self.native_lib_dir}',  # Ensure native libs can be found
                '-cp', self.classpath,                        # Key fix: lib/* loads ALL jars including jnetpcap
                'cic.cs.unb.ca.ifm.Cmd',
                input_dir,
                output_dir
            ]

            print(f"[Phase {phase_num}] Running: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_min * 60
            )

            if result.returncode == 0:
                csv_count = len([f for f in os.listdir(output_dir) if f.endswith('.csv')])
                print(f"[Phase {phase_num}] Success, generated {csv_count} CSV files")
                return True
            else:
                print(f"[Phase {phase_num}] Failed:\n{result.stderr.strip()}")
                return False
        except Exception as e:
            print(f"[Phase {phase_num}] Exception: {e}")
            return False
        finally:
            if os.path.exists(writing_flag):
                os.remove(writing_flag)

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