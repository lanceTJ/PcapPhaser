# FlowPhaser Artifact Evaluation Guide

## Artifact at a glance

FlowPhaser provides four interoperable components for controlled traffic
perturbation, phase-aware feature construction, parent-flow-safe data
management, and phase-guided contrastive learning. The repository includes two
compact CIRA-CIC-DoHBrw-2020 captures and a cross-platform smoke evaluator that
exercises the principal software interfaces without requiring Bash.

## Hardware and software

- Python 3.12
- CPU with 8 GB RAM for the smoke evaluation
- Linux, macOS, or Windows for the Python smoke evaluation
- Java and Maven for complete CICFlowMeter-backed PSS dataset generation
- CUDA-capable GPU recommended for paper-scale PGCL training

## Installation

From the repository root:

```bash
python -m venv .venv
```

Activate the environment with the command for your shell:

```bash
# Linux or macOS
source .venv/bin/activate

# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

Then install the dependencies and run the evaluator:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python artifact/run_smoke.py
```

The smoke evaluation uses CPU-only PGCL settings and normally finishes in a few
minutes after dependencies are installed. Generated files are written to
`artifact_outputs/smoke/`.

## Evaluation claims

| ID | Capability | Evaluation performed | Expected result |
|---|---|---|---|
| E1 | Deterministic PCAP perturbation | Apply seeded packet loss to the included attack capture and compare packet multisets | A valid output PCAP, metadata JSON, and a positive loss count |
| E2 | Direction-safe PSS flow construction | Run PSS flow-key and extraction regression tests | Two PSS tests pass |
| E3 | Parent-flow-safe experiment management | Build a 144-row manifest, assign group-aware splits, generate support sets, and run strict leakage checks | `All checks passed!` and split/index CSVs |
| E4 | PGCL training and evaluation | Generate deterministic three-phase features, run contrastive pretraining and two-stage fine-tuning on CPU, then evaluate a held-out test split | Training log, pretraining and fine-tuned checkpoints, metadata, and bounded metrics |

The final line of a successful run is:

```text
FlowPhaser artifact smoke evaluation: PASS
```

## Optional graphical perturbation evaluation

On Ubuntu/Debian, the complete PcapPerturbator demonstration prepares the
TrafficManipulator compatibility layer, runs all five perturbations, and
generates packet-length and per-flow IAT figures:

```bash
cd PcapPerturbator
sudo apt-get install build-essential python3-dev python3-venv
bash quicksetup.sh
bash TM1_loss.sh
bash TM1_retrans.sh
bash TM1_reorder.sh
bash TM2_length.sh
bash TM2_rate.sh
```

The tested demo plans preserve all 9,060 packets in both TM2 outputs. The
length experiment preserves timestamps; the rate experiment preserves packet
contents and lengths. Fresh figures and JSON summaries are written to `vis/`,
and inspectable examples are included under `reference_outputs/`.

## Outputs

The evaluator produces:

```text
artifact_outputs/smoke/
├── pcapperturbator/
│   ├── demo/cap_attack.pcap
│   └── TM1_loss_stats.json
├── flowmanifest/
│   ├── manifest.csv
│   ├── splits/parent_split_seed42.csv
│   └── indices/support_k2_seed*.csv
├── pgcl/
│   ├── smoke_3_train_pgcl_phase.csv
│   ├── smoke_3_best_pgcl_phase.safetensors
│   ├── smoke_3_best_pgcl_phase_finetuned.safetensors
│   └── smoke_3_best_pgcl_phase_metrics.json
└── smoke_summary.json
```

## Component customization

- Change a perturbation plan under `PcapPerturbator/plans/`.
- Change PSS phase count and feature weights in `PSS/configs/demo.ini`.
- Change split ratios and support-set sizes through the FlowManifest CLI.
- Change PGCL model and optimizer settings in `PGCL/configs/train.yaml`.

## Dataset attribution

The compact demo captures are redistributed under the terms stated by the
CIRA-CIC-DoHBrw-2020 publisher. Exact provenance, packet ranges, sizes, hashes,
and the required citation are recorded in
`PcapPerturbator/demo_inputs/DATASET.md`.
