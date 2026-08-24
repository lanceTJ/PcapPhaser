# FlowPhaser: Phase-aware and Robustness-aware Traffic Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)

## Overview

FlowPhaser is a research framework for discovering intrinsic phases in network
flows and evaluating phase-aware traffic representations under controlled
traffic perturbations. It combines four reusable components:

1. **PcapPerturbator** applies deterministic packet loss, retransmission,
   reordering, sequence-offset, length, and timing perturbations to PCAP data.
2. **PSS** extracts packet-level signals, identifies phase boundaries, rebuilds
   per-phase PCAPs, and exports phase-level flow features.
3. **FlowManifest** keeps origin and perturbed variants of the same parent flow
   in leakage-safe train/validation/test partitions and creates reproducible
   few-shot support sets.
4. **PGCL** performs phase-guided contrastive pretraining followed by supervised
   traffic-classification fine-tuning.

The repository is the open-science artifact for:

> *FlowPhaser: Towards Robust Traffic Analysis by Uncovering Intrinsic Flow
> Phases*, USENIX Security 2027 submission (anonymous review).

## Quick artifact evaluation

The cross-platform evaluator verifies the included PCAPs, runs a real seeded
packet-loss experiment, exercises the PSS flow-construction tests, performs the
complete FlowManifest split workflow, and trains/evaluates PGCL on CPU.

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python artifact/run_smoke.py
```

A successful run ends with:

```text
FlowPhaser artifact smoke evaluation: PASS
```

Generated PCAPs, manifests, support sets, training logs, metrics, and model
checkpoints are written to `artifact_outputs/smoke/`. See [ARTIFACT.md](ARTIFACT.md)
for the evaluation claims and output map.

## Included demonstration traffic

Two compact captures from the public CIRA-CIC-DoHBrw-2020 dataset are included:

| Capture | Label | Packets | Size |
|---|---|---:|---:|
| `benign_source.pcap` | Benign browser/DoH traffic | 5,000 | 2.72 MB |
| `cap_attack.pcap` | Malicious DNS2TCP DoH tunnel | 9,060 | 2.10 MB |

Their exact provenance, packet range, SHA-256 hashes, redistribution terms, and
required dataset citation are recorded in
[`PcapPerturbator/demo_inputs/DATASET.md`](PcapPerturbator/demo_inputs/DATASET.md).

## Repository structure

```text
FlowPhaser/
├── artifact/                 # Cross-platform smoke evaluator
├── PcapPerturbator/          # Controlled PCAP perturbation and analysis
│   ├── demo_inputs/          # Compact benign and attack captures
│   ├── plans/                # Reproducible perturbation plans
│   └── scripts/              # Quantitative and graphical comparisons
├── PSS/                      # Phase segmentation and feature construction
│   ├── configs/demo.ini
│   ├── src/
│   └── tests/
├── FlowManifest/             # Parent-flow manifest and safe splitting
│   ├── examples/
│   ├── src/
│   └── tests/
├── PGCL/                     # Contrastive pretraining and classification
│   ├── configs/smoke.yaml
│   ├── examples/
│   └── tests/
├── ARTIFACT.md
├── requirements.txt
└── LICENSE
```

## Component workflow

### 1. Controlled perturbation

PcapPerturbator accepts either quick CLI flags or ordered JSON plans. Each
processed capture is accompanied by a metadata JSON containing the seed,
executed stages, timing, packet counts, verification results, and measured
perturbation strength.

```bash
pcapperturbator \
  --in-root PcapPerturbator/demo_inputs \
  --out-root artifact_outputs/loss \
  --workers 1 \
  --seed 42 \
  --plan PcapPerturbator/plans/loss.json
```

The Linux demonstration scripts provide loss, retransmission, reordering,
length, and rate experiments. The length and rate experiments also generate
packet-length and inter-arrival-time figures.

Pre-generated figures and numerical summaries from the included captures are
available in
[`PcapPerturbator/reference_outputs/`](PcapPerturbator/reference_outputs/README.md).

### 2. Phase segmentation and feature construction

PSS fuses packet length, inter-arrival time, and direction-ratio signals,
computes phase boundaries with dynamic programming, reconstructs phase PCAPs,
and concatenates CICFlowMeter features across phases.

After building the bundled CICFlowMeter runtime, the included captures can be
processed with:

```bash
cd PSS
bash third_party/cicflowmeter/build_cicflowmeter_with_runtime_bundle.sh
python src/labeled_by_file_name_pipeline.py \
  --config configs/demo.ini \
  --input_dir ../PcapPerturbator/demo_inputs/demo \
  --output_dir ../artifact_outputs/pss \
  --dataset dohbrw-demo \
  --run
```

Important outputs include phase marks, per-phase PCAPs, CICFlowMeter CSVs, and
`3_phase/labeled_csv/*_Flow_labeled.csv`.

### 3. Parent-flow-safe data management

FlowManifest creates stable manifests, assigns group-aware train/validation/test
partitions, builds label-balanced few-shot support sets, and checks that parent
flows and their perturbed variants never cross data partitions.

The smoke evaluator generates a deterministic 144-row manifest example and runs
all four CLI stages: `build`, `split`, `fewshot`, and `check --strict`.

### 4. Phase-guided learning

PGCL consumes phase-feature CSVs with `pN_` prefixes or `_pN` suffixes. It
supports binary and multi-class tasks, fixed or externally supplied splits,
K-fold evaluation, balanced training, contrastive pretraining, and two-stage
fine-tuning.

The CPU smoke configuration writes:

- pretraining log and checkpoint;
- restored best fine-tuned encoder/classifier checkpoint;
- feature/scaler/class metadata;
- validation and held-out test Accuracy, Macro-F1, Precision, and Recall.

## Reproducibility controls

- Fixed seeds for perturbation, splitting, support-set generation, and training
- Sidecar perturbation metadata and input hashes
- Parent-flow and group-aware data isolation
- Explicit YAML/INI/JSON configurations
- Atomic model checkpoint and metrics writes
- Unit and executable smoke tests with non-zero failure status

## Requirements

- Python 3.12 for the unified smoke evaluation
- Java and Maven for CICFlowMeter-backed PSS feature generation
- Linux plus a C/C++ compiler and Python development headers for
  TrafficManipulator-assisted length/rate demonstrations
- CPU execution is supported; CUDA is recommended for large PGCL experiments

Component-specific setup and customization are documented in each subdirectory.

## License

FlowPhaser source code is released under the MIT License. The included
CIRA-CIC-DoHBrw-2020 subsets retain their dataset attribution and redistribution
conditions described in `PcapPerturbator/demo_inputs/DATASET.md`.
