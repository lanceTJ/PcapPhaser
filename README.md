# Network Traffic Dataset Generation System

## Overview
This project implements a high-performance dataset generation system for network traffic based on phase segmentation using the PSS algorithm. It processes PCAP files, performs phase division, feature extraction, fusion, and labeling to output CSV datasets for machine learning tasks, following the provided documentation.




# PcapPhaser: High-Performance Phased Traffic Dataset Generation System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**PcapPhaser** is a modular, high-performance dataset generation system for traffic phase analysis. It implements the full PSS (Potential Starting Point Scoring) pipeline: feature extraction → single-feature matrix → fusion → dynamic programming phase division → phased pcap reconstruction → CICFlowMeter feature generation → phase-level feature concatenation.

All modules are completely decoupled, support independent execution, breakpoint resumption, and full configuration traceability — fully aligned with the design principles in the official documentation.

## Table of Contents
- [Directory Structure](#directory-structure)
- [Dependencies](#dependencies)
  - [CICFlowMeter (Standard 80+ Flow Features)](#cicflowmeter-standard-80-flow-features)
- [Quick Start](#quick-start)
- [Pipeline Overview](#pipeline-overview)
- [Module Description](#module-description)
- [License](#license)

## Directory Structure (Multi-Phase Experiment Isolation)
```text
datasets/
└── feature_set_1/
    ├── 4_phase/                  # One complete experiment (num_phases=4)
    │   ├── phased_pcap/
    │   │   ├── phase_1/          # Split pcap files
    │   │   └── ...
    │   ├── cfm_features/         # CICFlowMeter output
    │   │   ├── phase_1/
    │   │   └── ...
    │   ├── phase_marks.json      # PhaseDivider output
    │   ├── fused_matrix.npz      # FeatureFusion output
    │   └── config.json           # Full configuration of this experiment
    └── 3_phase/                  # Another independent experiment (num_phases=3)
```

## Features
- Modular design with decoupled stages for reusability and fault tolerance.
- Supports multi-phase segmentation (p values) and feature fusion.
- Persistent intermediate results in a hierarchical directory structure.
- GPU/Numba acceleration for PSS matrix computations.
- Automated labeling based on CIC-IDS2018 rules.

## Installation
1. Clone the repository: `git clone <repo_url>`
2. Install dependencies: `pip install -r requirements.txt`
3. Manually install CicFlowMeter (external tool) and set path in config.ini.
4. Prepare data directory with PCAP files under data/day1/, etc.

## Dependencies

### CICFlowMeter (Standard 80+ Flow Features)

This project uses the **official latest CICFlowMeter** (Canadian Institute for Cybersecurity, University of New Brunswick, MIT License) to generate standard flow features, ensuring 100% consistency with CIC-IDS2017/2018 datasets.

- Official repository: https://github.com/ahlashkari/CICFlowMeter
- License: MIT (see `third_party/cicflowmeter/LICENSE_CICFlowMeter.txt`)

#### One-Click Local Build (Required only once, <30 seconds)

```bash
cd third_party/cicflowmeter
bash build_and_install.sh      # Recommended: covers 99% scenarios
```

**features:**
- Source code is cached by default (perfect for unstable networks/offline debugging)
- Forces fresh clone only when `--download_cfm 1` is specified
- Uses pure-Java jnetpcap bundled in the official repo → zero external downloads, never 404

#### All Usage Scenarios

| Scenario                          | Command Example                                   | Description                                                                 |
|-----------------------------------|---------------------------------------------------|-----------------------------------------------------------------------------|
| First time use (requires internet) | `bash build_and_install.sh --download_cfm 1`     | Force download latest source + compile (recommended on first run)          |
| Daily debugging (recommended)     | `bash build_and_install.sh`                      | No arguments → reuses cached source, completes in seconds (best for unstable network/offline) |
| Force update to latest source     | `bash build_and_install.sh -d 1`                 | Equivalent to `--download_cfm 1`, cleans old source and re-clones           |
| Windows user (Git Bash)           | `bash build_and_install.sh`                      | Fully supported, auto-detects Windows jnetpcap                              |
| WSL / Docker                      | `bash build_and_install.sh`                      | Same as Linux                                                               |
| Clean everything and start over   | `rm -rf temp_cicflowmeter_build cicflowmeter.jar && bash build_and_install.sh --download_cfm 1` | Full clean rebuild                                                          |

#### Verify Success

```bash
# Check generated JAR (~15 MB)
ls -lh third_party/cicflowmeter/cicflowmeter.jar

# Test execution (should print help)
java -jar third_party/cicflowmeter/cicflowmeter.jar -h
```

After successful compilation, `CFMRunner.py` can be used directly with zero configuration.

## Quick Start

```bash
# 1. Compile CICFlowMeter (only once)
cd third_party/cicflowmeter && bash build_and_install.sh

# 2. Run the full pipeline (example: 4 phases)
python main_pipeline.py --config configs/example_4phase.json --run
```

## Pipeline Overview

1. `FeatureExtractor` → per-packet feature sequences
2. `SingleFeatureMatrixBuilder` → U, M, J matrices per feature
3. `FeatureFusion` → weighted fusion of J matrices
4. `PhaseDivider` → dynamic programming phase division
5. `PhaseReconstructor` → generate phased pcap files
6. `CFMRunner` → run official CICFlowMeter on each phase
7. `FeatureConcatenator` → concatenate phase CSVs → final p×80 feature table
8. `AutoLabeler` / `Exporter` → labeling and dataset export

Each module is completely independent and supports individual execution + `.writing` integrity flags.

## Module Description

| Module                     | Input                    | Output                          | Core Function                                 | Reusability |
|----------------------------|--------------------------|---------------------------------|-----------------------------------------------|-------------|
| FeatureExtractor           | Raw pcap                 | Packet-level feature sequences  | Extract length, IAT, direction, rate, etc.    | High        |
| SingleFeatureMatrixBuilder | Feature sequences        | U, M, J matrices                | Welford online computation + Numba acceleration| High        |
| FeatureFusion              | Multiple J matrices      | Fused J matrix                  | Normalization + weighted sum                  | High        |
| PhaseDivider               | Fused J                  | phase_marks.json                | Dynamic programming phase segmentation        | High        |
| PhaseReconstructor         | Original pcap + marks    | Phased pcap directories        | Reconstruct per-phase pcap files              | Medium      |
| CFMRunner                  | Phased pcap              | Per-phase CSV (80+ features)    | Call official CICFlowMeter                    | High        |
| FeatureConcatenator        | Per-phase CSVs           | Final p×80 feature table        | Concatenate + add phase label                 | High        |

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

CICFlowMeter is used under its original MIT License (see `third_party/cicflowmeter/LICENSE_CICFlowMeter.txt`).

---


