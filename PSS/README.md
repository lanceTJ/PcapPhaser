# PSS: High-Performance Phased Traffic Dataset Generation System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)

## Overview

**PSS** is a high-performance dataset generation system for network traffic analysis based on **phase segmentation** using the PSS (Potential Starting Point Scoring) algorithm.

Given raw PCAP traffic, PSS performs:
- packet-level feature extraction,
- feature fusion and dynamic-programming-based phase division,
- phased PCAP reconstruction,
- standard flow feature generation,
- and phase-level dataset export for downstream machine learning tasks.

All modules are **fully decoupled**, support **independent execution**, **breakpoint resumption**, and **full configuration traceability**, following reproducible system design principles.

---

## Paper Reference

This repository is the **artifact** for the following paper:

> **Title**: *FlowPhaser: Towards Robust Traffic Analysis by Uncovering Intrinsic Flow Phases*
> **Authors**: Anonymous (under review)
> **Conference**: USENIX Security 2027
> **Status**: Artifact for reproducibility

This artifact supports reproducing:
- The complete **phase segmentation pipeline** described in the paper
- **Dataset generation results** used for analysis
- **Phase-level feature statistics** reported in downstream experiments

---

## Artifact Capabilities

- Full implementation of the **PSS-based traffic phase segmentation pipeline**
- End-to-end dataset generation from **PCAP → phased PCAP → CSV features**
- **Deterministic phase boundary reproduction** given identical inputs and configurations
- Configuration files aligned with those used in the paper
- Compact public benign and malicious DoHBrw captures for direct evaluation
- PGCL-ready labeled phase-feature export for downstream model training

---

## Directory Structure (Multi-Phase Experiment Isolation)

```text
datasets/
└── feature_set_1/
    ├── 3_phase/                  # One complete experiment (num_phases = 3)
    │   ├── phased_pcap/
    │   │   ├── phase_1/
    │   │   └── ...
    │   ├── cfm_features/         # CICFlowMeter outputs
    │   │   ├── phase_1/
    │   │   └── ...
    │   ├── phase_marks.json      # PhaseDivider output
    │   ├── fused_matrix.npz      # FeatureFusion output
    │   └── config.ini            # Full experiment configuration
    └── 4_phase/                  # Independent experiment (num_phases = 4)
```

Each experiment directory is **self-contained** and fully reproducible.

---

## Features

* Modular pipeline with decoupled, reusable stages
* Support for arbitrary numbers of traffic phases
* Persistent intermediate results for fault tolerance
* Numba acceleration for PSS matrix computation
* Automated labeling based on CIC-IDS2018-compatible rules

---

## Installation

### Requirements

* Python ≥ 3.12
* Linux (tested on Ubuntu 24.04)
* Java (for CICFlowMeter)
* ≥16 GB RAM recommended

### Setup

```bash
git clone <anonymous-artifact-url> FlowPhaser
cd FlowPhaser/PSS
pip install -r requirements.txt
```

---

## Dependencies

### CICFlowMeter (Standard 80+ Flow Features)

PSS uses the **official CICFlowMeter** released by the Canadian Institute for Cybersecurity (UNB) to generate standard flow features, ensuring consistency with CIC-IDS2017/2018 datasets.

* Official repository: [https://github.com/ahlashkari/CICFlowMeter](https://github.com/ahlashkari/CICFlowMeter)
* License: MIT (see the upstream [`LICENSE.txt`](https://github.com/ahlashkari/CICFlowMeter/blob/master/LICENSE.txt))

#### One-Time Local Build (< 30 seconds)

```bash
chmod +x third_party/cicflowmeter/build_cicflowmeter_with_runtime_bundle.sh
bash third_party/cicflowmeter/build_cicflowmeter_with_runtime_bundle.sh
```

After successful compilation, `CFMRunner.py` can be used without additional configuration.

---

## Quick Start (Artifact Evaluation)

### Running the Pipeline

```bash
# Compile CICFlowMeter (only once)
bash third_party/cicflowmeter/build_cicflowmeter_with_runtime_bundle.sh

# Run the included DoHBrw demonstration captures (3 phases)
python src/labeled_by_file_name_pipeline.py \
  --config configs/demo.ini \
  --input_dir ../PcapPerturbator/demo_inputs/demo \
  --output_dir ../artifact_outputs/pss \
  --dataset dohbrw-demo \
  --run
```

Run the flow-key and extraction regression tests with:

```bash
python -m unittest discover -s tests -v
```

### Expected Outputs

After successful execution, the following artifacts are generated below
`../artifact_outputs/pss/dohbrw-demo/3_phase/`:

* `phase_marks/*_phase_marks.json` — phase boundary results
* `phased_pcap/phase_*/` — reconstructed per-phase PCAP files
* `cfm_features/phase_*/` — per-phase flow-level CSV features
* `concat_csv/` — concatenated phase feature tables
* `labeled_csv/*_Flow_labeled.csv` — PGCL-ready labeled tables

The resolved experiment configuration is stored at
`../artifact_outputs/pss/dohbrw-demo/config.ini`.

These outputs correspond to the dataset generation process described in the paper.

---

## Pipeline Overview

1. **FeatureExtractor** — packet-level feature extraction
2. **SingleFeatureMatrixBuilder** — U / M / J matrix construction
3. **FeatureFusion** — weighted fusion of feature matrices
4. **PhaseDivider** — dynamic-programming-based phase segmentation
5. **PhaseReconstructor** — per-phase PCAP reconstruction
6. **CFMRunner** — CICFlowMeter execution
7. **FeatureConcatenator** — phase-level feature concatenation
8. **AutoLabeler / Exporter** — labeling and dataset export

---

## Compatibility Notes

### CICFlowMeter Timestamp Issues

CICFlowMeter may produce timestamps that cause labeling inconsistencies.

#### 1. Timezone Offset Issue

* **Detection**: Start/end times differ from original PCAP by exactly *N* hours
* **Solution**: Enable timezone correction in `AutoLabeler`

#### 2. AM/PM Ambiguity

* **Detection**: Inconsistent 12h/24h timestamp interpretation
* **Solution**: Validate timestamp formats before labeling

Each module operates independently and supports execution integrity checks via `.writing` flags.

---

## Module Description

| Module                     | Input             | Output                    | Core Function                           | Reusability |
| -------------------------- | ----------------- | ------------------------- | --------------------------------------- | ----------- |
| FeatureExtractor           | Raw PCAP          | Packet feature sequences  | Length, IAT, direction, rate extraction | High        |
| SingleFeatureMatrixBuilder | Feature sequences | U / M / J matrices        | Online statistics + Numba acceleration  | High        |
| FeatureFusion              | J matrices        | Fused matrix              | Normalization and weighted fusion       | High        |
| PhaseDivider               | Fused matrix      | phase_marks.json          | Dynamic programming phase segmentation  | High        |
| PhaseReconstructor         | PCAP + marks      | Phased PCAP files         | Phase-based reconstruction              | Medium      |
| CFMRunner                  | Phased PCAP       | Per-phase CSV features    | CICFlowMeter invocation                 | High        |
| FeatureConcatenator        | Phase CSVs        | Phase-level feature table | Concatenation and labeling              | High        |

---

## License and Open Science Statement

This artifact is released under the **MIT License** to support open science and reproducible research.

All third-party components are used in compliance with their original licenses.

If you use this artifact in academic work, please cite the associated paper.

---
