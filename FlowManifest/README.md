# FlowManifest: Parent-Flow Manifest-Based Split Pipeline

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)

## Overview

**FlowManifest** is a reviewer-friendly, reproducible, leakage-free data splitting pipeline at the parent-flow level for traffic analysis research.

Key features:
- Stable parent-flow ID generation ensuring origin and perturbed variants share the same ID
- Unified manifest management for all data views
- Parent-flow level train/val/test splits to prevent data leakage
- Label-balanced few-shot support set generation
- PA-FT (Perturbation-Aware Fine-Tuning) balanced sampling
- Automatic feature sanitization to prevent shortcut learning
- Built-in leakage checking and validation

## Project Structure

```
FlowManifest/
├── README.md              # This file
├── pyproject.toml         # Python package configuration
├── .gitignore             # Git ignore rules
├── src/
│   └── FlowManifest/      # Core library package
│       ├── __init__.py    # Public API exports
│       ├── cli.py         # Unified command line interface
│       ├── manifest_builder.py
│       ├── split_manager.py
│       ├── support_set.py
│       ├── leakage_checker.py
│       ├── dataset.py
│       ├── samplers.py
│       ├── sanitization.py
│       ├── parent_flow_id.py
│       ├── data_store.py
│       └── pcap_integration.py
├── scripts/               # Legacy command-line scripts (for backward compatibility)
│   ├── build_manifest.py
│   ├── create_parent_flow_split.py
│   ├── create_few_shot_indices.py
│   └── check_data_leakage.py
├── configs/               # Configuration files
│   └── data_split.yaml
├── docs/                  # Additional documentation
└── tests/                 # Core smoke tests
```

## Paper Reference

This is the **data splitting and manifest management component** for the following paper:

> **Title**: *FlowPhaser: Towards Robust Traffic Analysis by Uncovering Intrinsic Flow Phases*
> **Authors**: Anonymous (under review)
> **Conference**: USENIX Security 2027
> **Status**: Artifact for reproducibility

This artifact supports:
- Parent-flow level data splitting without leakage
- Manifest-based data management
- Reproducible few-shot support set generation
- Leakage validation for research artifacts

## Artifact Capabilities

- Complete implementation of the core **manifest, split, support-set, and leakage-checking pipeline**
- Deterministic data splitting with fixed seeds
- Built-in leakage detection and validation
- PyTorch Dataset integration for easy usage
- Group-aware splitting for origin and all five perturbation variants
- A deterministic 144-row example with executable CLI commands
- CSV index views that connect directly to PGCL training inputs

## Key Design Principles

### Parent-Flow Level Splitting

Random splitting at the packet, segment, or variant level can cause:
- Same parent flow's origin and perturbed variants appearing in different splits
- Models learning via shortcut memorization instead of true patterns
- Information leakage across perturbation types

FlowManifest ensures all variants of the same parent flow stay in the same split.

### Sanitized Features

In addition to IP/port filtering, we remove:
- TCP sequence numbers, acknowledgments, timestamps
- IP ID, checksum
- Absolute timestamps
- Capture IDs, filenames
- Any metadata that could uniquely identify a specific flow

### Unified Index Views

Instead of creating separate physical datasets for pretraining/finetuning:
- Maintain one unified manifest
- Access data through different index views
- Pretraining uses only origin variants from train split
- Fine-tuning uses support sets from train split

### Shared Support Sets for Clean-FT and PA-FT

For fair comparison:
- Clean-FT (origin-only) and PA-FT (all variants) use identical parent_flow_ids
- PA-FT controls variant mixing via balanced sampling
- Both use the same number of gradient updates

### PA-FT Balanced Sampling Ratio

Desired composition per mini-batch:
- origin: 50%
- case1_loss: 10%
- case2_retransmission: 10%
- case3_reordering: 10%
- case4_length_padding: 10%
- case5_timing_delay: 10%

## Quick Start (Artifact Evaluation)

### Unified CLI (Recommended)

Install the core manifest and splitting pipeline from this directory:

```bash
pip install -e .
```

PyTorch dataset/sampler support and PCAP integration are optional extras:

```bash
pip install -e ".[ml]"    # PyTorch integration
pip install -e ".[pcap]"  # PCAP integration
pip install -e ".[all]"   # Both
```

Generate a deterministic demonstration manifest template:

```bash
python examples/generate_tiny_manifest.py \
  --output ../artifact_outputs/flowmanifest/manifest_template.csv
```

```bash
# Step 1: Initialize manifest from an experiment-specific template
flowmanifest build \
    --config configs/data_split.yaml \
    --dataset FlowPhaser-demo \
    --template ../artifact_outputs/flowmanifest/manifest_template.csv \
    --output-dir ../artifact_outputs/flowmanifest

# Step 2: Create parent-flow level splits
flowmanifest split \
    --manifest ../artifact_outputs/flowmanifest/manifest.csv \
    --group-aware \
    --seed 42

# Step 3: Create few-shot support sets
flowmanifest fewshot \
    --manifest ../artifact_outputs/flowmanifest/manifest.csv \
    --k 2 \
    --seeds 0 1

# Step 4: Check for data leakage
flowmanifest check \
    --manifest ../artifact_outputs/flowmanifest/manifest.csv \
    --indices-dir ../artifact_outputs/flowmanifest/indices \
    --strict
```

### Legacy Scripts (Backward Compatible)

The original scripts are still available for backward compatibility:

```bash
# Step 1: Initialize manifest from an experiment-specific template
python scripts/build_manifest.py \
    --config configs/data_split.yaml \
    --dataset CIC-IDS-2018 \
    --template data/processed/CIC-IDS-2018/manifest_template.csv

# Step 2: Create parent-flow level splits
python scripts/create_parent_flow_split.py \
    --manifest data/processed/CIC-IDS-2018/manifest.csv \
    --seed 42

# Step 3: Create few-shot support sets
python scripts/create_few_shot_indices.py \
    --manifest data/processed/CIC-IDS-2018/manifest.csv \
    --k 50 \
    --seeds 0 1 2 3 4

# Step 4: Check for data leakage
python scripts/check_data_leakage.py \
    --manifest data/processed/CIC-IDS-2018/manifest.csv \
    --indices-dir data/processed/CIC-IDS-2018/indices \
    --strict
```

### Command Help

For more information on any command, use:

```bash
flowmanifest --help
flowmanifest build --help
flowmanifest split --help
flowmanifest fewshot --help
flowmanifest check --help
```

Run the core test suite with:

```bash
python -m unittest discover -s tests -v
```

## Data Organization

### Manifest Format

The `manifest.csv` contains these columns:

| Column | Description |
|--------|-------------|
| dataset | Dataset name |
| parent_flow_id | Stable hash ID for parent flow |
| variant_id | parent_flow_id + variant |
| variant | origin / case1_loss / ... / case5_timing_delay |
| split | train / val / test |
| label | Label name |
| label_id | Label ID |
| group_id | Group ID (capture_id / device_id / ...) |
| source_pcap | Source PCAP path |
| processed_path | Processed CSV path |
| num_packets | Number of packets |
| start_time | Start timestamp |
| end_time | End timestamp |
| perturbation_config | Perturbation config JSON |
| perturbation_applied | Whether perturbation was applied |
| feature_schema_version | Feature schema version |
| sanitization_version | Sanitization version |
| checksum | Checksum |

### Directory Structure

```
data/processed/{dataset}/
├── manifest.csv              # Unified manifest file
├── splits/
│   └── parent_split_seed42.csv
├── indices/
│   ├── pretrain_pool_seed42.csv
│   ├── support_k50_seed0.csv
│   ├── support_k50_seed1.csv
│   └── ...
├── pcap/                     # PCAP files (origin + variants)
│   ├── origin/
│   ├── case1_loss/
│   ├── case2_retransmission/
│   ├── case3_reordering/
│   ├── case4_length_padding/
│   └── case5_timing_delay/
└── csv/                      # PSS-processed feature CSVs
    ├── origin/
    └── ...
```

## Usage Examples

### PCAP Integration

```python
from pathlib import Path
from FlowManifest import PcapManifestPipeline

# These adapters must invoke the exact PcapPerturbator/PSS commands and configs
# used by the experiment, then create output_pcap/output_csv respectively.
def run_perturbation(input_pcap, output_pcap, plan):
    ...

def run_pss(input_pcap, output_csv):
    ...

pipeline = PcapManifestPipeline(
    dataset_name="CIC-IDS-2018",
    raw_pcap_dir=Path("data/raw"),
    processed_dir=Path("data/processed"),
    seed=42,
    perturbation_runner=run_perturbation,
    pss_runner=run_pss,
)

# Process all PCAPs and auto-generate manifest
pipeline.process_all_pcaps(
    label_map={
        "capture1": "attack",
        "capture2": "benign",
    }
)
```

### Import from Existing Perturbed PCAPs

```python
from FlowManifest import create_manifest_from_existing_pcaps

manifest = create_manifest_from_existing_pcaps(
    processed_dir="data/processed",
    dataset_name="CIC-IDS-2018",
    pcap_dir_structure={
        "origin": "data/processed/pcap/origin",
        "case1_loss": "data/processed/pcap/case1_loss",
        "case2_retransmission": "data/processed/pcap/case2_retransmission",
        "case3_reordering": "data/processed/pcap/case3_reordering",
        "case4_length_padding": "data/processed/pcap/case4_length_padding",
        "case5_timing_delay": "data/processed/pcap/case5_timing_delay",
    }
)

manifest.save("data/processed/manifest.csv")
```

### Dataset Loading with PyTorch

```python
from FlowManifest.dataset import ManifestDataset, DatasetMode

# Pretrain mode (train split, origin only)
pretrain_dataset = ManifestDataset(
    manifest_path="data/manifest.csv",
    mode=DatasetMode.PRETRAIN,
)

# Clean-FT mode (support set, origin only)
clean_ft_dataset = ManifestDataset(
    manifest_path="data/manifest.csv",
    mode=DatasetMode.CLEAN_FT,
    index_path="data/indices/support_k50_seed0.csv",
)

# PA-FT mode (support set, all variants)
pa_ft_dataset = ManifestDataset(
    manifest_path="data/manifest.csv",
    mode=DatasetMode.PA_FT,
    index_path="data/indices/support_k50_seed0.csv",
)
```

### Using PA-FT Balanced Sampler

```python
from FlowManifest.samplers import BalancedVariantBatchSampler

# Create sampler
sampler = BalancedVariantBatchSampler(
    dataset_indices=list(range(len(dataset))),
    parent_flow_ids=[...],
    variants=[...],
    batch_size=32,
    seed=42,
)

# Create DataLoader
from torch.utils.data import DataLoader

loader = DataLoader(dataset, batch_sampler=sampler)
```

### Checking for Data Leakage

```python
from FlowManifest.leakage_checker import LeakageChecker
from FlowManifest.manifest_builder import ManifestBuilder

# Load manifest
manifest = ManifestBuilder.load("data/manifest.csv")

# Check for leakage
checker = LeakageChecker()
report = checker.check(manifest, support_sets=[...])

print(report)
assert report.is_valid
```

## Configuration

See `configs/data_split.yaml` for all configuration options.

## Validation Report

After running `check_data_leakage.py --strict`, the generated report validates:
- ✓ All variants of same parent_flow_id have same split
- ✓ Same group_id does not cross splits
- ✓ Support sets are only from train split
- ✓ All banned features have been removed
- ✓ Every parent flow has origin variant in manifest

## License and Open Science Statement

This artifact is released under the **MIT License** to support open science and reproducible research.

If you use FlowManifest in academic work, please cite the associated paper.
