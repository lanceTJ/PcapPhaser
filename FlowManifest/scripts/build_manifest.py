#!/usr/bin/env python
"""
Initialize a manifest from an existing template CSV.

Usage:
    python build_manifest.py --config configs/data_split.yaml --dataset CIC-IDS-2018
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import yaml

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from FlowManifest.manifest_builder import ManifestBuilder


def parse_args():
    parser = argparse.ArgumentParser(description="Build manifest from processed data")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--input-dir", type=str, help="Input directory with processed data")
    parser.add_argument("--output-dir", type=str, help="Output directory for manifest")
    parser.add_argument("--template", type=str, help="Existing manifest template CSV")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def load_config(config_path: str) -> Dict:
    """Load config from YAML."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_manifest_from_existing(
    manifest_template_path: Path,
    dataset_name: str,
) -> ManifestBuilder:
    """Build manifest from an existing template or CSV."""
    if manifest_template_path.exists():
        print(f"Loading existing manifest from: {manifest_template_path}")
        return ManifestBuilder.load(str(manifest_template_path))
    return ManifestBuilder(dataset_name=dataset_name)


def main():
    args = parse_args()
    config = load_config(args.config)

    dataset_name = args.dataset
    input_dir = Path(args.input_dir) if args.input_dir else None
    output_dir = Path(args.output_dir) if args.output_dir else None

    # Default paths
    if input_dir is None:
        data_config = config.get("dataset", {})
        input_dir = Path(data_config.get("processed_dir", "data/processed")) / dataset_name

    if output_dir is None:
        output_dir = input_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    template_path = (
        Path(args.template)
        if args.template
        else output_dir / "manifest_template.csv"
    )
    if not template_path.is_file():
        raise FileNotFoundError(
            "A manifest template is required. Pass --template PATH or place "
            f"manifest_template.csv at {output_dir}."
        )
    builder = build_manifest_from_existing(template_path, dataset_name)

    # Validate
    is_valid, issues = builder.validate()
    if not is_valid:
        print("Manifest validation warnings:")
        for issue in issues:
            print(f"  - {issue}")

    # Save
    manifest_path = output_dir / "manifest.csv"
    builder.save(str(manifest_path))
    print(f"Manifest saved to: {manifest_path}")

    # Summary
    df = builder.to_dataframe()
    print("Manifest summary:")
    print(f"  Total entries: {len(df)}")
    if "variant" in df.columns:
        print(f"  Variants:\n{df['variant'].value_counts().to_string()}")
    if "split" in df.columns and df["split"].notna().any():
        print(f"  Splits:\n{df['split'].value_counts().to_string()}")
    if "label" in df.columns:
        print(f"  Labels:\n{df['label'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
