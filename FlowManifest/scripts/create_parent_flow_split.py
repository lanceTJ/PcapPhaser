#!/usr/bin/env python
"""
Create parent flow split.

Usage:
    python create_parent_flow_split.py --manifest data/manifest.csv --seed 42
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from FlowManifest.manifest_builder import ManifestBuilder
from FlowManifest.split_manager import SplitManager


def parse_args():
    parser = argparse.ArgumentParser(description="Create parent flow split")
    parser.add_argument("--manifest", type=str, required=True, help="Path to manifest.csv")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--train-ratio", type=float, help="Train ratio")
    parser.add_argument("--val-ratio", type=float, help="Val ratio")
    parser.add_argument("--test-ratio", type=float, help="Test ratio")
    parser.add_argument("--group-aware", action="store_true", help="Use group-aware split")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    return parser.parse_args()


def load_config(config_path: str):
    """Load config from YAML."""
    if config_path and Path(config_path).exists():
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}


def main():
    args = parse_args()
    config = load_config(args.config)

    # Get split config
    split_config = config.get("split", {})

    seed = args.seed if args.seed is not None else split_config.get("seed", 42)
    train_ratio = args.train_ratio if args.train_ratio is not None else split_config.get("train_ratio", 0.7)
    val_ratio = args.val_ratio if args.val_ratio is not None else split_config.get("val_ratio", 0.1)
    test_ratio = args.test_ratio if args.test_ratio is not None else split_config.get("test_ratio", 0.2)
    group_aware = args.group_aware if args.group_aware is not None else split_config.get("group_aware", True)

    # Load manifest
    manifest_path = Path(args.manifest)
    print(f"Loading manifest from: {manifest_path}")
    manifest = ManifestBuilder.load(str(manifest_path))

    # Create split manager
    manager = SplitManager(seed=seed)

    # Perform split
    print(f"Creating split with seed={seed}, train={train_ratio}, val={val_ratio}, test={test_ratio}")
    split = manager.split_manifest(
        manifest,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        group_aware=group_aware,
        stratify=split_config.get("stratify_by_label", True),
    )

    # Save updated manifest
    manifest.save(str(manifest_path))
    print(f"Updated manifest saved to: {manifest_path}")

    # Save split file
    output_dir = Path(args.output_dir) if args.output_dir else manifest_path.parent / "splits"
    output_dir.mkdir(parents=True, exist_ok=True)
    split_path = output_dir / f"parent_split_seed{seed}.csv"
    manager.save_split(split, str(split_path))
    print(f"Split file saved to: {split_path}")

    # Summary
    print("\nSplit summary:")
    print(f"  Train: {len(split.train)} parent flows")
    print(f"  Val: {len(split.val)} parent flows")
    print(f"  Test: {len(split.test)} parent flows")

    # Label distribution summary
    df = manifest.to_dataframe()
    if "split" in df.columns and "label" in df.columns:
        print("\nLabel distribution by split:")
        summary = df.groupby(["split", "label"]).size().unstack(fill_value=0)
        print(summary.to_string())


if __name__ == "__main__":
    main()
