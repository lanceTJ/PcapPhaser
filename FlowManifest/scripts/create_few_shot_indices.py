#!/usr/bin/env python
"""
Create few-shot support set indices.

Usage:
    python create_few_shot_indices.py --manifest data/manifest.csv --k 50 --seeds 0 1 2 3 4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from FlowManifest.manifest_builder import ManifestBuilder
from FlowManifest.support_set import SupportSetGenerator


def parse_args():
    parser = argparse.ArgumentParser(description="Create few-shot support set indices")
    parser.add_argument("--manifest", type=str, required=True, help="Path to manifest.csv")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--k", type=int, default=50, help="Number of shots per class")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4], help="Random seeds")
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

    # Get few-shot config
    few_shot_config = config.get("few_shot", {})

    k = args.k if args.k is not None else few_shot_config.get("k", 50)
    seeds = args.seeds if args.seeds else few_shot_config.get("seeds", [0, 1, 2, 3, 4])

    # Load manifest
    manifest_path = Path(args.manifest)
    print(f"Loading manifest from: {manifest_path}")
    manifest = ManifestBuilder.load(str(manifest_path))

    # Create generator
    output_dir = Path(args.output_dir) if args.output_dir else manifest_path.parent / "indices"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate for each seed
    generator = SupportSetGenerator(seed=seeds[0])

    print(f"Creating {len(seeds)} support sets with k={k}")
    support_sets = generator.generate_multiple_seeds(
        manifest,
        k=k,
        seeds=seeds,
        label_balanced=few_shot_config.get("label_balanced", True),
    )

    # Save
    for seed, support_set in zip(seeds, support_sets):
        output_path = output_dir / f"support_k{k}_seed{seed}.csv"
        generator.save(support_set, str(output_path))
        print(f"Saved support set (seed={seed}) to: {output_path}")
        print(f"  Contains {len(support_set.parent_flow_ids)} parent flows")

    # Summary
    print("\nSummary:")
    print(f"  Generated {len(support_sets)} support sets")
    print(f"  k = {k} shots per class")
    print(f"  Seeds: {seeds}")


if __name__ == "__main__":
    main()
