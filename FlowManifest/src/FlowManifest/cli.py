#!/usr/bin/env python
"""
FlowManifest: Unified Command Line Interface

Parent-flow manifest-based split pipeline for reproducible, leakage-free traffic analysis.

Usage:
    flowmanifest build --config configs/data_split.yaml --dataset CIC-IDS-2018 --template manifest_template.csv
    flowmanifest split --manifest data/manifest.csv --seed 42
    flowmanifest fewshot --manifest data/manifest.csv --k 50
    flowmanifest check --manifest data/manifest.csv --strict
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

import yaml


def add_sys_path():
    """Ensure FlowManifest is importable."""
    src_path = str(Path(__file__).parent.parent)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


add_sys_path()

from FlowManifest.manifest_builder import ManifestBuilder
from FlowManifest.split_manager import SplitManager
from FlowManifest.support_set import SupportSetGenerator
from FlowManifest.leakage_checker import LeakageChecker


def load_config(config_path: Optional[str]) -> Dict:
    """Load config from YAML if provided."""
    if config_path and Path(config_path).exists():
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}


def build_manifest(args):
    """Initialize a manifest from an explicit template CSV."""
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

    print(f"Building manifest for dataset: {dataset_name}")
    print(f"Scanning directory: {input_dir}")

    template_path = (
        Path(args.template)
        if args.template
        else output_dir / "manifest_template.csv"
    )
    if not template_path.is_file():
        raise FileNotFoundError(
            "A manifest template is required because processed-data layouts are "
            "experiment specific. Pass --template PATH or place "
            f"manifest_template.csv at {output_dir}."
        )
    print(f"Loading existing manifest template from: {template_path}")
    builder = ManifestBuilder.load(str(template_path))

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
    print("\nManifest summary:")
    print(f"  Total entries: {len(df)}")
    if "variant" in df.columns and not df["variant"].isna().all():
        print(f"  Variants:\n{df['variant'].value_counts().to_string()}")
    if "split" in df.columns and df["split"].notna().any():
        print(f"  Splits:\n{df['split'].value_counts().to_string()}")
    if "label" in df.columns:
        print(f"  Labels:\n{df['label'].value_counts().to_string()}")


def create_split(args):
    """Create parent-flow split subcommand."""
    config = load_config(args.config)
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


def create_fewshot(args):
    """Create few-shot support sets subcommand."""
    config = load_config(args.config)
    few_shot_config = config.get("few_shot", {})

    k = args.k if args.k is not None else few_shot_config.get("k", 50)
    seeds = args.seeds if args.seeds else few_shot_config.get("seeds", [0, 1, 2, 3, 4])

    # Load manifest
    manifest_path = Path(args.manifest)
    print(f"Loading manifest from: {manifest_path}")
    manifest = ManifestBuilder.load(str(manifest_path))

    # Output dir
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


def check_leakage(args):
    """Check for data leakage subcommand."""
    manifest_path = Path(args.manifest)
    print(f"Checking manifest: {manifest_path}")

    # Load manifest
    manifest = ManifestBuilder.load(str(manifest_path))

    # Load index files if specified
    support_sets = None
    if args.indices_dir:
        indices_dir = Path(args.indices_dir)
        if indices_dir.exists():
            # Find index files
            support_files = list(indices_dir.glob("support_*.csv"))
            pretrain_files = list(indices_dir.glob("pretrain_*.csv"))
            index_files = support_files + pretrain_files

            if index_files:
                print(f"Found {len(index_files)} index files")
                generator = SupportSetGenerator()
                support_sets = []

                for idx_path in index_files:
                    if "support" in idx_path.name:
                        try:
                            ss = generator.load(str(idx_path))
                            support_sets.append(ss)
                            print(f"  Loaded: {idx_path.name}")
                        except Exception as e:
                            print(f"  Warning: Could not load {idx_path.name}: {e}")

    # Load feature columns if specified
    feature_columns = None
    if args.feature_file:
        import pandas as pd
        feature_df = pd.read_csv(args.feature_file, nrows=0)
        feature_columns = list(feature_df.columns)
        print(f"Loaded {len(feature_columns)} feature columns from: {args.feature_file}")

    # Run checks
    checker = LeakageChecker()
    report = checker.check(
        manifest=manifest,
        support_sets=support_sets,
        feature_columns=feature_columns,
        allow_group_overlap=args.allow_group_overlap,
    )

    # Print report
    print("\n" + "=" * 60)
    print("LEAKAGE CHECK REPORT")
    print("=" * 60)
    print(report)

    # Exit with error code if failed
    if not report.is_valid:
        print("\nERROR: Leakage checks failed!")
        sys.exit(1)

    # Check warnings in strict mode
    warnings = [i for i in report.issues if i.severity == "warning"]
    if args.strict and warnings:
        print(f"\nERROR: Strict mode - {len(warnings)} warnings found")
        sys.exit(1)

    print("\nAll checks passed!")
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description="FlowManifest: Parent-flow manifest-based split pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  flowmanifest build --config configs/data_split.yaml --dataset CIC-IDS-2018 --template manifest_template.csv
  flowmanifest split --manifest data/processed/CIC-IDS-2018/manifest.csv --seed 42
  flowmanifest fewshot --manifest data/processed/CIC-IDS-2018/manifest.csv --k 50
  flowmanifest check --manifest data/processed/CIC-IDS-2018/manifest.csv --strict
"""
    )

    # Subparsers
    subparsers = parser.add_subparsers(title="Commands", dest="command", required=True)

    # Build manifest command
    parser_build = subparsers.add_parser("build", help="Initialize manifest from a template CSV")
    parser_build.add_argument("--config", type=str, help="Path to config file")
    parser_build.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser_build.add_argument("--input-dir", type=str, help="Input directory with processed data")
    parser_build.add_argument("--output-dir", type=str, help="Output directory for manifest")
    parser_build.add_argument("--template", type=str, help="Existing manifest template CSV")
    parser_build.add_argument("--seed", type=int, default=42, help="Random seed")
    parser_build.set_defaults(func=build_manifest)

    # Split command
    parser_split = subparsers.add_parser("split", help="Create parent-flow train/val/test splits")
    parser_split.add_argument("--manifest", type=str, required=True, help="Path to manifest.csv")
    parser_split.add_argument("--config", type=str, help="Path to config file")
    parser_split.add_argument("--seed", type=int, default=42, help="Random seed")
    parser_split.add_argument("--train-ratio", type=float, help="Train ratio")
    parser_split.add_argument("--val-ratio", type=float, help="Val ratio")
    parser_split.add_argument("--test-ratio", type=float, help="Test ratio")
    parser_split.add_argument("--group-aware", action="store_true", help="Use group-aware split")
    parser_split.add_argument("--output-dir", type=str, help="Output directory")
    parser_split.set_defaults(func=create_split)

    # Fewshot command
    parser_fewshot = subparsers.add_parser("fewshot", help="Create few-shot support set indices")
    parser_fewshot.add_argument("--manifest", type=str, required=True, help="Path to manifest.csv")
    parser_fewshot.add_argument("--config", type=str, help="Path to config file")
    parser_fewshot.add_argument("--k", type=int, default=50, help="Number of shots per class")
    parser_fewshot.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4], help="Random seeds")
    parser_fewshot.add_argument("--output-dir", type=str, help="Output directory")
    parser_fewshot.set_defaults(func=create_fewshot)

    # Check command
    parser_check = subparsers.add_parser("check", help="Check for data leakage")
    parser_check.add_argument("--manifest", type=str, required=True, help="Path to manifest.csv")
    parser_check.add_argument("--indices-dir", type=str, help="Directory with index files (support sets)")
    parser_check.add_argument("--strict", action="store_true", help="Strict mode (fail on warnings)")
    parser_check.add_argument("--allow-group-overlap", action="store_true", help="Allow group overlap across splits")
    parser_check.add_argument("--feature-file", type=str, help="CSV file with features to check")
    parser_check.set_defaults(func=check_leakage)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
