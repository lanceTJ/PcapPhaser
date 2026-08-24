#!/usr/bin/env python
"""
Check for data leakage.

Usage:
    python check_data_leakage.py --manifest data/manifest.csv --indices-dir data/indices --strict
"""
from __future__ import annotations

import argparse
from pathlib import Path

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from FlowManifest.manifest_builder import ManifestBuilder
from FlowManifest.leakage_checker import LeakageChecker
from FlowManifest.support_set import SupportSetGenerator


def parse_args():
    parser = argparse.ArgumentParser(description="Check for data leakage")
    parser.add_argument("--manifest", type=str, required=True, help="Path to manifest.csv")
    parser.add_argument("--indices-dir", type=str, help="Directory with index files (support sets)")
    parser.add_argument("--strict", action="store_true", help="Strict mode (fail on warnings)")
    parser.add_argument("--allow-group-overlap", action="store_true", help="Allow group overlap across splits")
    parser.add_argument("--feature-file", type=str, help="CSV file with features to check")
    return parser.parse_args()


def find_index_files(indices_dir: Path) -> list[Path]:
    """Find index files in directory."""
    if not indices_dir.exists():
        return []

    # Look for support set files
    support_files = list(indices_dir.glob("support_*.csv"))

    # Look for pretrain pool files
    pretrain_files = list(indices_dir.glob("pretrain_*.csv"))

    return support_files + pretrain_files


def main():
    args = parse_args()

    manifest_path = Path(args.manifest)
    print(f"Checking manifest: {manifest_path}")

    # Load manifest
    manifest = ManifestBuilder.load(str(manifest_path))

    # Load index files if specified
    support_sets = None
    if args.indices_dir:
        indices_dir = Path(args.indices_dir)
        index_files = find_index_files(indices_dir)

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


if __name__ == "__main__":
    main()
