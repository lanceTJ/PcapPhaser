#!/usr/bin/env python3
"""
Bridge script: Apply FlowManifest split mapping to PSS feature CSVs.

FlowManifest outputs a split mapping CSV (parent_flow_id -> train/val/test).
PSS outputs feature CSVs with phase-level columns (e.g., Flow Duration_p1).
This script joins them and produces three clean CSVs ready for PGCL training.

Usage:
    python prepare_flowmanifest_splits.py \
        --features-csv /path/to/pss/labeled_features.csv \
        --split-csv /path/to/flowmanifest/split_mapping.csv \
        --flow-key-col "Flow Key" \
        --output-dir ./pgcl_splits/

Output:
    ./pgcl_splits/train.csv
    ./pgcl_splits/val.csv
    ./pgcl_splits/test.csv

These CSVs can be fed directly to PGCL via:
    python main.py \
        --train-config configs/train.yaml \
        --train-csv ./pgcl_splits/train.csv \
        --val-csv ./pgcl_splits/val.csv \
        --test-csv ./pgcl_splits/test.csv \
        --output-dir ./outputs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Apply FlowManifest splits to PSS features")
    parser.add_argument("--features-csv", type=str, required=True,
                        help="PSS output CSV with phase features (e.g., labeled_features.csv)")
    parser.add_argument("--split-csv", type=str, required=True,
                        help="FlowManifest split mapping CSV (parent_flow_id, split)")
    parser.add_argument("--flow-key-col", type=str, default="Flow Key",
                        help="Column name in features CSV that matches parent_flow_id")
    parser.add_argument("--split-id-col", type=str, default="parent_flow_id",
                        help="Column name in split CSV containing flow identifier")
    parser.add_argument("--split-col", type=str, default="split",
                        help="Column name in split CSV containing split label")
    parser.add_argument("--output-dir", type=str, default="./pgcl_splits",
                        help="Directory to write train.csv / val.csv / test.csv")
    parser.add_argument("--drop-cols", type=str, nargs="+",
                        default=["Timestamp"],
                        help="Non-feature columns to drop before output")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading features: {args.features_csv}")
    features_df = pd.read_csv(args.features_csv)
    print(f"  -> {len(features_df)} rows, {len(features_df.columns)} columns")

    print(f"Loading split mapping: {args.split_csv}")
    splits_df = pd.read_csv(args.split_csv)
    print(f"  -> {len(splits_df)} rows")

    # Validate required columns
    if args.flow_key_col not in features_df.columns:
        print(f"ERROR: Flow key column '{args.flow_key_col}' not found in features CSV.", file=sys.stderr)
        print(f"Available columns: {list(features_df.columns)}", file=sys.stderr)
        sys.exit(1)

    for col in (args.split_id_col, args.split_col):
        if col not in splits_df.columns:
            print(f"ERROR: Column '{col}' not found in split CSV.", file=sys.stderr)
            print(f"Available columns: {list(splits_df.columns)}", file=sys.stderr)
            sys.exit(1)

    # Join on flow identifier
    merged = features_df.merge(
        splits_df[[args.split_id_col, args.split_col]],
        left_on=args.flow_key_col,
        right_on=args.split_id_col,
        how="inner",
    )
    print(f"Joined -> {len(merged)} rows matched")

    unmatched = len(features_df) - len(merged)
    if unmatched > 0:
        print(f"  WARNING: {unmatched} feature rows had no matching split entry")

    # Drop join keys and non-feature columns
    drop_cols = [args.flow_key_col, args.split_id_col] + args.drop_cols
    drop_cols = [c for c in drop_cols if c in merged.columns]
    if drop_cols:
        merged = merged.drop(columns=drop_cols)
        print(f"Dropped columns: {drop_cols}")

    # Write per-split CSVs
    split_counts = {}
    for split_name in ("train", "val", "test"):
        split_df = merged[merged[args.split_col] == split_name]
        split_counts[split_name] = len(split_df)

        out_path = out_dir / f"{split_name}.csv"
        split_df = split_df.drop(columns=[args.split_col], errors="ignore")
        split_df.to_csv(out_path, index=False)
        print(f"  {split_name}: {len(split_df)} rows -> {out_path}")

    print(f"\nDone. Outputs in: {out_dir.resolve()}")
    print(f"Summary: train={split_counts['train']}, val={split_counts['val']}, test={split_counts['test']}")


if __name__ == "__main__":
    main()
