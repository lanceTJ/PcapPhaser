#!/usr/bin/env python3
"""Generate deterministic, separable phase-feature CSVs for a CPU smoke run."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


FEATURES = ("duration", "packet_count", "iat_mean", "direction_ratio")


def make_split(size: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for index in range(size):
        malicious = index % 2 == 1
        center = 1.0 if malicious else -1.0
        row = {"Flow ID": f"flow-{seed}-{index:04d}"}
        for phase in range(1, 4):
            phase_shift = 0.15 * phase
            for feature_index, feature in enumerate(FEATURES):
                row[f"{feature}_p{phase}"] = float(
                    center
                    + phase_shift
                    + 0.05 * feature_index
                    + rng.normal(0.0, 0.18)
                )
        row["Label"] = "Malicious" if malicious else "Benign"
        rows.append(row)
    frame = pd.DataFrame(rows)
    return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, size, seed in (
        ("train", 64, 42),
        ("val", 24, 43),
        ("test", 24, 44),
    ):
        path = output_dir / f"{name}.csv"
        make_split(size, seed).to_csv(path, index=False)
        print(f"Wrote {size} rows to {path}")


if __name__ == "__main__":
    main()
