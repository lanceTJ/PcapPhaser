#!/usr/bin/env python3
"""
Compare per-flow inter-arrival-time (IAT) distributions for benign, original
attack, and rate-perturbed attack traffic.

This script is designed for TM-2 rate demonstrations where a simple histogram
with log-scaled x-axis often looks visually distorted because IAT is heavy-tailed.
It produces a two-panel figure:
  1) Histogram with log-spaced bins
  2) ECDF on a log-scaled x-axis

It also writes a JSON summary with descriptive statistics and Wasserstein
distances to the benign reference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _pcap_utils import flow_iats


def positive_iats(path: str) -> np.ndarray:
    """Load per-flow IATs and keep finite positive values only."""
    vals = np.asarray(flow_iats(path), dtype=float)
    vals = vals[np.isfinite(vals)]
    vals = vals[vals > 0]
    return vals


def describe(x: np.ndarray) -> dict:
    """Return summary statistics for a 1D array."""
    if x.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "min": None,
            "max": None,
        }
    return {
        "count": int(x.size),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p90": float(np.percentile(x, 90)),
        "p95": float(np.percentile(x, 95)),
        "p99": float(np.percentile(x, 99)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def ecdf(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return x/y arrays for an empirical CDF."""
    x = np.sort(x)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def wasserstein_1d(u: np.ndarray, v: np.ndarray) -> float:
    """
    Compute the first Wasserstein distance (Earth Mover's Distance) between two
    1D empirical distributions without requiring SciPy.
    """
    u = np.sort(np.asarray(u, dtype=float))
    v = np.sort(np.asarray(v, dtype=float))

    if u.size == 0 or v.size == 0:
        return float("nan")

    all_values = np.concatenate([u, v])
    all_values.sort()

    if all_values.size < 2:
        return 0.0

    deltas = np.diff(all_values)
    u_cdf = np.searchsorted(u, all_values[:-1], side="right") / u.size
    v_cdf = np.searchsorted(v, all_values[:-1], side="right") / v.size
    return float(np.sum(np.abs(u_cdf - v_cdf) * deltas))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benign", required=True, help="Path to benign PCAP")
    parser.add_argument("--before", required=True, help="Path to original attack PCAP")
    parser.add_argument("--after", required=True, help="Path to perturbed attack PCAP")
    parser.add_argument("--out-png", required=True, help="Output PNG path")
    parser.add_argument("--out-json", required=True, help="Output JSON path")
    parser.add_argument(
        "--bins",
        type=int,
        default=50,
        help="Number of log-spaced bins for the histogram",
    )
    args = parser.parse_args()

    benign = positive_iats(args.benign)
    before = positive_iats(args.before)
    after = positive_iats(args.after)

    if benign.size == 0 or before.size == 0 or after.size == 0:
        raise SystemExit("One or more inputs contain no positive per-flow IAT values.")

    all_vals = np.concatenate([benign, before, after])
    xmin = np.min(all_vals)
    xmax = np.max(all_vals)

    # Keep bins on a log scale.
    bins = np.logspace(np.log10(xmin), np.log10(xmax), args.bins)

    # Summary statistics.
    benign_stats = describe(benign)
    before_stats = describe(before)
    after_stats = describe(after)

    wd_before = wasserstein_1d(before, benign)
    wd_after = wasserstein_1d(after, benign)
    wd_improvement = wd_before - wd_after

    # Plot.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: histogram with log-spaced bins.
    ax = axes[0]
    ax.hist(benign, bins=bins, density=True, histtype="step", linewidth=2, label="benign")
    ax.hist(before, bins=bins, density=True, histtype="step", linewidth=2, label="attack original")
    ax.hist(after, bins=bins, density=True, histtype="step", linewidth=2, label="attack mutated")
    ax.set_xscale("log")
    ax.set_xlabel("Per-flow inter-arrival time (seconds, log scale)")
    ax.set_ylabel("Density")
    ax.set_title("Histogram (log-spaced bins)")
    ax.legend()

    # Right: ECDF on log-x axis.
    ax = axes[1]
    x_b, y_b = ecdf(benign)
    x_o, y_o = ecdf(before)
    x_m, y_m = ecdf(after)
    ax.semilogx(x_b, y_b, linewidth=2, label="benign")
    ax.semilogx(x_o, y_o, linewidth=2, label="attack original")
    ax.semilogx(x_m, y_m, linewidth=2, label="attack mutated")
    ax.set_xlabel("Per-flow inter-arrival time (seconds, log scale)")
    ax.set_ylabel("ECDF")
    ax.set_title("Empirical CDF")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend()

    fig.suptitle(
        "TM-2 Rate: per-flow IAT comparison\n"
        f"Wasserstein distance to benign: before={wd_before:.6g}, "
        f"after={wd_after:.6g}, improvement={wd_improvement:.6g}"
    )
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=220)

    out = {
        "benign": benign_stats,
        "attack_original": before_stats,
        "attack_mutated": after_stats,
        "distance_to_benign": {
            "wasserstein_before": wd_before,
            "wasserstein_after": wd_after,
            "improvement": wd_improvement,
        },
        "notes": {
            "metric": "per-flow inter-arrival time (IAT)",
            "plots": [
                "histogram with log-spaced bins",
                "ECDF with log-scaled x-axis",
            ],
        },
    }

    Path(args.out_json).write_text(
        json.dumps(out, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()