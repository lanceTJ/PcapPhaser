#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import matplotlib.pyplot as plt
from _pcap_utils import pkt_lengths_and_ts

def main():
    ap = argparse.ArgumentParser(description="Plot packet-length distributions for benign, original attack, and mutated attack.")
    ap.add_argument("--benign", required=True)
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--out-png", required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    benign_lengths, _, _ = pkt_lengths_and_ts(args.benign)
    before_lengths, _, _ = pkt_lengths_and_ts(args.before)
    after_lengths, _, _ = pkt_lengths_and_ts(args.after)

    changed_packets = sum(1 for x, y in zip(before_lengths, after_lengths) if x != y)

    result = {
        "mode": "length",
        "before_packet_count": len(before_lengths),
        "after_packet_count": len(after_lengths),
        "changed_packets_by_length": int(changed_packets),
        "before_total_packet_bytes": int(sum(before_lengths)),
        "after_total_packet_bytes": int(sum(after_lengths)),
        "delta_packet_bytes": int(sum(after_lengths) - sum(before_lengths)),
    }

    plt.figure(figsize=(9, 6))
    plt.hist(benign_lengths, bins=100, alpha=0.35, density=True, label="benign")
    plt.hist(before_lengths, bins=100, alpha=0.35, density=True, label="attack original")
    plt.hist(after_lengths, bins=100, alpha=0.35, density=True, label="attack mutated")
    plt.xlabel("Packet length (bytes)")
    plt.ylabel("Density")
    plt.title("TM-2 Length: packet-length distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=220)

    Path(args.out_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved plot: {args.out_png}")

if __name__ == "__main__":
    main()
