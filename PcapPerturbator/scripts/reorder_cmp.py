#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from _pcap_utils import pkt_lengths_and_ts, basic_summary

def main():
    ap = argparse.ArgumentParser(description="Compare reorder effect between two PCAPs.")
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    b_len, b_ts, b_hash = pkt_lengths_and_ts(args.before)
    a_len, a_ts, a_hash = pkt_lengths_and_ts(args.after)

    same_position = sum(1 for x, y in zip(b_hash, a_hash) if x == y)
    position_changed = min(len(b_hash), len(a_hash)) - same_position

    def mean_iat(ts):
        return float(sum(ts[i] - ts[i-1] for i in range(1, len(ts))) / max(1, len(ts)-1)) if len(ts) >= 2 else 0.0

    result = {
        "mode": "reorder",
        "before": basic_summary(args.before),
        "after": basic_summary(args.after),
        "interpretable_metrics": {
            "packet_count_same": len(b_hash) == len(a_hash),
            "total_packet_bytes_delta": int(sum(a_len) - sum(b_len)),
            "same_position_packets": int(same_position),
            "position_changed_packets": int(position_changed),
            "position_changed_ratio": float(position_changed / max(1, min(len(b_hash), len(a_hash)))),
            "iat_mean_before_sec": mean_iat(b_ts),
            "iat_mean_after_sec": mean_iat(a_ts),
        },
    }

    Path(args.out_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
