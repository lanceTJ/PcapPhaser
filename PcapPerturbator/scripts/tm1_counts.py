#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from _pcap_utils import pkt_lengths_and_ts, basic_summary, multiset_difference

def main():
    ap = argparse.ArgumentParser(description="Compare before/after PCAP for loss or retransmission.")
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--mode", choices=["loss", "retrans"], required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    before_lengths, before_ts, before_hashes = pkt_lengths_and_ts(args.before)
    after_lengths, after_ts, after_hashes = pkt_lengths_and_ts(args.after)

    s_before = basic_summary(args.before)
    s_after = basic_summary(args.after)
    diff = multiset_difference(before_hashes, after_hashes)

    result = {
        "mode": args.mode,
        "before": s_before,
        "after": s_after,
        "delta": {
            "packet_count": int(s_after["packet_count"] - s_before["packet_count"]),
            "total_packet_bytes": int(s_after["total_packet_bytes"] - s_before["total_packet_bytes"]),
            "duration_sec": float(s_after["duration_sec"] - s_before["duration_sec"]),
        },
        "multiset_difference": diff,
    }

    if args.mode == "loss":
        removed = max(0, s_before["packet_count"] - s_after["packet_count"])
        result["interpretable_metrics"] = {
            "lost_packets": removed,
            "loss_ratio": float(removed / max(1, s_before["packet_count"])),
            "traffic_size_delta_bytes": int(s_after["total_packet_bytes"] - s_before["total_packet_bytes"]),
        }
    else:
        duplicated = max(0, s_after["packet_count"] - s_before["packet_count"])
        result["interpretable_metrics"] = {
            "duplicated_packets": duplicated,
            "duplication_ratio": float(duplicated / max(1, s_before["packet_count"])),
            "traffic_size_delta_bytes": int(s_after["total_packet_bytes"] - s_before["total_packet_bytes"]),
        }

    Path(args.out_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
