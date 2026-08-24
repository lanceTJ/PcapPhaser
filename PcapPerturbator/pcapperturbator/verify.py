from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class VerifyPacket:
    ordinal: int
    ts_sec: int
    ts_usec: int
    raw_len: int



def _timestamp_usec(packet: VerifyPacket) -> int:
    return int(packet.ts_sec) * 1_000_000 + int(packet.ts_usec)



def summarize_length_stage(original: Sequence[VerifyPacket], mutated: Sequence[VerifyPacket]) -> dict:
    changed_packets = 0
    transferred_bytes = 0
    for left, right in zip(original, mutated):
        if left.raw_len != right.raw_len:
            changed_packets += 1
            transferred_bytes += max(0, right.raw_len - left.raw_len)

    total_bytes = sum(packet.raw_len for packet in original)
    timestamps_unchanged = all(
        (left.ts_sec, left.ts_usec) == (right.ts_sec, right.ts_usec)
        for left, right in zip(original, mutated)
    )

    return {
        "invariants": {
            "packet_count_same": len(original) == len(mutated),
            "timestamps_unchanged": timestamps_unchanged,
        },
        "metrics": {
            "transferred_bytes": transferred_bytes,
            "changed_packets": changed_packets,
            "total_bytes_original": total_bytes,
            "S_len": (transferred_bytes / total_bytes) if total_bytes else 0.0,
            "P_len": (changed_packets / len(original)) if original else 0.0,
        },
    }



def summarize_rate_stage(original: Sequence[VerifyPacket], mutated_by_ordinal: dict[int, VerifyPacket], tau_usec: int = 1) -> dict:
    changed_packets = 0
    total_ratio = 0.0
    ratio_count = 0
    lengths_unchanged = True

    ordered_original = list(original)
    ordered_mutated = [mutated_by_ordinal[p.ordinal] for p in ordered_original]

    for left, right in zip(ordered_original, ordered_mutated):
        lengths_unchanged = lengths_unchanged and (left.raw_len == right.raw_len)
        delta = abs(_timestamp_usec(left) - _timestamp_usec(right))
        if delta > tau_usec:
            changed_packets += 1

    for index in range(1, len(ordered_original)):
        old_gap = max(1, _timestamp_usec(ordered_original[index]) - _timestamp_usec(ordered_original[index - 1]))
        new_gap = _timestamp_usec(ordered_mutated[index]) - _timestamp_usec(ordered_mutated[index - 1])
        total_ratio += abs(new_gap - old_gap) / old_gap
        ratio_count += 1

    return {
        "invariants": {
            "packet_count_same": len(ordered_original) == len(ordered_mutated),
            "lengths_unchanged": lengths_unchanged,
        },
        "metrics": {
            "changed_packets": changed_packets,
            "tau_usec": tau_usec,
            "S_rate": (total_ratio / ratio_count) if ratio_count else 0.0,
            "P_rate": (changed_packets / len(ordered_original)) if ordered_original else 0.0,
        },
    }
