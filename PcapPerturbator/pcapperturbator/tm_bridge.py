from __future__ import annotations

import argparse
import importlib
import json
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scapy.all import Raw, wrpcap


def _load_config(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))



def _patch_runtime_compatibility() -> None:
    if not hasattr(time, "clock"):
        time.clock = time.perf_counter  # type: ignore[attr-defined]



def _reset_tm_globals(manipulator_module) -> None:
    for attr in [
        "STA_X_list",
        "STA_feature_list",
        "STA_pktList_list",
        "STA_gbl_dis_list",
        "STA_avg_dis_list",
        "STA_all_feature_list",
    ]:
        if hasattr(manipulator_module, attr):
            getattr(manipulator_module, attr).clear()



def _payload_len(pkt) -> int:
    return len(bytes(pkt[Raw].load)) if Raw in pkt else 0



def _flatten_packet_lists(pkt_list_list) -> list[Any]:
    flat = []
    for group in pkt_list_list:
        flat.extend(group)
    return flat



def _safe_int_count(value: Any) -> int:
    try:
        x = float(value)
    except Exception:
        return 0
    if not np.isfinite(x):
        return 0
    # Be conservative: use floor instead of round
    return max(0, int(np.floor(x + 1e-9)))


def _build_manifest(x_list, pkt_list_list, mode: str) -> dict[str, Any]:
    entries = []
    global_ordinal = 0

    for group_index, (x_state, group_packets) in enumerate(zip(x_list, pkt_list_list)):
        position = 0
        group_size = len(x_state.mal)

        for local_index in range(group_size):
            remaining_originals = group_size - local_index
            remaining_packets = len(group_packets) - position

            if remaining_packets < remaining_originals:
                raise RuntimeError(
                    f"TrafficManipulator layout mismatch in group {group_index}: "
                    f"remaining_packets={remaining_packets}, remaining_originals={remaining_originals}"
                )

            raw_count = x_state.mal[local_index][1]
            wanted_crafted = _safe_int_count(raw_count)

            # Reserve at least one packet for each remaining original packet.
            max_available_crafted = remaining_packets - remaining_originals
            crafted_count = min(wanted_crafted, max_available_crafted)

            crafted_packets = group_packets[position : position + crafted_count]
            position += crafted_count

            if position >= len(group_packets):
                raise RuntimeError(
                    f"TrafficManipulator layout exhausted in group {group_index}, "
                    f"local_index={local_index}"
                )

            original_packet = group_packets[position]
            position += 1

            entries.append(
                {
                    "ordinal": global_ordinal,
                    "local_index": local_index,
                    "crafted_packets": crafted_count,
                    "crafted_packet_bytes": int(sum(len(bytes(pkt)) for pkt in crafted_packets)),
                    "crafted_payload_bytes": int(sum(_payload_len(pkt) for pkt in crafted_packets)),
                    "original_packet_bytes": int(len(bytes(original_packet))),
                    "raw_crafted_value": float(raw_count),
                }
            )
            global_ordinal += 1

        # Ignore any trailing packets instead of crashing.
        # They still exist in tm_mutated.pcap, but are not projected into the length-only manifest.
        # This is acceptable for the current projection pipeline.

    return {"mode": mode, "entries": entries}



def run_bridge(config: dict[str, Any]) -> dict[str, Any]:
    _patch_runtime_compatibility()
    repo_path = Path(config["repo_path"]).resolve()
    if not repo_path.exists():
        raise FileNotFoundError(f"TrafficManipulator repository not found: {repo_path}")

    sys.path.insert(0, str(repo_path))
    try:
        manipulator_module = importlib.import_module("manipulator")
        _reset_tm_globals(manipulator_module)
        Manipulator = manipulator_module.Manipulator

        stage_seed = int(config.get("stage_seed", 0))
        random.seed(stage_seed)
        np.random.seed(stage_seed & 0xFFFFFFFF)

        manipulator = Manipulator(
            config["mal_pcap"],
            config["mimic_set"],
            config["normalizer"],
            config.get("init_pcap", "./data/empty.pcap"),
        )

        manipulator.change_particle_params(**(config.get("particle_params") or {}))
        manipulator.change_pso_params(**(config.get("pso_params") or {}))
        manipulator.change_manipulator_params(**(config.get("manipulator_params") or {}))
        configured_limit = config.get("packet_limit")
        available_packets = len(manipulator.pktList)
        packet_limit = (
            available_packets
            if configured_limit is None
            else min(int(configured_limit), available_packets)
        )
        if packet_limit <= 0:
            raise ValueError("packet_limit must be positive")
        manipulator.grp_size = min(int(manipulator.grp_size), packet_limit)
        manipulator.process(
            config["stats_pkl"],
            limit=packet_limit,
            heuristic=bool(config.get("heuristic", False)),
        )

        with open(config["stats_pkl"], "rb") as handle:
            x_list = pickle.load(handle)
            _ = pickle.load(handle)
            pkt_list_list = pickle.load(handle)
            _ = pickle.load(handle)
            _ = pickle.load(handle)
            _ = pickle.load(handle)

        flat_packets = _flatten_packet_lists(pkt_list_list)
        wrpcap(config["output_pcap"], flat_packets)

        manifest = _build_manifest(x_list, pkt_list_list, str(config.get("mode", "unknown")))
        manifest["processed_packet_count"] = len(manifest["entries"])
        Path(config["manifest_json"]).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest
    finally:
        if str(repo_path) in sys.path:
            sys.path.remove(str(repo_path))



def main() -> None:
    parser = argparse.ArgumentParser(description="TrafficManipulator bridge for pcapperturbator")
    parser.add_argument("--config", required=True, help="Path to the bridge JSON config")
    args = parser.parse_args()
    run_bridge(_load_config(args.config))


if __name__ == "__main__":
    main()
