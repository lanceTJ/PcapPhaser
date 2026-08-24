from __future__ import annotations

import bisect
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

from scapy.all import CookedLinux, Ether, IP, IPv6, Raw, TCP, UDP

from .io import PcapSinkBuffered
from .stream import _sniff_kind, stream_pcap_packets
from .utils import atomic_write_json, ensure_dir, log
from .verify import VerifyPacket, summarize_length_stage, summarize_rate_stage


DEFAULT_LENGTH_CAP_BYTES = 1460
DEFAULT_PAD_BYTE = "00"
DEFAULT_RATE_TAU_USEC = 1


@dataclass(frozen=True)
class PacketRecord:
    ordinal: int
    ts_sec: int
    ts_usec: int
    raw_bytes: bytes
    raw_len: int
    flow_id: str
    direction_id: str



def _mix_seed(selection_seed: int, in_pcap: str, stage_index: int, stage_type: str) -> int:
    digest = hashlib.blake2b(digest_size=8)
    digest.update(str(selection_seed).encode("utf-8"))
    digest.update(b"||")
    digest.update(str(stage_index).encode("utf-8"))
    digest.update(b"||")
    digest.update(stage_type.encode("utf-8"))
    digest.update(b"||")
    digest.update(in_pcap.encode("utf-8"))
    return int.from_bytes(digest.digest(), "little")



def _packet_timestamp_key(record: PacketRecord) -> tuple[int, int, int]:
    return record.ts_sec, record.ts_usec, record.ordinal



def _normalize_time(ts_sec: int, ts_usec: int) -> tuple[int, int]:
    ts_sec = int(ts_sec)
    ts_usec = int(ts_usec)
    if ts_usec >= 1_000_000:
        ts_sec += ts_usec // 1_000_000
        ts_usec %= 1_000_000
    elif ts_usec < 0:
        borrow = ((-ts_usec) + 999_999) // 1_000_000
        ts_sec -= borrow
        ts_usec += borrow * 1_000_000
    return ts_sec, ts_usec



def _flow_keys(pkt_bytes: bytes, linktype: int) -> tuple[str, str]:
    try:
        if linktype == 113:
            pkt = CookedLinux(pkt_bytes)
        elif linktype == 101:
            pkt = IPv6(pkt_bytes) if pkt_bytes and pkt_bytes[0] >> 4 == 6 else IP(pkt_bytes)
        else:
            pkt = Ether(pkt_bytes)
    except Exception:
        digest = hashlib.blake2b(pkt_bytes[:64], digest_size=8).hexdigest()
        return f"opaque:{digest}", f"opaque:{digest}"

    network = None
    if IP in pkt:
        ip_layer = pkt[IP]
        src_ip, dst_ip = ip_layer.src, ip_layer.dst
        proto = int(ip_layer.proto)
        network = "ip"
    elif IPv6 in pkt:
        ip_layer = pkt[IPv6]
        src_ip, dst_ip = ip_layer.src, ip_layer.dst
        proto = int(ip_layer.nh)
        network = "ipv6"
    else:
        digest = hashlib.blake2b(pkt_bytes[:64], digest_size=8).hexdigest()
        return f"l2:{digest}", f"l2:{digest}"

    if TCP in pkt:
        src_port, dst_port = int(pkt[TCP].sport), int(pkt[TCP].dport)
    elif UDP in pkt:
        src_port, dst_port = int(pkt[UDP].sport), int(pkt[UDP].dport)
    else:
        src_port, dst_port = 0, 0

    direction = f"{network}|{src_ip}|{dst_ip}|{src_port}|{dst_port}|{proto}"
    endpoints = sorted([(src_ip, src_port), (dst_ip, dst_port)])
    flow = f"{network}|{endpoints[0][0]}|{endpoints[0][1]}|{endpoints[1][0]}|{endpoints[1][1]}|{proto}"
    return flow, direction



def load_packet_records(pcap_path: str) -> tuple[list[PacketRecord], int]:
    """Load packets into memory for offline batch stages."""
    linktype = 1
    if _sniff_kind(pcap_path) == "pcap":
        with open(pcap_path, "rb") as handle:
            header = handle.read(24)
        byte_order = "little" if header[0:4] in (b"\xd4\xc3\xb2\xa1", b"\x4d\x3c\xb2\xa1") else "big"
        linktype = int.from_bytes(header[20:24], byte_order)

    records: list[PacketRecord] = []
    for ordinal, (pkt_bytes, ts_sec, ts_usec) in enumerate(stream_pcap_packets(pcap_path)):
        flow_id, direction_id = _flow_keys(pkt_bytes, linktype)
        records.append(
            PacketRecord(
                ordinal=ordinal,
                ts_sec=int(ts_sec),
                ts_usec=int(ts_usec),
                raw_bytes=bytes(pkt_bytes),
                raw_len=len(pkt_bytes),
                flow_id=flow_id,
                direction_id=direction_id,
            )
        )
    return records, linktype



def write_packet_records(path: str, records: Iterable[PacketRecord], linktype: int) -> None:
    sink = PcapSinkBuffered(path, linktype=linktype)
    try:
        for record in records:
            sink.write_raw(record.ts_sec, record.ts_usec, record.raw_bytes)
    finally:
        sink.close()



def _pad_byte_from_hex(value: str) -> bytes:
    parsed = bytes.fromhex(value)
    if len(parsed) != 1:
        raise ValueError("pad_byte must decode to exactly one byte")
    return parsed



def _extend_payload(pkt_bytes: bytes, grow_bytes: int, pad_byte: bytes) -> bytes:
    if grow_bytes <= 0:
        return pkt_bytes

    pkt = Ether(pkt_bytes)
    updated = pkt.copy()
    if Raw in updated:
        payload = bytes(updated[Raw].load) + (pad_byte * grow_bytes)
        updated[Raw].load = payload
    else:
        updated = updated / Raw(load=(pad_byte * grow_bytes))

    if IP in updated:
        if hasattr(updated[IP], "len"):
            del updated[IP].len
        if hasattr(updated[IP], "chksum"):
            del updated[IP].chksum
    if TCP in updated and hasattr(updated[TCP], "chksum"):
        del updated[TCP].chksum
    if UDP in updated and hasattr(updated[UDP], "len"):
        del updated[UDP].len
    if UDP in updated and hasattr(updated[UDP], "chksum"):
        del updated[UDP].chksum
    return bytes(updated)



def _run_tm_bridge(stage_config: dict[str, Any], tmp_dir: Path) -> dict[str, Any]:
    ensure_dir(tmp_dir)
    config_path = tmp_dir / "tm_stage_config.json"
    atomic_write_json(config_path, stage_config)

    bridge_path = Path(__file__).with_name("tm_bridge.py")
    cmd = [sys.executable, str(bridge_path), "--config", str(config_path)]
    log.info("Running TrafficManipulator bridge: %s", " ".join(cmd))
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "TrafficManipulator bridge failed with exit code "
            f"{completed.returncode}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )

    manifest_path = Path(stage_config["manifest_json"])
    return json.loads(manifest_path.read_text(encoding="utf-8"))



def _length_budget_value(entry: dict[str, Any], budget_basis: str) -> int:
    if budget_basis == "payload_len":
        return int(entry.get("crafted_payload_bytes", 0))
    if budget_basis == "packet_len":
        return int(entry.get("crafted_packet_bytes", 0))
    raise ValueError(f"Unsupported budget_basis: {budget_basis}")



def _build_direction_positions(records: list[PacketRecord]) -> dict[str, list[int]]:
    positions: dict[str, list[int]] = {}
    for record in records:
        positions.setdefault(record.direction_id, []).append(record.ordinal)
    return positions



def apply_length_manip_stage(
    in_pcap: str,
    out_pcap: str,
    stage: dict[str, Any],
    selection_seed: int,
    stage_index: int,
    tmp_dir: str,
) -> dict[str, Any]:
    params = dict(stage.get("params") or {})
    original_records, linktype = load_packet_records(in_pcap)
    stage_seed = _mix_seed(selection_seed, in_pcap, stage_index, "length_manip")
    work_dir = Path(tmp_dir) / f"stage_{stage_index:02d}_length_manip"
    ensure_dir(work_dir)

    tm_mutated_pcap = work_dir / "tm_mutated.pcap"
    tm_stats_path = work_dir / "tm_statistics.pkl"
    tm_manifest_path = work_dir / "tm_manifest.json"

    bridge_manifest = _run_tm_bridge(
        {
            "repo_path": params["tm"]["repo"],
            "mal_pcap": str(Path(in_pcap).resolve()),
            "mimic_set": params["tm"]["mimic_set"],
            "normalizer": params["tm"]["normalizer"],
            "init_pcap": params["tm"].get("init_pcap", params["tm"].get("init", "./data/empty.pcap")),
            "output_pcap": str(tm_mutated_pcap),
            "stats_pkl": str(tm_stats_path),
            "manifest_json": str(tm_manifest_path),
            "mode": "length",
            "stage_seed": stage_seed,
            "packet_limit": params.get("packet_limit"),
            "heuristic": bool(params.get("heuristic", False)),
            "particle_params": params.get("particle", {}),
            "pso_params": params.get("pso", {}),
            "manipulator_params": params.get("manipulator", {}),
        },
        work_dir,
    )

    budget_basis = str(params.get("budget_basis", "packet_len")).lower()
    spill_mode = str(params.get("spill_mode", "forward")).lower()
    cap_bytes = int(params.get("cap_bytes", DEFAULT_LENGTH_CAP_BYTES))
    pad_byte = _pad_byte_from_hex(str(params.get("pad_byte", DEFAULT_PAD_BYTE)))

    direction_positions = _build_direction_positions(original_records)
    increments = [0] * len(original_records)
    dropped_budget = 0

    for entry in bridge_manifest.get("entries", []):
        budget = _length_budget_value(entry, budget_basis)
        if budget <= 0:
            continue

        original_ordinal = int(entry["ordinal"])
        direction_id = original_records[original_ordinal].direction_id
        candidates = direction_positions.get(direction_id, [original_ordinal])
        current_index = bisect.bisect_left(candidates, original_ordinal)
        target_pointer = max(0, current_index - 1)

        while budget > 0 and target_pointer < len(candidates):
            target_ordinal = candidates[target_pointer]
            remaining_cap = max(0, cap_bytes - increments[target_ordinal])
            if remaining_cap > 0:
                delta = min(budget, remaining_cap)
                increments[target_ordinal] += delta
                budget -= delta
            if budget <= 0:
                break
            if spill_mode == "forward":
                target_pointer += 1
            elif spill_mode == "drop":
                dropped_budget += budget
                budget = 0
            else:
                raise ValueError(f"Unsupported spill_mode: {spill_mode}")

        if budget > 0:
            dropped_budget += budget

    mutated_records: list[PacketRecord] = []
    for record in original_records:
        grow_bytes = increments[record.ordinal]
        new_bytes = _extend_payload(record.raw_bytes, grow_bytes, pad_byte) if grow_bytes > 0 else record.raw_bytes
        mutated_records.append(
            replace(record, raw_bytes=new_bytes, raw_len=len(new_bytes))
        )

    write_packet_records(out_pcap, mutated_records, linktype)

    verify = summarize_length_stage(
        [VerifyPacket(r.ordinal, r.ts_sec, r.ts_usec, r.raw_len) for r in original_records],
        [VerifyPacket(r.ordinal, r.ts_sec, r.ts_usec, r.raw_len) for r in mutated_records],
    )
    verify["metrics"]["dropped_budget_bytes"] = dropped_budget
    verify["metrics"]["cap_bytes"] = cap_bytes
    verify["metrics"]["budget_basis"] = budget_basis
    verify["metrics"]["spill_mode"] = spill_mode

    return {
        "type": "length_manip",
        "stage_seed": stage_seed,
        "total_in": len(original_records),
        "total_out": len(mutated_records),
        "tm": {
            "mutated_pcap": str(tm_mutated_pcap),
            "statistics_pkl": str(tm_stats_path),
            "manifest_json": str(tm_manifest_path),
        },
        "verify": verify,
    }



def _weighted_flow_subset(
    flow_counts: dict[str, int],
    pct: float,
    mode: str,
    seed: int,
) -> set[str]:
    import random

    rng = random.Random(seed)
    flow_items = list(flow_counts.items())
    if not flow_items or pct <= 0:
        return set()
    if pct >= 1:
        return {flow_id for flow_id, _ in flow_items}

    if mode == "flow_uniform":
        flow_ids = [flow_id for flow_id, _ in flow_items]
        rng.shuffle(flow_ids)
        target = max(1, round(len(flow_ids) * pct))
        return set(flow_ids[:target])

    if mode == "packet_weighted":
        total_packets = sum(count for _, count in flow_items)
        target_packets = max(1, round(total_packets * pct))
        rng.shuffle(flow_items)
        chosen: set[str] = set()
        covered = 0
        for flow_id, count in flow_items:
            chosen.add(flow_id)
            covered += count
            if covered >= target_packets:
                break
        return chosen

    raise ValueError(f"Unsupported select mode: {mode}")



def apply_rate_manip_stage(
    in_pcap: str,
    out_pcap: str,
    stage: dict[str, Any],
    selection_seed: int,
    stage_index: int,
    tmp_dir: str,
) -> dict[str, Any]:
    params = dict(stage.get("params") or {})
    original_records, linktype = load_packet_records(in_pcap)
    stage_seed = _mix_seed(selection_seed, in_pcap, stage_index, "rate_manip")
    work_dir = Path(tmp_dir) / f"stage_{stage_index:02d}_rate_manip"
    ensure_dir(work_dir)

    flow_counts: dict[str, int] = {}
    for record in original_records:
        flow_counts[record.flow_id] = flow_counts.get(record.flow_id, 0) + 1

    select_mode = str(params.get("select", {}).get("mode", "flow_uniform")).lower()
    selected_flows = _weighted_flow_subset(flow_counts, float(stage.get("pct", 0.0)), select_mode, stage_seed)
    selected_flow_records = [record for record in original_records if record.flow_id in selected_flows]
    configured_limit = params.get("packet_limit")
    if configured_limit is None:
        selected_records = selected_flow_records
    else:
        packet_limit = int(configured_limit)
        if packet_limit <= 0:
            raise ValueError("packet_limit must be positive")
        selected_records = selected_flow_records[:packet_limit]
    selected_ordinals = {record.ordinal for record in selected_records}
    passthrough_records = [record for record in original_records if record.ordinal not in selected_ordinals]

    split_selected_pcap = work_dir / "selected_input.pcap"
    tm_mutated_pcap = work_dir / "selected_tm_mutated.pcap"
    tm_stats_path = work_dir / "tm_statistics.pkl"
    tm_manifest_path = work_dir / "tm_manifest.json"

    write_packet_records(str(split_selected_pcap), selected_records, linktype)

    rate_manipulator_params = dict(params.get("manipulator", {}))
    rate_manipulator_params["max_cft_pkt"] = max(1, int(rate_manipulator_params.get("max_cft_pkt", 1)))
    rate_manipulator_params["max_crafted_pkt_prob"] = 0.0

    if selected_records:
        _run_tm_bridge(
            {
                "repo_path": params["tm"]["repo"],
                "mal_pcap": str(split_selected_pcap),
                "mimic_set": params["tm"]["mimic_set"],
                "normalizer": params["tm"]["normalizer"],
                "init_pcap": params["tm"].get("init_pcap", params["tm"].get("init", "./data/empty.pcap")),
                "output_pcap": str(tm_mutated_pcap),
                "stats_pkl": str(tm_stats_path),
                "manifest_json": str(tm_manifest_path),
                "mode": "rate",
                "stage_seed": stage_seed,
                "heuristic": bool(params.get("heuristic", False)),
                "particle_params": params.get("particle", {}),
                "pso_params": params.get("pso", {}),
                "manipulator_params": rate_manipulator_params,
            },
            work_dir,
        )
        mutated_selected_records, _ = load_packet_records(str(tm_mutated_pcap))
        if len(mutated_selected_records) != len(selected_records):
            raise RuntimeError(
                "TrafficManipulator rate stage changed packet count. "
                "Expected a time-only mutation with crafted-packet probability zero."
            )
    else:
        mutated_selected_records = []
        write_packet_records(str(tm_mutated_pcap), [], linktype)
        atomic_write_json(tm_manifest_path, {"entries": [], "mode": "rate"})

    mutated_by_ordinal: dict[int, PacketRecord] = {}
    for original_record, mutated_record in zip(selected_records, mutated_selected_records):
        mutated_by_ordinal[original_record.ordinal] = replace(
            original_record,
            ts_sec=mutated_record.ts_sec,
            ts_usec=mutated_record.ts_usec,
        )
    for record in passthrough_records:
        mutated_by_ordinal[record.ordinal] = record

    merged_records = [mutated_by_ordinal[index] for index in range(len(original_records))]
    if bool(params.get("merge", {}).get("sort_by_ts", True)):
        merged_records = sorted(merged_records, key=_packet_timestamp_key)

    write_packet_records(out_pcap, merged_records, linktype)

    verify = summarize_rate_stage(
        [VerifyPacket(r.ordinal, r.ts_sec, r.ts_usec, r.raw_len) for r in original_records],
        {
            ordinal: VerifyPacket(record.ordinal, record.ts_sec, record.ts_usec, record.raw_len)
            for ordinal, record in mutated_by_ordinal.items()
        },
        tau_usec=int(params.get("verify", {}).get("tau_usec", DEFAULT_RATE_TAU_USEC)),
    )
    verify["metrics"]["selected_flow_count"] = len(selected_flows)
    verify["metrics"]["selected_packet_count"] = len(selected_records)
    verify["metrics"]["selected_flow_packet_count"] = len(selected_flow_records)
    verify["metrics"]["selection_mode"] = select_mode

    return {
        "type": "rate_manip",
        "stage_seed": stage_seed,
        "total_in": len(original_records),
        "total_out": len(merged_records),
        "tm": {
            "selected_input_pcap": str(split_selected_pcap),
            "mutated_pcap": str(tm_mutated_pcap),
            "statistics_pkl": str(tm_stats_path),
            "manifest_json": str(tm_manifest_path),
        },
        "verify": verify,
    }
