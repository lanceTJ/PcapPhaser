from __future__ import annotations

import hashlib
import random
import shutil
import struct
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
from scapy.all import Ether

from .io import PcapSinkBuffered
from .manip_stages import apply_length_manip_stage, apply_rate_manip_stage
from .perturbations import PERTURBATIONS
from .stream import _sniff_kind, stream_pcap_packets, stream_raw_pcap_records
from .utils import log


STREAM_STAGE_TYPES = {"loss", "retransmit", "retrans", "seq_offset", "reorder", "jitter"}
BATCH_STAGE_TYPES = {"length_manip", "rate_manip"}
NEED_PARSE_OPS = {"seq_offset"}



def _plan_needs_parse(plan: list[dict]) -> bool:
    return any(str(step.get("type", "")).lower() in NEED_PARSE_OPS for step in plan)



def _extract_ts_from_raw_records(buf: list[bytes]) -> tuple[np.ndarray, np.ndarray, Callable[[bytes, int, int], bytes]]:
    if not buf:
        raise ValueError("Empty packet buffer")

    first_hdr = buf[0][0:16]
    try:
        struct.unpack("<IIII", first_hdr)
        fmt = "<IIII"
    except struct.error:
        struct.unpack(">IIII", first_hdr)
        fmt = ">IIII"

    ts_sec_list: list[int] = []
    ts_usec_list: list[int] = []
    for rec in buf:
        ts_sec, ts_usec, _, _ = struct.unpack(fmt, rec[0:16])
        ts_sec_list.append(ts_sec)
        ts_usec_list.append(ts_usec)

    def patch_ts(raw_record: bytes, new_sec: int, new_usec: int) -> bytes:
        incl_len = len(raw_record) - 16
        header = struct.pack(fmt, int(new_sec), int(new_usec), incl_len, incl_len)
        return header + raw_record[16:]

    return (
        np.array(ts_sec_list, dtype=np.int64),
        np.array(ts_usec_list, dtype=np.int64),
        patch_ts,
    )



def _select_indices(plan: list[dict], n: int, rng: np.random.Generator, stats: dict) -> np.ndarray:
    idx = np.arange(n, dtype=np.int64)

    for step in plan:
        stage_type = str(step.get("type", "")).lower()

        if stage_type == "loss":
            pct = float(step.get("pct", 0.0))
            keep_mask = rng.random(idx.size) >= pct
            stats["loss_drop"] += int((~keep_mask).sum())
            idx = idx[keep_mask]

        elif stage_type in {"retransmit", "retrans"}:
            pct = float(step.get("pct", 0.0))
            duplicate_mask = rng.random(idx.size) < pct
            stats["retransmit_dup"] += int(duplicate_mask.sum())
            if duplicate_mask.any():
                idx = np.concatenate([idx, idx[duplicate_mask]])

        elif stage_type in {"reorder", "jitter"}:
            pct = float(step.get("pct", 0.0))
            params = dict(step.get("params") or {})
            max_segment = int(params.get("m", 5))
            if pct <= 0 or idx.size <= 2:
                continue

            used = np.zeros(idx.size, dtype=bool)
            reorder_count = 0
            for start in rng.permutation(np.arange(idx.size)):
                if used[start] or rng.random() > pct:
                    continue
                end = min(start + int(rng.integers(2, max_segment + 1)), idx.size)
                if np.any(used[start:end]):
                    continue
                used[start:end] = True
                segment_positions = np.arange(start, end)
                shuffled_positions = segment_positions.copy()
                rng.shuffle(shuffled_positions)
                idx[segment_positions] = idx[shuffled_positions]
                reorder_count += end - start
            stats["reorder_packets"] += int(reorder_count)

    return idx



def _process_chunk(buf, sink, plan, py_rng, stats):
    packet_count = len(buf)
    if packet_count == 0:
        return 0, 0

    np_rng = np.random.default_rng(py_rng.getrandbits(64))
    out_indices = _select_indices(plan, packet_count, np_rng, stats)
    needs_parse = _plan_needs_parse(plan)

    first = buf[0]
    is_raw = isinstance(first, (bytes, bytearray))

    if is_raw:
        ts_sec, ts_usec, patch_ts = _extract_ts_from_raw_records(buf)
    else:
        ts_sec = np.fromiter((item[0] for item in buf), dtype=np.int64, count=packet_count)
        ts_usec = np.fromiter((item[1] for item in buf), dtype=np.int64, count=packet_count)
        patch_ts = None

    if len(out_indices) == 0:
        return packet_count, 0

    ts_sec_new = ts_sec[out_indices].copy()
    ts_usec_new = ts_usec[out_indices].copy()
    for index in range(1, len(out_indices)):
        if (ts_sec_new[index], ts_usec_new[index]) <= (ts_sec_new[index - 1], ts_usec_new[index - 1]):
            ts_sec_new[index] = ts_sec_new[index - 1]
            ts_usec_new[index] = ts_usec_new[index - 1] + 1
            if ts_usec_new[index] >= 1_000_000:
                ts_sec_new[index] += ts_usec_new[index] // 1_000_000
                ts_usec_new[index] %= 1_000_000

    out_count = 0
    mod_rng = random.Random(py_rng.getrandbits(64))

    for output_pos, source_index in enumerate(out_indices):
        ts_s = int(ts_sec_new[output_pos])
        ts_u = int(ts_usec_new[output_pos])

        if is_raw:
            record = buf[source_index]
            pkt_bytes = record[16:]
        else:
            pkt_bytes = buf[source_index][2]

        if needs_parse:
            try:
                pkt = Ether(pkt_bytes)
            except Exception as exc:
                log.warning("Failed to parse packet for content mutation: %s", exc)
                if is_raw:
                    sink.write_raw_record(patch_ts(record, ts_s, ts_u))
                else:
                    sink.write_raw(ts_s, ts_u, pkt_bytes)
                out_count += 1
                continue

            emitted = None
            skip_packet = False
            for step in plan:
                stage_type = str(step.get("type", "")).lower()
                if stage_type not in NEED_PARSE_OPS:
                    continue
                if mod_rng.random() >= float(step.get("pct", 0.0)):
                    continue
                mutated = PERTURBATIONS[stage_type](pkt, **dict(step.get("params") or {}))
                if mutated is None:
                    skip_packet = True
                    break
                if isinstance(mutated, list):
                    emitted = mutated
                    break
                pkt = mutated

            if skip_packet:
                continue

            if emitted is not None:
                for item in emitted:
                    sink.write_raw(ts_s, ts_u, bytes(item))
                    out_count += 1
                continue

            pkt_bytes = bytes(pkt)

        if is_raw:
            sink.write_raw_record(patch_ts(record[:16] + pkt_bytes, ts_s, ts_u))
        else:
            sink.write_raw(ts_s, ts_u, pkt_bytes)
        out_count += 1

    return packet_count, out_count



def _mix_seed(selection_seed: int, in_pcap: str) -> int:
    digest = hashlib.blake2b(digest_size=8)
    digest.update(str(selection_seed).encode("utf-8"))
    digest.update(b"||")
    digest.update(in_pcap.encode("utf-8"))
    return int.from_bytes(digest.digest(), "little")



def apply_perturbations_stream(
    in_pcap: str,
    out_pcap: str,
    perturb_plan: list[dict],
    selection_seed: int = 0,
    chunk_size: int = 10000,
    show_progress: bool = False,
    progress_every: int = 200_000,
):
    py_seed = _mix_seed(selection_seed, in_pcap)
    rng = random.Random(py_seed)

    kind = _sniff_kind(in_pcap)
    linktype = 1
    if kind == "pcap":
        with open(in_pcap, "rb") as handle:
            header = handle.read(24)
        byte_order = "little" if header[0:4] in (b"\xd4\xc3\xb2\xa1", b"\x4d\x3c\xb2\xa1") else "big"
        linktype = int.from_bytes(header[20:24], byte_order)

    sink = PcapSinkBuffered(out_pcap, linktype=linktype)
    total_in = 0
    total_out = 0
    stats = defaultdict(int)

    needs_parse = _plan_needs_parse(perturb_plan)
    stream_func = stream_pcap_packets if needs_parse or kind != "pcap" else stream_raw_pcap_records
    log.info("Using %s for %s", stream_func.__name__, in_pcap)

    buffer = []
    try:
        for item in stream_func(in_pcap):
            if stream_func is stream_pcap_packets:
                pkt_bytes, ts_sec, ts_usec = item
                buffer.append((ts_sec, ts_usec, pkt_bytes))
            else:
                buffer.append(item)

            if len(buffer) >= chunk_size:
                chunk_in, chunk_out = _process_chunk(buffer, sink, perturb_plan, rng, stats)
                total_in += chunk_in
                total_out += chunk_out
                if show_progress and total_in // progress_every != (total_in - chunk_in) // progress_every:
                    log.info("[progress] %s in=%d out=%d", in_pcap, total_in, total_out)
                buffer.clear()

        if buffer:
            chunk_in, chunk_out = _process_chunk(buffer, sink, perturb_plan, rng, stats)
            total_in += chunk_in
            total_out += chunk_out
            if show_progress:
                log.info("[progress] %s in=%d out=%d", in_pcap, total_in, total_out)
    finally:
        sink.close()

    return {"total_in": total_in, "total_out": total_out, "stats": dict(stats)}



def _copy_if_needed(src: str, dst: str) -> None:
    if Path(src).resolve() == Path(dst).resolve():
        return
    shutil.copyfile(src, dst)



def apply_perturbations_plan(
    in_pcap: str,
    out_pcap: str,
    perturb_plan: list[dict],
    selection_seed: int = 0,
    chunk_size: int = 10000,
    show_progress: bool = False,
    progress_every: int = 200_000,
    tmp_dir: str | None = None,
):
    """Apply a mixed streaming/batch perturbation plan to a single capture file."""
    if not perturb_plan:
        _copy_if_needed(in_pcap, out_pcap)
        return {"total_in": 0, "total_out": 0, "stats": {}, "stages": []}

    with tempfile.TemporaryDirectory(prefix="pcapperturbator_plan_", dir=tmp_dir) as temp_root:
        temp_root_path = Path(temp_root)
        current_input = str(Path(in_pcap).resolve())
        stages_meta: list[dict] = []
        stream_group: list[dict] = []
        stream_group_start = 0

        def flush_stream_group(final_stage_index: int, destination: str | None = None) -> None:
            nonlocal current_input, stream_group, stream_group_start
            if not stream_group:
                return
            target_path = destination or str(temp_root_path / f"stage_{final_stage_index:02d}_stream_output.pcap")
            result = apply_perturbations_stream(
                in_pcap=current_input,
                out_pcap=target_path,
                perturb_plan=stream_group,
                selection_seed=selection_seed + stream_group_start,
                chunk_size=chunk_size,
                show_progress=show_progress,
                progress_every=progress_every,
            )
            stages_meta.append(
                {
                    "type": "stream_group",
                    "start_index": stream_group_start,
                    "end_index": final_stage_index - 1,
                    "plan": stream_group,
                    "result": result,
                }
            )
            current_input = target_path
            stream_group = []

        for stage_index, stage in enumerate(perturb_plan):
            stage_type = str(stage.get("type", "")).lower()
            if stage_type in STREAM_STAGE_TYPES:
                if not stream_group:
                    stream_group_start = stage_index
                stream_group.append(stage)
                continue

            flush_stream_group(stage_index)

            stage_output = str(temp_root_path / f"stage_{stage_index:02d}_{stage_type}.pcap")
            if stage_type == "length_manip":
                stage_meta = apply_length_manip_stage(
                    in_pcap=current_input,
                    out_pcap=stage_output,
                    stage=stage,
                    selection_seed=selection_seed,
                    stage_index=stage_index,
                    tmp_dir=str(temp_root_path),
                )
            elif stage_type == "rate_manip":
                stage_meta = apply_rate_manip_stage(
                    in_pcap=current_input,
                    out_pcap=stage_output,
                    stage=stage,
                    selection_seed=selection_seed,
                    stage_index=stage_index,
                    tmp_dir=str(temp_root_path),
                )
            else:
                raise ValueError(f"Unsupported stage type: {stage_type}")

            stages_meta.append(stage_meta)
            current_input = stage_output

        flush_stream_group(len(perturb_plan), out_pcap if current_input == str(Path(in_pcap).resolve()) else None)

        if Path(current_input).resolve() != Path(out_pcap).resolve():
            shutil.copyfile(current_input, out_pcap)

    aggregate = {"total_in": 0, "total_out": 0, "stats": {}, "stages": stages_meta}
    for stage_meta in stages_meta:
        result = stage_meta.get("result") or {}
        aggregate["total_in"] += int(result.get("total_in", 0))
        aggregate["total_out"] += int(result.get("total_out", 0))
    return aggregate
