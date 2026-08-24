#!/usr/bin/env python3
from __future__ import annotations
from typing import Iterator, Tuple
import hashlib
from scapy.all import RawPcapReader, PcapNgReader, PcapReader, IP, TCP, UDP

_PCAP_MAGIC = {b"\xd4\xc3\xb2\xa1", b"\xa1\xb2\xc3\xd4", b"\x4d\x3c\xb2\xa1", b"\xa1\xb2\x3c\x4d"}
_PCAPNG_MAGIC = b"\x0a\x0d\x0d\x0a"

def sniff_kind(path: str) -> str:
    with open(path, "rb") as f:
        head = f.read(4)
    if head in _PCAP_MAGIC:
        return "pcap"
    if head == _PCAPNG_MAGIC:
        return "pcapng"
    raise ValueError(f"Unknown capture format: {path}")

def stream_packets(path: str) -> Iterator[Tuple[bytes, float]]:
    kind = sniff_kind(path)
    if kind == "pcap":
        reader = RawPcapReader(path)
        try:
            for pkt_bytes, meta in reader:
                ts = float(int(meta.sec)) + float(int(meta.usec)) / 1_000_000.0
                yield bytes(pkt_bytes), ts
        finally:
            reader.close()
    else:
        reader = PcapNgReader(path)
        try:
            for pkt in reader:
                ts = float(getattr(pkt, "time", 0.0))
                yield bytes(pkt.original), ts
        finally:
            reader.close()

def packet_hash(pkt_bytes: bytes) -> str:
    return hashlib.blake2b(pkt_bytes, digest_size=16).hexdigest()

def pkt_lengths_and_ts(path: str):
    lengths, ts_list, hashes = [], [], []
    for pkt_bytes, ts in stream_packets(path):
        lengths.append(len(pkt_bytes))
        ts_list.append(ts)
        hashes.append(packet_hash(pkt_bytes))
    return lengths, ts_list, hashes

def flow_key_from_packet(pkt):
    if IP not in pkt:
        return None
    src = pkt[IP].src
    dst = pkt[IP].dst
    proto = int(pkt[IP].proto)
    sport = dport = 0
    if TCP in pkt:
        sport = int(pkt[TCP].sport)
        dport = int(pkt[TCP].dport)
    elif UDP in pkt:
        sport = int(pkt[UDP].sport)
        dport = int(pkt[UDP].dport)
    return (src, dst, proto, sport, dport)

def flow_iats(path: str):
    last_ts = {}
    values = []
    with PcapReader(path) as reader:
        for pkt in reader:
            ts = float(pkt.time)
            key = flow_key_from_packet(pkt)
            if key is None:
                continue
            if key in last_ts:
                dt = ts - last_ts[key]
                if dt >= 0:
                    values.append(dt)
            last_ts[key] = ts
    return values

def basic_summary(path: str):
    lengths, ts_list, hashes = pkt_lengths_and_ts(path)
    return {
        "path": str(path),
        "packet_count": len(lengths),
        "total_packet_bytes": int(sum(lengths)),
        "first_ts": float(ts_list[0]) if ts_list else None,
        "last_ts": float(ts_list[-1]) if ts_list else None,
        "duration_sec": float(ts_list[-1] - ts_list[0]) if len(ts_list) >= 2 else 0.0,
        "iat_count": max(0, len(ts_list) - 1),
    }

def multiset_difference(before_hashes, after_hashes):
    from collections import Counter
    c_before = Counter(before_hashes)
    c_after = Counter(after_hashes)
    removed = 0
    added = 0
    keys = set(c_before) | set(c_after)
    for key in keys:
        diff = c_after.get(key, 0) - c_before.get(key, 0)
        if diff > 0:
            added += diff
        elif diff < 0:
            removed += -diff
    return {"added_packets": added, "removed_packets": removed}
