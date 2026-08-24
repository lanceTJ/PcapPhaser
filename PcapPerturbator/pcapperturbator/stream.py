from __future__ import annotations

import os
import struct
from typing import Iterator, Tuple

from scapy.all import PcapNgReader, RawPcapReader

from .utils import log


_PCAP_MAGIC = {b"\xd4\xc3\xb2\xa1", b"\xa1\xb2\xc3\xd4", b"\x4d\x3c\xb2\xa1", b"\xa1\xb2\x3c\x4d"}
_PCAPNG_MAGIC = b"\x0a\x0d\x0d\x0a"



def _sniff_kind(path: str) -> str:
    with open(path, "rb") as handle:
        head = handle.read(4)
    if head in _PCAP_MAGIC:
        return "pcap"
    if head == _PCAPNG_MAGIC:
        return "pcapng"
    raise ValueError(f"Unknown capture format: {path}")



def stream_pcap_packets(pcap_path: str) -> Iterator[Tuple[bytes, int, int]]:
    """Yield (pkt_bytes, ts_sec, ts_usec) for PCAP and PCAPNG."""
    kind = _sniff_kind(pcap_path)
    if kind == "pcap":
        reader = RawPcapReader(pcap_path)
        try:
            for pkt_bytes, meta in reader:
                yield pkt_bytes, int(meta.sec), int(meta.usec)
        finally:
            reader.close()
        return

    reader = PcapNgReader(pcap_path)
    try:
        for pkt in reader:
            timestamp = float(getattr(pkt, "time", 0.0))
            ts_sec = int(timestamp)
            ts_usec = int((timestamp - ts_sec) * 1_000_000)
            yield bytes(pkt.original), ts_sec, ts_usec
    finally:
        reader.close()



def stream_raw_pcap_records(pcap_path: str) -> Iterator[bytes]:
    """
    Yield full packet records (16-byte header + packet bytes) from a classic PCAP file.
    Invalid or truncated records are repaired when possible instead of failing hard.
    """
    if _sniff_kind(pcap_path) != "pcap":
        raise ValueError("Raw record streaming only supports classic PCAP files")

    with open(pcap_path, "rb") as handle:
        global_header = handle.read(24)
        if len(global_header) < 24:
            log.warning("File %s is too small to contain a valid PCAP header", pcap_path)
            return

        magic = global_header[0:4]
        if magic in (b"\xd4\xc3\xb2\xa1", b"\x4d\x3c\xb2\xa1"):
            fmt_hdr = "<IIII"
            byte_order = "little"
        elif magic in (b"\xa1\xb2\xc3\xd4", b"\xa1\xb2\x3c\x4d"):
            fmt_hdr = ">IIII"
            byte_order = "big"
        else:
            raise ValueError(f"Invalid PCAP magic for {pcap_path}")

        snaplen = int.from_bytes(global_header[16:20], byte_order)
        if snaplen <= 0 or snaplen > 262144:
            log.warning("Invalid snaplen in %s. Falling back to 262144.", pcap_path)
            snaplen = 262144

        file_size = os.fstat(handle.fileno()).st_size
        discarded = 0

        while True:
            record_start = handle.tell()
            header = handle.read(16)
            if len(header) < 16:
                break

            try:
                ts_sec, ts_usec, incl_len, orig_len = struct.unpack(fmt_hdr, header)
            except struct.error:
                log.warning("Invalid packet header at offset %d in %s", record_start, pcap_path)
                discarded += 1
                continue

            remaining = file_size - handle.tell()
            if incl_len <= 0 or incl_len > snaplen or incl_len > remaining or incl_len > 1_048_576:
                log.warning(
                    "Repairing invalid incl_len=%d at offset %d in %s",
                    incl_len,
                    record_start,
                    pcap_path,
                )
                incl_len = min(snaplen, remaining)
                orig_len = incl_len
                if incl_len <= 0:
                    discarded += 1
                    continue

            packet = handle.read(incl_len)
            if len(packet) < incl_len:
                log.warning(
                    "Truncated packet at offset %d in %s. Adjusting recorded length.",
                    record_start,
                    pcap_path,
                )
                incl_len = len(packet)
                orig_len = incl_len

            if incl_len <= 0:
                discarded += 1
                continue

            repaired_header = struct.pack(fmt_hdr, ts_sec, ts_usec, incl_len, orig_len)
            yield repaired_header + packet

        if discarded:
            log.info("Discarded %d invalid records while reading %s", discarded, pcap_path)
