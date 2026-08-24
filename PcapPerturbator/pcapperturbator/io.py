from __future__ import annotations


class PcapSinkBuffered:
    """Buffered classic-PCAP writer using little-endian headers."""

    def __init__(self, out_path: str, linktype: int = 1, buf_bytes: int = 8 * 1024 * 1024):
        self._file = open(out_path, "wb", buffering=0)
        self._file.write(
            b"\xd4\xc3\xb2\xa1"
            + b"\x02\x00"
            + b"\x04\x00"
            + b"\x00\x00\x00\x00"
            + b"\x00\x00\x00\x00"
            + b"\xff\xff\x00\x00"
            + linktype.to_bytes(4, "little")
        )
        self._buffer = bytearray()
        self._limit = int(buf_bytes)

    def write_raw(self, ts_sec: int, ts_usec: int, pkt_bytes: bytes) -> None:
        incl_len = len(pkt_bytes)
        self._buffer += (
            int(ts_sec).to_bytes(4, "little")
            + int(ts_usec).to_bytes(4, "little")
            + int(incl_len).to_bytes(4, "little")
            + int(incl_len).to_bytes(4, "little")
        )
        self._buffer += pkt_bytes
        if len(self._buffer) >= self._limit:
            self._file.write(self._buffer)
            self._buffer.clear()

    def write_raw_record(self, record: bytes) -> None:
        self._buffer += record
        if len(self._buffer) >= self._limit:
            self._file.write(self._buffer)
            self._buffer.clear()

    def flush(self) -> None:
        if self._buffer:
            self._file.write(self._buffer)
            self._buffer.clear()

    def close(self) -> None:
        self.flush()
        self._file.close()
