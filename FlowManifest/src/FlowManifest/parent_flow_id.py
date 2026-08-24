"""
Parent Flow ID Generation: Stable, deterministic parent_flow_id based on flow metadata.

The parent_flow_id must:
- Be identical for origin and all perturbed variants of the same flow
- Not depend on processing order
- Not change when flow is perturbed (packet timestamps/lengths)
- Be collision-resistant
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Tuple


@dataclass
class FlowMetadata:
    """Metadata required to generate a stable parent_flow_id."""
    dataset_name: str
    capture_id: str  # e.g., pcap filename without extension
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: int  # IP protocol number (6 for TCP, 17 for UDP, etc.)
    start_time: float  # Unix timestamp of first packet
    end_time: float  # Unix timestamp of last packet
    packet_count: int
    flow_index: int  # Index of flow within capture (to disambiguate collisions)


def canonical_5tuple(
    src_ip: str,
    dst_ip: str,
    src_port: int,
    dst_port: int,
    protocol: int,
) -> Tuple[str, str, int, int, int]:
    """
    Return canonical bidirectional 5-tuple: the smaller IP first, then smaller port if IPs equal.
    This ensures forward and reverse directions get the same 5-tuple.
    """
    ip_a, ip_b = src_ip, dst_ip
    port_a, port_b = src_port, dst_port

    # First compare IP addresses lexicographically
    if ip_a > ip_b:
        ip_a, ip_b = ip_b, ip_a
        port_a, port_b = port_b, port_a
    elif ip_a == ip_b:
        # If IPs are the same, use port to break tie
        if port_a > port_b:
            port_a, port_b = port_b, port_a

    return (ip_a, ip_b, port_a, port_b, protocol)


def _hash_string(s: str) -> bytes:
    """Hash a string to bytes using SHA-256."""
    return hashlib.sha256(s.encode("utf-8")).digest()


def _hash_float(f: float) -> bytes:
    """Hash a float in a stable way (avoid precision issues)."""
    # Convert to string with sufficient decimal digits to preserve timestamp precision
    return _hash_string(f"{f:.9f}")


def _hash_int(i: int) -> bytes:
    """Hash an integer."""
    return _hash_string(str(i))


def generate_parent_flow_id(
    metadata: FlowMetadata,
    hash_algorithm: str = "sha256",
) -> str:
    """
    Generate a stable, deterministic parent_flow_id from flow metadata.

    The ID is computed using:
    - dataset name
    - capture id
    - canonical 5-tuple
    - flow start/end timestamps
    - packet count
    - flow index within capture

    This ensures that:
    - Origin and perturbed variants get the same parent_flow_id
    - Different flows get different IDs
    - The ID is stable across re-runs
    """
    h = hashlib.new(hash_algorithm)

    # Dataset and capture
    h.update(_hash_string(metadata.dataset_name))
    h.update(b"||")
    h.update(_hash_string(metadata.capture_id))
    h.update(b"||")

    # Canonical 5-tuple
    canon = canonical_5tuple(
        metadata.src_ip,
        metadata.dst_ip,
        metadata.src_port,
        metadata.dst_port,
        metadata.protocol,
    )
    h.update(_hash_string(canon[0]))  # ip_a
    h.update(b"|")
    h.update(_hash_string(canon[1]))  # ip_b
    h.update(b"|")
    h.update(_hash_int(canon[2]))  # port_a
    h.update(b"|")
    h.update(_hash_int(canon[3]))  # port_b
    h.update(b"|")
    h.update(_hash_int(canon[4]))  # protocol
    h.update(b"||")

    # Timing and size
    h.update(_hash_float(metadata.start_time))
    h.update(b"|")
    h.update(_hash_float(metadata.end_time))
    h.update(b"|")
    h.update(_hash_int(metadata.packet_count))
    h.update(b"||")

    # Flow index
    h.update(_hash_int(metadata.flow_index))

    return h.hexdigest()


def generate_variant_id(parent_flow_id: str, variant: str) -> str:
    """Generate a variant-specific ID by combining parent_flow_id and variant name."""
    return f"{parent_flow_id}::{variant}"
