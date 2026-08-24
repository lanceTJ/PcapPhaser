"""
PCAP Integration: Bridge between PcapPerturbator and Manifest system.

This module handles:
1. Extracting flow metadata from PCAP to generate parent_flow_id
2. Coordinating PcapPerturbator perturbation with manifest registration
3. Managing the directory structure for origin + perturbed PCAPs
4. Running PSS feature extraction on perturbed PCAPs to generate CSVs
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from scapy.all import Ether, IP, TCP, UDP, PcapReader, PcapWriter

from .parent_flow_id import FlowMetadata, generate_parent_flow_id, canonical_5tuple
from .manifest_builder import ManifestBuilder, Variant
from .data_store import DataStore, StorageStrategy, DataLayout


@dataclass
class FlowKey:
    """Key for identifying a bidirectional flow."""
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: int

    def canonical_tuple(self) -> Tuple[str, str, int, int, int]:
        """Get canonical form (smaller IP first)."""
        return canonical_5tuple(
            self.src_ip, self.dst_ip, self.src_port, self.dst_port, self.protocol
        )

    def __hash__(self) -> int:
        return hash(self.canonical_tuple())

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, FlowKey):
            return False
        return self.canonical_tuple() == other.canonical_tuple()


@dataclass
class FlowStats:
    """Statistics extracted from a flow."""
    start_time: float = 0.0
    end_time: float = 0.0
    packet_count: int = 0
    total_bytes: int = 0
    packets: List[Tuple[float, bytes]] = field(default_factory=list)


def extract_flows_from_pcap(
    pcap_path: str,
    dataset_name: str,
    capture_id: str,
    flow_timeout: float = 300.0,  # 5 minutes
) -> Tuple[List[Tuple[FlowMetadata, List[Tuple[float, bytes]]]], Dict[str, Any]]:
    """
    Extract bidirectional flows from a PCAP file.

    Returns:
        List of (FlowMetadata, list_of_packets) tuples
        Dictionary of capture-level metadata
    """
    # Track flows by canonical key
    flows: Dict[FlowKey, FlowStats] = {}
    all_packets: List[Tuple[float, bytes]] = []

    # Capture metadata
    first_pkt_time: Optional[float] = None
    last_pkt_time: Optional[float] = None
    total_packets = 0

    with PcapReader(pcap_path) as pcap:
        for pkt in pcap:
            try:
                # Get timestamp if available
                ts = getattr(pkt, 'time', None)
                if ts is None:
                    ts = float(total_packets) * 0.001

                if first_pkt_time is None:
                    first_pkt_time = ts
                last_pkt_time = ts

                # Get raw bytes
                pkt_bytes = bytes(pkt)

                all_packets.append((ts, pkt_bytes))

                # Parse 5-tuple
                if IP in pkt:
                    ip = pkt[IP]
                    src_ip = ip.src
                    dst_ip = ip.dst
                    protocol = ip.proto

                    if TCP in pkt:
                        tcp = pkt[TCP]
                        src_port = tcp.sport
                        dst_port = tcp.dport
                    elif UDP in pkt:
                        udp = pkt[UDP]
                        src_port = udp.sport
                        dst_port = udp.dport
                    else:
                        # Non-TCP/UDP, skip flow tracking but keep packet
                        total_packets += 1
                        continue

                    # Create flow key
                    key = FlowKey(
                        src_ip=src_ip,
                        dst_ip=dst_ip,
                        src_port=src_port,
                        dst_port=dst_port,
                        protocol=protocol,
                    )

                    # Update flow stats
                    if key not in flows:
                        flows[key] = FlowStats(start_time=ts, end_time=ts)

                    stats = flows[key]
                    stats.end_time = max(stats.end_time, ts)
                    stats.packet_count += 1
                    stats.total_bytes += len(pkt_bytes)
                    stats.packets.append((ts, pkt_bytes))

                total_packets += 1

            except Exception:
                # Skip malformed packets
                total_packets += 1
                continue

    # Generate FlowMetadata for each flow
    flow_list: List[Tuple[FlowMetadata, List[Tuple[float, bytes]]]] = []
    flow_index = 0

    for key, stats in flows.items():
        flow_meta = FlowMetadata(
            dataset_name=dataset_name,
            capture_id=capture_id,
            src_ip=key.src_ip,
            dst_ip=key.dst_ip,
            src_port=key.src_port,
            dst_port=key.dst_port,
            protocol=key.protocol,
            start_time=stats.start_time,
            end_time=stats.end_time,
            packet_count=stats.packet_count,
            flow_index=flow_index,
        )
        flow_list.append((flow_meta, stats.packets))
        flow_index += 1

    capture_meta = {
        "pcap_path": pcap_path,
        "capture_id": capture_id,
        "dataset_name": dataset_name,
        "first_pkt_time": first_pkt_time,
        "last_pkt_time": last_pkt_time,
        "total_packets": total_packets,
        "num_flows": len(flow_list),
    }

    return flow_list, capture_meta


def write_single_flow_pcap(
    packets: List[Tuple[float, bytes]],
    output_path: str,
) -> None:
    """Write a single flow's packets to a PCAP file using scapy."""
    with PcapWriter(output_path, linktype=1) as writer:
        for ts_sec, pkt_bytes in packets:
            pkt = Ether(pkt_bytes)
            pkt.time = ts_sec
            writer.write(pkt)


@dataclass
class PerturbationPlan:
    """Plan for perturbing a single parent flow."""
    variant: str
    config: Dict[str, Any] = field(default_factory=dict)
    seed: int = 42


DEFAULT_PERTURBATION_PLANS: List[PerturbationPlan] = [
    PerturbationPlan(variant=Variant.CASE1_LOSS.value, config={"type": "loss", "pct": 0.05}),
    PerturbationPlan(variant=Variant.CASE2_RETRANSMIT.value, config={"type": "retransmit", "pct": 0.03}),
    PerturbationPlan(variant=Variant.CASE3_REORDER.value, config={"type": "reorder", "pct": 0.1, "params": {"m": 10}}),
    PerturbationPlan(variant=Variant.CASE4_LENGTH.value, config={"type": "length_manip"}),
    PerturbationPlan(variant=Variant.CASE5_RATE.value, config={"type": "rate_manip", "pct": 0.2}),
]


@dataclass
class PcapManifestPipeline:
    """
    Orchestrate raw PCAP -> parent flows -> perturbations -> manifest.

    The project-specific perturbation and PSS commands must be supplied as
    callables. This class deliberately fails if either runner is missing so an
    artifact cannot silently substitute copied PCAPs or synthetic features.

    Directory structure:
        processed_dir/
        ├── manifest.csv
        ├── splits/
        ├── indices/
        ├── raw_flows/           # Layer 1: origin single-flow PCAPs (always cached)
        │   └── {capture_id}/
        │       └── {parent_flow_id}.pcap
        ├── pcap/                # Layer 2: variant PCAPs (cached on demand or pre-generated)
        │   ├── case1_loss/
        │   ├── case2_retransmission/
        │   └── ...
        └── csv/                 # Layer 3: PSS-processed features (always stored)
            ├── origin/
            ├── case1_loss/
            └── ...
    """
    dataset_name: str
    raw_pcap_dir: Path
    processed_dir: Path
    pss_config_path: Optional[str] = None
    seed: int = 42
    storage_strategy: StorageStrategy = StorageStrategy.CACHE_ORIGIN
    perturbation_runner: Optional[
        Callable[[str, str, PerturbationPlan], None]
    ] = None
    pss_runner: Optional[Callable[[str, str], None]] = None

    def __post_init__(self) -> None:
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = ManifestBuilder(dataset_name=self.dataset_name)
        self.store = DataStore(
            self.processed_dir,
            config={
                "storage_strategy": self.storage_strategy.value,
                "layout": DataLayout.BY_VARIANT.value,
            }
        )

    def process_raw_pcap(
        self,
        pcap_path: str,
        capture_id: Optional[str] = None,
        label: str = "unknown",
        group_id: Optional[str] = None,
        perturbation_plans: Optional[List[PerturbationPlan]] = None,
    ) -> ManifestBuilder:
        """
        Process a single raw PCAP file into the manifest.

        Steps:
        1. Extract parent flows from PCAP
        2. Write origin flow PCAPs to raw_flows/
        3. Run PSS on origin to get CSV features
        4. Generate perturbed CSVs (and PCAPs if strategy == CACHE_ALL)
        5. Add all to manifest
        """
        if capture_id is None:
            capture_id = Path(pcap_path).stem

        if group_id is None:
            group_id = capture_id

        print(f"Processing {pcap_path} (capture: {capture_id})...")

        # Step 1: Extract parent flows
        flow_list, capture_meta = extract_flows_from_pcap(
            pcap_path,
            dataset_name=self.dataset_name,
            capture_id=capture_id,
        )
        print(f"  Extracted {len(flow_list)} flows")

        # Step 2: Process each flow
        for flow_meta, packets in flow_list:
            parent_flow_id = generate_parent_flow_id(flow_meta)

            # Write origin PCAP to raw_flows/
            origin_pcap_path = self.store.get_raw_flow_path(parent_flow_id, capture_id)
            origin_pcap_path.parent.mkdir(parents=True, exist_ok=True)
            write_single_flow_pcap(packets, str(origin_pcap_path))

            # Run PSS on origin to get CSV
            origin_csv_path = self.store.get_csv_path(parent_flow_id, Variant.ORIGIN.value, capture_id=capture_id)
            origin_csv_path.parent.mkdir(parents=True, exist_ok=True)
            self._run_pss_on_pcap(str(origin_pcap_path), str(origin_csv_path))

            # Add origin to manifest
            self.manifest.add_flow(
                parent_flow_id=parent_flow_id,
                label=label,
                group_id=group_id,
                source_pcap=pcap_path,
                processed_path=str(origin_csv_path.relative_to(self.processed_dir)),
                num_packets=flow_meta.packet_count,
                start_time=flow_meta.start_time,
                end_time=flow_meta.end_time,
            )

            # Step 3: Generate perturbations
            if perturbation_plans is None:
                perturbation_plans = DEFAULT_PERTURBATION_PLANS

            for plan in perturbation_plans:
                variant_csv_path = self.store.get_csv_path(parent_flow_id, plan.variant, capture_id=capture_id)
                variant_csv_path.parent.mkdir(parents=True, exist_ok=True)

                if self.storage_strategy == StorageStrategy.CACHE_ALL:
                    # Keep variant PCAP on disk
                    variant_pcap_path = self.store.get_pcap_path(parent_flow_id, plan.variant, capture_id=capture_id)
                    variant_pcap_path.parent.mkdir(parents=True, exist_ok=True)
                    self._run_perturbation_on_pcap(
                        str(origin_pcap_path),
                        str(variant_pcap_path),
                        plan,
                    )
                    self._run_pss_on_pcap(str(variant_pcap_path), str(variant_csv_path))
                else:
                    # Generate variant PCAP in a temp file, keep only CSV
                    with tempfile.NamedTemporaryFile(suffix=".pcap", delete=False) as tmp:
                        tmp_pcap = tmp.name
                    try:
                        self._run_perturbation_on_pcap(
                            str(origin_pcap_path),
                            tmp_pcap,
                            plan,
                        )
                        self._run_pss_on_pcap(tmp_pcap, str(variant_csv_path))
                    finally:
                        try:
                            os.unlink(tmp_pcap)
                        except FileNotFoundError:
                            pass

                # Add variant to manifest
                self.manifest.add_variant(
                    parent_flow_id=parent_flow_id,
                    variant=plan.variant,
                    processed_path=str(variant_csv_path.relative_to(self.processed_dir)),
                    perturbation_config=json.dumps(plan.config),
                    perturbation_applied=True,
                    num_packets=flow_meta.packet_count,
                )

        return self.manifest

    def _run_perturbation_on_pcap(
        self,
        input_pcap: str,
        output_pcap: str,
        plan: PerturbationPlan,
    ) -> None:
        """Run perturbation on a single flow PCAP using PcapPerturbator."""
        if self.perturbation_runner is None:
            raise RuntimeError(
                "PcapManifestPipeline requires a perturbation_runner callback. "
                "Wire this callback to PcapPerturbator for the plan formats "
                "used by your experiment."
            )
        self.perturbation_runner(input_pcap, output_pcap, plan)
        if not Path(output_pcap).is_file():
            raise RuntimeError(
                f"Perturbation runner did not create the expected PCAP: {output_pcap}"
            )

    def _run_pss_on_pcap(
        self,
        input_pcap: str,
        output_csv: str,
    ) -> None:
        """Run PSS feature extraction on a PCAP file."""
        if self.pss_runner is None:
            raise RuntimeError(
                "PcapManifestPipeline requires a pss_runner callback. "
                "Wire this callback to the PSS pipeline and write its final "
                "feature row to output_csv."
            )
        self.pss_runner(input_pcap, output_csv)
        if not Path(output_csv).is_file():
            raise RuntimeError(
                f"PSS runner did not create the expected CSV: {output_csv}"
            )

    def save_manifest(self) -> None:
        """Save manifest to processed directory."""
        manifest_path = self.processed_dir / "manifest.csv"
        self.manifest.save(str(manifest_path))
        print(f"Saved manifest to {manifest_path}")

    def load_manifest(self) -> None:
        """Load manifest from processed directory."""
        manifest_path = self.processed_dir / "manifest.csv"
        if manifest_path.exists():
            self.manifest = ManifestBuilder.load(str(manifest_path))
            print(f"Loaded manifest from {manifest_path}")

    def process_all_pcaps(
        self,
        label_map: Optional[Dict[str, str]] = None,
    ) -> ManifestBuilder:
        """
        Process all PCAPs in raw_pcap_dir.

        label_map: {pcap_stem: label} for labeling
        """
        label_map = label_map or {}

        # Find all PCAPs
        pcap_files = list(self.raw_pcap_dir.glob("*.pcap"))
        pcap_files.extend(self.raw_pcap_dir.glob("*.pcapng"))

        for pcap_path in sorted(pcap_files):
            capture_id = pcap_path.stem
            label = label_map.get(capture_id, "unknown")
            self.process_raw_pcap(
                str(pcap_path),
                capture_id=capture_id,
                label=label,
                group_id=capture_id,
            )

        self.save_manifest()
        return self.manifest


def create_manifest_from_existing_pcaps(
    processed_dir: str,
    dataset_name: str,
    pcap_dir_structure: Dict[str, str],  # {variant: dir_path}
) -> ManifestBuilder:
    """
    Create manifest from already-processed PCAP files.

    Use this if you already have origin + perturbed PCAPs generated.
    """
    builder = ManifestBuilder(dataset_name=dataset_name)

    # First pass: process origin to get parent_flow_ids
    origin_dir = Path(pcap_dir_structure.get(Variant.ORIGIN.value, ""))
    if not origin_dir.exists():
        raise ValueError(f"Origin directory not found: {origin_dir}")

    # Scan origin dir
    flow_index = 0
    parent_map: Dict[str, Tuple[FlowMetadata, Path]] = {}  # stem: (meta, csv_path)

    for capture_dir in origin_dir.iterdir():
        if not capture_dir.is_dir():
            continue
        capture_id = capture_dir.name

        for pcap_path in capture_dir.glob("*.pcap"):
            # Try to extract metadata from PCAP
            flow_list, _ = extract_flows_from_pcap(
                str(pcap_path),
                dataset_name=dataset_name,
                capture_id=capture_id,
            )

            if flow_list:
                flow_meta, _ = flow_list[0]  # Single-flow PCAP
                flow_meta.flow_index = flow_index
                parent_flow_id = generate_parent_flow_id(flow_meta)
                parent_map[pcap_path.stem] = (flow_meta, pcap_path)
                flow_index += 1

    # Add all variants
    for variant, variant_dir in pcap_dir_structure.items():
        variant_path = Path(variant_dir)
        if not variant_path.exists():
            continue

        for capture_dir in variant_path.iterdir():
            if not capture_dir.is_dir():
                continue
            capture_id = capture_dir.name

            for pcap_path in capture_dir.glob("*.pcap"):
                stem = pcap_path.stem
                if stem not in parent_map:
                    continue

                flow_meta, _ = parent_map[stem]
                parent_flow_id = generate_parent_flow_id(flow_meta)

                # Find corresponding CSV
                csv_path = pcap_path.with_suffix(".csv")
                if not csv_path.exists():
                    csv_path = variant_path / "csv" / capture_id / f"{stem}.csv"

                if variant == Variant.ORIGIN.value:
                    builder.add_flow(
                        parent_flow_id=parent_flow_id,
                        label="unknown",  # Would need label map
                        group_id=capture_id,
                        source_pcap=str(pcap_path),
                        processed_path=str(csv_path) if csv_path.exists() else str(pcap_path),
                        num_packets=flow_meta.packet_count,
                        start_time=flow_meta.start_time,
                        end_time=flow_meta.end_time,
                    )
                else:
                    builder.add_variant(
                        parent_flow_id=parent_flow_id,
                        variant=variant,
                        processed_path=str(csv_path) if csv_path.exists() else str(pcap_path),
                    )

    return builder


__all__ = [
    "FlowKey",
    "FlowStats",
    "PerturbationPlan",
    "PcapManifestPipeline",
    "extract_flows_from_pcap",
    "write_single_flow_pcap",
    "create_manifest_from_existing_pcaps",
]
