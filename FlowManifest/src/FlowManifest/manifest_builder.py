"""
Manifest Builder: Unified manifest.csv management.

The manifest contains one row per (parent_flow, variant) pair with:
- Dataset name
- Parent flow ID
- Variant ID
- Variant type
- Split assignment (train/val/test)
- Label
- Group ID (for group-aware splits)
- Source PCAP
- Processed CSV path
- Metadata about perturbation
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from .parent_flow_id import generate_variant_id


class Variant(str, Enum):
    """Supported variant types."""
    ORIGIN = "origin"
    CASE1_LOSS = "case1_loss"
    CASE2_RETRANSMIT = "case2_retransmission"
    CASE3_REORDER = "case3_reordering"
    CASE4_LENGTH = "case4_length_padding"
    CASE5_RATE = "case5_timing_delay"

    @classmethod
    def all(cls) -> List["Variant"]:
        return [cls.ORIGIN, cls.CASE1_LOSS, cls.CASE2_RETRANSMIT, cls.CASE3_REORDER, cls.CASE4_LENGTH, cls.CASE5_RATE]

    @classmethod
    def perturbed(cls) -> List["Variant"]:
        return [cls.CASE1_LOSS, cls.CASE2_RETRANSMIT, cls.CASE3_REORDER, cls.CASE4_LENGTH, cls.CASE5_RATE]


class Split(str, Enum):
    """Split assignments."""
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


@dataclass
class ManifestEntry:
    """Single entry in the manifest."""
    dataset: str
    parent_flow_id: str
    variant_id: str
    variant: str
    split: Optional[str]  # train/val/test, may be None before split assignment
    label: str
    label_id: int
    group_id: str
    source_pcap: str
    processed_path: str
    num_packets: int
    start_time: float
    end_time: float
    perturbation_config: str = "{}"  # JSON string of perturbation params
    perturbation_applied: bool = True
    feature_schema_version: str = "1.0"
    sanitization_version: str = "1.0"
    checksum: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ManifestEntry":
        # Handle optional fields that may be missing
        d = d.copy()
        if "perturbation_applied" not in d:
            d["perturbation_applied"] = True
        if "feature_schema_version" not in d:
            d["feature_schema_version"] = "1.0"
        if "sanitization_version" not in d:
            d["sanitization_version"] = "1.0"
        if "checksum" not in d:
            d["checksum"] = ""
        return cls(**d)


class ManifestBuilder:
    """
    Build and manage the unified manifest.csv.

    Usage:
        builder = ManifestBuilder(dataset_name="CIC-IDS-2018")
        builder.add_flow(...)
        builder.add_variant(...)
        builder.save(output_path)
    """

    # Manifest columns in order
    COLUMNS = [
        "dataset",
        "parent_flow_id",
        "variant_id",
        "variant",
        "split",
        "label",
        "label_id",
        "group_id",
        "source_pcap",
        "processed_path",
        "num_packets",
        "start_time",
        "end_time",
        "perturbation_config",
        "perturbation_applied",
        "feature_schema_version",
        "sanitization_version",
        "checksum",
    ]

    def __init__(
        self,
        dataset_name: str,
        feature_schema_version: str = "1.0",
        sanitization_version: str = "1.0",
    ):
        self.dataset_name = dataset_name
        self.feature_schema_version = feature_schema_version
        self.sanitization_version = sanitization_version
        self.entries: Dict[str, ManifestEntry] = {}  # variant_id -> entry
        self.label_to_id: Dict[str, int] = {}

    def _compute_checksum(self, entry: ManifestEntry) -> str:
        """Compute a checksum for the manifest entry."""
        h = hashlib.sha256()
        h.update(entry.dataset.encode("utf-8"))
        h.update(entry.parent_flow_id.encode("utf-8"))
        h.update(entry.variant.encode("utf-8"))
        h.update(entry.label.encode("utf-8"))
        h.update(str(entry.num_packets).encode("utf-8"))
        h.update(str(entry.start_time).encode("utf-8"))
        h.update(str(entry.end_time).encode("utf-8"))
        return h.hexdigest()[:32]

    def add_flow(
        self,
        parent_flow_id: str,
        label: str,
        group_id: str,
        source_pcap: str,
        processed_path: str,
        num_packets: int,
        start_time: float,
        end_time: float,
        split: Optional[str] = None,
    ) -> str:
        """
        Add the origin variant for a parent flow.
        Returns the variant_id.
        """
        # Assign label_id
        if label not in self.label_to_id:
            self.label_to_id[label] = len(self.label_to_id)
        label_id = self.label_to_id[label]

        variant = Variant.ORIGIN.value
        variant_id = generate_variant_id(parent_flow_id, variant)

        entry = ManifestEntry(
            dataset=self.dataset_name,
            parent_flow_id=parent_flow_id,
            variant_id=variant_id,
            variant=variant,
            split=split,
            label=label,
            label_id=label_id,
            group_id=group_id,
            source_pcap=source_pcap,
            processed_path=processed_path,
            num_packets=num_packets,
            start_time=start_time,
            end_time=end_time,
            perturbation_config="{}",
            perturbation_applied=False,
            feature_schema_version=self.feature_schema_version,
            sanitization_version=self.sanitization_version,
            checksum="",
        )
        entry.checksum = self._compute_checksum(entry)
        self.entries[variant_id] = entry
        return variant_id

    def add_variant(
        self,
        parent_flow_id: str,
        variant: str,
        processed_path: str,
        perturbation_config: str = "{}",
        perturbation_applied: bool = True,
        num_packets: Optional[int] = None,
        split: Optional[str] = None,
    ) -> Optional[str]:
        """
        Add a perturbed variant for an existing parent flow.
        Returns the variant_id, or None if the origin doesn't exist.
        """
        # Find origin entry to inherit metadata
        origin_variant_id = generate_variant_id(parent_flow_id, Variant.ORIGIN.value)
        if origin_variant_id not in self.entries:
            return None

        origin = self.entries[origin_variant_id]
        variant_id = generate_variant_id(parent_flow_id, variant)

        entry = ManifestEntry(
            dataset=self.dataset_name,
            parent_flow_id=parent_flow_id,
            variant_id=variant_id,
            variant=variant,
            split=split if split is not None else origin.split,
            label=origin.label,
            label_id=origin.label_id,
            group_id=origin.group_id,
            source_pcap=origin.source_pcap,
            processed_path=processed_path,
            num_packets=num_packets if num_packets is not None else origin.num_packets,
            start_time=origin.start_time,
            end_time=origin.end_time,
            perturbation_config=perturbation_config,
            perturbation_applied=perturbation_applied,
            feature_schema_version=self.feature_schema_version,
            sanitization_version=self.sanitization_version,
            checksum="",
        )
        entry.checksum = self._compute_checksum(entry)
        self.entries[variant_id] = entry
        return variant_id

    def add_all_variants_for_flow(
        self,
        parent_flow_id: str,
        label: str,
        group_id: str,
        source_pcap: str,
        origin_path: str,
        variant_paths: Dict[str, str],
        num_packets: int,
        start_time: float,
        end_time: float,
        split: Optional[str] = None,
    ) -> List[str]:
        """
        Add origin and all perturbed variants for a parent flow.
        variant_paths is {variant_name: processed_path}.
        Returns list of variant_ids.
        """
        variant_ids = []

        # Add origin
        origin_vid = self.add_flow(
            parent_flow_id=parent_flow_id,
            label=label,
            group_id=group_id,
            source_pcap=source_pcap,
            processed_path=origin_path,
            num_packets=num_packets,
            start_time=start_time,
            end_time=end_time,
            split=split,
        )
        variant_ids.append(origin_vid)

        # Add perturbed variants
        for variant, path in variant_paths.items():
            vid = self.add_variant(
                parent_flow_id=parent_flow_id,
                variant=variant,
                processed_path=path,
                split=split,
            )
            if vid is not None:
                variant_ids.append(vid)

        return variant_ids

    def update_split(self, parent_flow_id: str, split: str) -> int:
        """Update split assignment for all variants of a parent flow. Returns number updated."""
        count = 0
        for entry in self.entries.values():
            if entry.parent_flow_id == parent_flow_id:
                entry.split = split
                count += 1
        return count

    def get_parent_flows(self) -> Set[str]:
        """Get all unique parent_flow_ids."""
        return set(e.parent_flow_id for e in self.entries.values())

    def get_entries_for_parent(self, parent_flow_id: str) -> List[ManifestEntry]:
        """Get all entries for a parent flow."""
        return [e for e in self.entries.values() if e.parent_flow_id == parent_flow_id]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert manifest to DataFrame."""
        if not self.entries:
            return pd.DataFrame(columns=self.COLUMNS)

        rows = [e.to_dict() for e in self.entries.values()]
        df = pd.DataFrame(rows)

        # Ensure columns are in order and all present
        for col in self.COLUMNS:
            if col not in df.columns:
                df[col] = None

        return df[self.COLUMNS]

    def save(self, output_path: str) -> None:
        """Save manifest to CSV."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df = self.to_dataframe()
        df.to_csv(output_path, index=False)

    @classmethod
    def load(cls, input_path: str) -> "ManifestBuilder":
        """Load manifest from CSV."""
        df = pd.read_csv(input_path)

        # Infer dataset name from first row
        dataset_name = df["dataset"].iloc[0] if len(df) > 0 else "unknown"

        builder = cls(dataset_name=dataset_name)

        for _, row in df.iterrows():
            entry = ManifestEntry.from_dict(row.to_dict())
            builder.entries[entry.variant_id] = entry

            # Rebuild label_to_id
            if entry.label not in builder.label_to_id:
                builder.label_to_id[entry.label] = entry.label_id

        return builder

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate manifest integrity. Returns (is_valid, list of issues)."""
        issues = []

        # Check that every parent flow has origin
        parent_flows = self.get_parent_flows()
        for parent_id in parent_flows:
            origin_vid = generate_variant_id(parent_id, Variant.ORIGIN.value)
            if origin_vid not in self.entries:
                issues.append(f"Parent flow {parent_id} missing origin variant")

        # Check that all variants of a parent have the same split
        for parent_id in parent_flows:
            entries = self.get_entries_for_parent(parent_id)
            splits = set(e.split for e in entries)
            if len(splits) > 1:
                issues.append(f"Parent flow {parent_id} has inconsistent splits: {splits}")

        # Check that group_id is consistent for a parent flow
        for parent_id in parent_flows:
            entries = self.get_entries_for_parent(parent_id)
            groups = set(e.group_id for e in entries)
            if len(groups) > 1:
                issues.append(f"Parent flow {parent_id} has inconsistent group_ids: {groups}")

        # Check that variant_id is correctly formed
        for entry in self.entries.values():
            expected_vid = generate_variant_id(entry.parent_flow_id, entry.variant)
            if entry.variant_id != expected_vid:
                issues.append(f"Invalid variant_id for {entry.parent_flow_id}:{entry.variant}")

        return len(issues) == 0, issues
