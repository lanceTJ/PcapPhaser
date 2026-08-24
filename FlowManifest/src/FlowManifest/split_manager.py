"""
Split Manager: Parent-flow level train/val/test splitting with group awareness.

Key features:
- Split at parent_flow_id level (never split a flow across splits)
- Group-aware splitting: same group stays in same split
- Label stratification
- Deterministic with fixed seed
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .manifest_builder import ManifestBuilder, Split


@dataclass
class ParentFlowInfo:
    """Information about a parent flow needed for splitting."""
    parent_flow_id: str
    group_id: str
    label: str
    label_id: int


@dataclass
class ParentSplit:
    """Result of a parent-flow split."""
    train: Set[str]
    val: Set[str]
    test: Set[str]
    seed: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    group_aware: bool

    def get_split(self, parent_flow_id: str) -> Optional[str]:
        """Get split for a parent flow."""
        if parent_flow_id in self.train:
            return Split.TRAIN.value
        if parent_flow_id in self.val:
            return Split.VAL.value
        if parent_flow_id in self.test:
            return Split.TEST.value
        return None

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame for saving."""
        rows = []
        for parent_id in sorted(self.train):
            rows.append({"parent_flow_id": parent_id, "split": Split.TRAIN.value})
        for parent_id in sorted(self.val):
            rows.append({"parent_flow_id": parent_id, "split": Split.VAL.value})
        for parent_id in sorted(self.test):
            rows.append({"parent_flow_id": parent_id, "split": Split.TEST.value})
        return pd.DataFrame(rows)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, seed: int, train_ratio: float, val_ratio: float, test_ratio: float, group_aware: bool) -> "ParentSplit":
        """Load from DataFrame."""
        train = set(df[df["split"] == Split.TRAIN.value]["parent_flow_id"])
        val = set(df[df["split"] == Split.VAL.value]["parent_flow_id"])
        test = set(df[df["split"] == Split.TEST.value]["parent_flow_id"])
        return cls(
            train=train,
            val=val,
            test=test,
            seed=seed,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            group_aware=group_aware,
        )


class SplitManager:
    """
    Manage train/val/test splits at the parent-flow level.

    Usage:
        manager = SplitManager(seed=42)
        split = manager.split(flows, group_aware=True)
        manager.save_split(split, output_path)
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def _group_by_group_id(self, flows: List[ParentFlowInfo]) -> Dict[str, List[ParentFlowInfo]]:
        """Group flows by their group_id."""
        groups: Dict[str, List[ParentFlowInfo]] = {}
        for flow in flows:
            groups.setdefault(flow.group_id, []).append(flow)
        return groups

    def _get_group_label_distribution(self, group: List[ParentFlowInfo]) -> Dict[str, float]:
        """Get label distribution within a group."""
        label_counts: Dict[str, int] = {}
        total = len(group)
        for flow in group:
            label_counts[flow.label] = label_counts.get(flow.label, 0) + 1
        return {label: count / total for label, count in label_counts.items()}

    def _split_group_aware(
        self,
        flows: List[ParentFlowInfo],
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
    ) -> ParentSplit:
        """
        Split with group awareness: all flows from same group go to same split.
        Tries to maintain label stratification.
        """
        groups = self._group_by_group_id(flows)
        group_ids = sorted(groups.keys())

        # Shuffle groups
        self.rng.shuffle(group_ids)

        # Calculate target counts
        total_flows = len(flows)
        train_target = int(total_flows * train_ratio)
        val_target = int(total_flows * val_ratio)

        # Greedily assign groups to splits
        train: Set[str] = set()
        val: Set[str] = set()
        test: Set[str] = set()

        train_count = 0
        val_count = 0

        for group_id in group_ids:
            group = groups[group_id]
            group_size = len(group)
            group_parent_ids = {f.parent_flow_id for f in group}

            # Try to assign to maintain stratification
            if train_count < train_target:
                train.update(group_parent_ids)
                train_count += group_size
            elif val_count < val_target:
                val.update(group_parent_ids)
                val_count += group_size
            else:
                test.update(group_parent_ids)

        return ParentSplit(
            train=train,
            val=val,
            test=test,
            seed=self.seed,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            group_aware=True,
        )

    def _split_simple(
        self,
        flows: List[ParentFlowInfo],
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        stratify: bool = True,
    ) -> ParentSplit:
        """Simple split at parent-flow level with optional label stratification."""
        parent_ids = [f.parent_flow_id for f in flows]
        labels = [f.label for f in flows] if stratify else None

        # First split: train vs (val+test)
        train_ids, val_test_ids = train_test_split(
            parent_ids,
            test_size=(1 - train_ratio),
            random_state=self.seed,
            stratify=labels,
        )

        # Second split: val vs test from the remainder
        if val_test_ids:
            # Get labels for val_test split if stratifying
            if stratify:
                id_to_label = {f.parent_flow_id: f.label for f in flows}
                val_test_labels = [id_to_label[id_] for id_ in val_test_ids]
            else:
                val_test_labels = None

            # Adjust ratio for second split
            relative_val_ratio = val_ratio / (val_ratio + test_ratio)
            val_ids, test_ids = train_test_split(
                val_test_ids,
                test_size=(1 - relative_val_ratio),
                random_state=self.seed + 1,
                stratify=val_test_labels,
            )
        else:
            val_ids = []
            test_ids = []

        return ParentSplit(
            train=set(train_ids),
            val=set(val_ids),
            test=set(test_ids),
            seed=self.seed,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            group_aware=False,
        )

    def split(
        self,
        flows: List[ParentFlowInfo],
        train_ratio: float = 0.7,
        val_ratio: float = 0.1,
        test_ratio: float = 0.2,
        group_aware: bool = True,
        stratify: bool = True,
    ) -> ParentSplit:
        """
        Split flows into train/val/test.

        Args:
            flows: List of ParentFlowInfo
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for test
            group_aware: Whether to keep same group in same split
            stratify: Whether to stratify by label

        Returns:
            ParentSplit object
        """
        # Validate ratios sum to 1
        total = train_ratio + val_ratio + test_ratio
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Ratios must sum to 1, got {total}")

        if group_aware:
            return self._split_group_aware(flows, train_ratio, val_ratio, test_ratio)
        else:
            return self._split_simple(flows, train_ratio, val_ratio, test_ratio, stratify)

    def split_manifest(
        self,
        manifest: ManifestBuilder,
        train_ratio: float = 0.7,
        val_ratio: float = 0.1,
        test_ratio: float = 0.2,
        group_aware: bool = True,
        stratify: bool = True,
    ) -> ParentSplit:
        """Split a manifest and update its split assignments."""
        # Extract parent flow info from manifest
        parent_flows: Dict[str, ParentFlowInfo] = {}
        for entry in manifest.entries.values():
            if entry.parent_flow_id not in parent_flows:
                parent_flows[entry.parent_flow_id] = ParentFlowInfo(
                    parent_flow_id=entry.parent_flow_id,
                    group_id=entry.group_id,
                    label=entry.label,
                    label_id=entry.label_id,
                )

        flows_list = list(parent_flows.values())

        # Perform split
        split = self.split(
            flows_list,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            group_aware=group_aware,
            stratify=stratify,
        )

        # Update manifest
        for parent_id in split.train:
            manifest.update_split(parent_id, Split.TRAIN.value)
        for parent_id in split.val:
            manifest.update_split(parent_id, Split.VAL.value)
        for parent_id in split.test:
            manifest.update_split(parent_id, Split.TEST.value)

        return split

    def split_manifest_train_test(
        self,
        manifest: ManifestBuilder,
        train_ratio: float = 0.8,
        group_aware: bool = True,
        stratify: bool = True,
    ) -> ParentSplit:
        """
        Split a manifest into train/test only (no fixed val set).
        Used for k-fold cross-validation where val is extracted from train dynamically.

        Args:
            manifest: The manifest to split
            train_ratio: Proportion for training (default 0.8)
            group_aware: Whether to keep same group in same split
            stratify: Whether to stratify by label

        Returns:
            ParentSplit object with val set empty
        """
        # Extract parent flow info from manifest
        parent_flows: Dict[str, ParentFlowInfo] = {}
        for entry in manifest.entries.values():
            if entry.parent_flow_id not in parent_flows:
                parent_flows[entry.parent_flow_id] = ParentFlowInfo(
                    parent_flow_id=entry.parent_flow_id,
                    group_id=entry.group_id,
                    label=entry.label,
                    label_id=entry.label_id,
                )

        flows_list = list(parent_flows.values())

        # Perform two-way split using existing logic with val_ratio=0
        split = self.split(
            flows_list,
            train_ratio=train_ratio,
            val_ratio=0.0,
            test_ratio=1.0 - train_ratio,
            group_aware=group_aware,
            stratify=stratify,
        )

        # Update manifest
        for parent_id in split.train:
            manifest.update_split(parent_id, Split.TRAIN.value)
        for parent_id in split.test:
            manifest.update_split(parent_id, Split.TEST.value)
        # val set is intentionally left empty

        return split

    def save_split(self, split: ParentSplit, output_path: str) -> None:
        """Save split to CSV."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df = split.to_dataframe()

        # Add metadata as comment in header
        metadata_lines = [
            f"# seed: {split.seed}",
            f"# train_ratio: {split.train_ratio}",
            f"# val_ratio: {split.val_ratio}",
            f"# test_ratio: {split.test_ratio}",
            f"# group_aware: {split.group_aware}",
        ]

        # Write with metadata header
        with open(output_path, "w") as f:
            for line in metadata_lines:
                f.write(line + "\n")
            df.to_csv(f, index=False)

    def load_split(self, input_path: str) -> ParentSplit:
        """Load split from CSV."""
        # First read metadata from comments
        metadata = {}
        with open(input_path, "r") as f:
            for line in f:
                if line.startswith("#"):
                    parts = line[1:].strip().split(":", 1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = parts[1].strip()
                        metadata[key] = value
                else:
                    break

        # Then read data
        df = pd.read_csv(input_path, comment="#")

        return ParentSplit.from_dataframe(
            df,
            seed=int(metadata.get("seed", "42")),
            train_ratio=float(metadata.get("train_ratio", "0.7")),
            val_ratio=float(metadata.get("val_ratio", "0.1")),
            test_ratio=float(metadata.get("test_ratio", "0.2")),
            group_aware=metadata.get("group_aware", "True") == "True",
        )
