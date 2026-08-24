"""
Manifest-based Dataset: Load data from manifest with different modes.

Supported modes:
- pretrain: Train split, origin only, no labels
- clean_ft: Support set, origin only, labels
- pa_ft: Support set, all variants, labels, balanced sampling
- val: Val split, all variants
- test: Test split, all variants
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from .manifest_builder import ManifestBuilder, ManifestEntry, Variant, Split
from .sanitization import FeatureSanitizer, SanitizationResult
from .support_set import SupportSet


class DatasetMode(str, Enum):
    """Dataset loading modes."""
    PRETRAIN = "pretrain"
    CLEAN_FT = "clean_ft"
    PA_FT = "pa_ft"
    VAL = "val"
    TEST = "test"


# Aliases for backward compatibility
PretrainMode = DatasetMode.PRETRAIN
CleanFTMode = DatasetMode.CLEAN_FT
PAFTMode = DatasetMode.PA_FT
ValMode = DatasetMode.VAL
TestMode = DatasetMode.TEST


@dataclass
class DatasetEntry:
    """A single dataset entry with metadata."""
    parent_flow_id: str
    variant: str
    features: np.ndarray
    label: Optional[int]
    label_name: Optional[str]
    split: Optional[str]


class ManifestDataset(Dataset):
    """
    Dataset that loads from a manifest.

    Usage:
        dataset = ManifestDataset(
            manifest_path="data/manifest.csv",
            mode=DatasetMode.TRAIN,
        )
    """

    def __init__(
        self,
        manifest_path: str,
        mode: DatasetMode,
        index_path: Optional[str] = None,
        split: Optional[str] = None,
        variants: Optional[List[str]] = None,
        support_set: Optional[SupportSet] = None,
        label_column: str = "label",
        sanitize: bool = True,
        sanitizer: Optional[FeatureSanitizer] = None,
        phase_prefix_pattern: str = r"^p(\d+)_",
        phase_suffix_pattern: str = r"_p(\d+)$",
    ):
        """
        Initialize manifest dataset.

        Args:
            manifest_path: Path to manifest.csv
            mode: Dataset mode
            index_path: Optional path to index file (pretrain pool, support set)
            split: Filter by split (overrides mode default)
            variants: Filter by variants (overrides mode default)
            support_set: Support set for clean_ft/pa_ft modes
            label_column: Name of label column
            sanitize: Whether to sanitize features
            sanitizer: Custom feature sanitizer
            phase_prefix_pattern: Regex for phase column prefix
            phase_suffix_pattern: Regex for phase column suffix
        """
        self.manifest_path = Path(manifest_path)
        self.mode = mode
        self.label_column = label_column
        self.sanitize = sanitize
        self.sanitizer = sanitizer or FeatureSanitizer()

        # Compile phase regex
        self.phase_prefix_re = re.compile(phase_prefix_pattern)
        self.phase_suffix_re = re.compile(phase_suffix_pattern)

        # Load manifest
        self.manifest = ManifestBuilder.load(str(manifest_path))

        # Determine filters based on mode
        self.filter_split, self.filter_variants, self.filter_parents = self._get_mode_filters(
            mode=mode,
            split=split,
            variants=variants,
            support_set=support_set,
            index_path=index_path,
        )

        # Load and prepare data
        self.entries: List[ManifestEntry] = []
        self.data: pd.DataFrame = pd.DataFrame()
        self.feature_scaler: Optional[StandardScaler] = None
        self.phase_feature_names: Dict[int, List[str]] = {}
        self.feats_per_phase: int = 0
        self.K: int = 1
        self.sanitization_result: Optional[SanitizationResult] = None

        self._prepare_data()

    def _load_index_file(self, index_path: str) -> Set[str]:
        """Load parent_flow_ids from an index file."""
        df = pd.read_csv(index_path, comment="#")
        if "parent_flow_id" in df.columns:
            return set(df["parent_flow_id"])
        raise ValueError(f"Index file {index_path} has no parent_flow_id column")

    def _get_mode_filters(
        self,
        mode: DatasetMode,
        split: Optional[str],
        variants: Optional[List[str]],
        support_set: Optional[SupportSet],
        index_path: Optional[str],
    ) -> Tuple[Optional[str], Optional[List[str]], Optional[Set[str]]]:
        """Get split/variant/parent filters based on mode."""
        filter_split: Optional[str] = split
        filter_variants: Optional[List[str]] = variants
        filter_parents: Optional[Set[str]] = None

        if mode == DatasetMode.PRETRAIN:
            if filter_split is None:
                filter_split = Split.TRAIN.value
            if filter_variants is None:
                filter_variants = [Variant.ORIGIN.value]
            if index_path:
                filter_parents = self._load_index_file(index_path)

        elif mode == DatasetMode.CLEAN_FT:
            if filter_split is None:
                filter_split = Split.TRAIN.value
            if filter_variants is None:
                filter_variants = [Variant.ORIGIN.value]
            if support_set:
                filter_parents = support_set.parent_flow_ids
            elif index_path:
                filter_parents = self._load_index_file(index_path)

        elif mode == DatasetMode.PA_FT:
            if filter_split is None:
                filter_split = Split.TRAIN.value
            if filter_variants is None:
                filter_variants = [v.value for v in Variant.all()]
            if support_set:
                filter_parents = support_set.parent_flow_ids
            elif index_path:
                filter_parents = self._load_index_file(index_path)

        elif mode == DatasetMode.VAL:
            if filter_split is None:
                filter_split = Split.VAL.value
            if filter_variants is None:
                filter_variants = [v.value for v in Variant.all()]

        elif mode == DatasetMode.TEST:
            if filter_split is None:
                filter_split = Split.TEST.value
            if filter_variants is None:
                filter_variants = [v.value for v in Variant.all()]

        return filter_split, filter_variants, filter_parents

    def _apply_filters(self) -> List[ManifestEntry]:
        """Apply filters to manifest entries."""
        filtered: List[ManifestEntry] = []

        for entry in self.manifest.entries.values():
            # Filter by split
            if self.filter_split is not None and entry.split != self.filter_split:
                continue

            # Filter by variant
            if self.filter_variants is not None and entry.variant not in self.filter_variants:
                continue

            # Filter by parent
            if self.filter_parents is not None and entry.parent_flow_id not in self.filter_parents:
                continue

            filtered.append(entry)

        return filtered

    def _prepare_data(self) -> None:
        """Load and prepare all data."""
        # Get filtered entries
        self.entries = self._apply_filters()
        if not self.entries:
            raise ValueError("No entries found matching filters")

        # Load CSV files and concatenate
        dfs: List[pd.DataFrame] = []
        for entry in self.entries:
            path = Path(entry.processed_path)
            if not path.exists():
                # Try relative to manifest
                path = self.manifest_path.parent / entry.processed_path

            if path.exists():
                df = pd.read_csv(path)
                # Add metadata columns for reference
                df["_parent_flow_id"] = entry.parent_flow_id
                df["_variant"] = entry.variant
                df["_label"] = entry.label
                df["_label_id"] = entry.label_id
                df["_split"] = entry.split
                dfs.append(df)
            else:
                print(f"Warning: File not found: {entry.processed_path}")

        if not dfs:
            raise ValueError("No data files could be loaded")

        self.data = pd.concat(dfs, ignore_index=True)

        # Sanitize features
        if self.sanitize:
            self.data, self.sanitization_result = self.sanitizer.sanitize(
                self.data,
                label_column="_label",
            )

        # Detect phase structure
        self._detect_phase_structure()

        # Prepare features
        self._prepare_features()

    def _detect_phase_structure(self) -> None:
        """Detect phase columns and structure."""
        # Skip metadata columns
        metadata_cols = {"_parent_flow_id", "_variant", "_label", "_label_id", "_split"}
        feature_cols = [c for c in self.data.columns if c not in metadata_cols]

        phase_cols: Dict[int, List[str]] = {}

        for col in feature_cols:
            # Check for prefix: pN_xxx
            match_prefix = self.phase_prefix_re.match(col)
            if match_prefix:
                phase_num = int(match_prefix.group(1))
                phase_cols.setdefault(phase_num, []).append(col)
                continue

            # Check for suffix: xxx_pN
            match_suffix = self.phase_suffix_re.search(col)
            if match_suffix:
                phase_num = int(match_suffix.group(1))
                phase_cols.setdefault(phase_num, []).append(col)
                continue

            # No phase, treat as phase 1
            phase_cols.setdefault(1, []).append(col)

        if not phase_cols:
            raise ValueError("No feature columns found")

        # Determine K and check consistency
        self.K = max(phase_cols.keys()) if phase_cols else 1

        # Ensure all phases 1..K exist
        for p in range(1, self.K + 1):
            phase_cols.setdefault(p, [])

        # Sort phase columns
        for p in phase_cols:
            phase_cols[p].sort()

        self.phase_feature_names = phase_cols

        # Determine feats_per_phase
        lens = [len(cols) for cols in phase_cols.values() if cols]
        if not lens:
            raise ValueError("No features found in any phase")
        self.feats_per_phase = lens[0]

    def _prepare_features(self) -> None:
        """Prepare feature matrix."""
        # Build phase-wise feature matrix (N, K, F)
        N = len(self.data)
        self.phase_data = np.zeros((N, self.K, self.feats_per_phase), dtype=np.float32)

        for p in range(1, self.K + 1):
            cols = self.phase_feature_names.get(p, [])
            if not cols:
                continue
            # Take first feats_per_phase columns or pad
            take = min(len(cols), self.feats_per_phase)
            self.phase_data[:, p - 1, :take] = self.data[cols[:take]].values.astype(np.float32)

        # Fit scaler if in train mode
        if self.mode in {DatasetMode.PRETRAIN, DatasetMode.CLEAN_FT, DatasetMode.PA_FT}:
            self.feature_scaler = StandardScaler()
            flat = self.phase_data.reshape(N, -1)
            self.feature_scaler.fit(flat)
            flat_scaled = self.feature_scaler.transform(flat).astype(np.float32)
            self.phase_data = flat_scaled.reshape(N, self.K, self.feats_per_phase)

    def __len__(self) -> int:
        return len(self.phase_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Get item as (features, label)."""
        features = torch.from_numpy(self.phase_data[idx])

        if self.mode == DatasetMode.PRETRAIN:
            # Pretrain has no labels
            return features, None

        # Get label
        label_id = self.data.iloc[idx]["_label_id"]
        if pd.isna(label_id):
            return features, None

        return features, torch.tensor(int(label_id), dtype=torch.long)

    def get_metadata(self, idx: int) -> Dict:
        """Get metadata for an index."""
        row = self.data.iloc[idx]
        return {
            "parent_flow_id": row["_parent_flow_id"],
            "variant": row["_variant"],
            "label": row["_label"],
            "label_id": row["_label_id"],
            "split": row["_split"],
        }

    def get_parent_flow_ids(self) -> List[str]:
        """Get list of parent_flow_ids in dataset."""
        return list(self.data["_parent_flow_id"].unique())

    def get_variants(self) -> List[str]:
        """Get list of variants in dataset."""
        return list(self.data["_variant"].unique())

    def get_class_to_idx(self) -> Dict[str, int]:
        """Get label name to id mapping."""
        return {
            entry.label: entry.label_id
            for entry in self.entries
        }


def DatasetFromManifest(
    manifest_path: str,
    index_path: Optional[str] = None,
    split: Optional[str] = None,
    variants: Optional[List[str]] = None,
    mode: DatasetMode = DatasetMode.TEST,
    sanitize: bool = True,
) -> ManifestDataset:
    """
    Factory function to create a ManifestDataset.

    Args:
        manifest_path: Path to manifest.csv
        index_path: Optional index file
        split: Filter by split
        variants: Filter by variants
        mode: Dataset mode
        sanitize: Whether to sanitize features

    Returns:
        ManifestDataset
    """
    return ManifestDataset(
        manifest_path=manifest_path,
        mode=mode,
        index_path=index_path,
        split=split,
        variants=variants,
        sanitize=sanitize,
    )
