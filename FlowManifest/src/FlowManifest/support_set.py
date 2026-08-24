"""
Support Set Generator: Few-shot support set generation from train split.

Key properties:
- Only samples from train split
- Label-balanced sampling
- All models use the same support set
- Clean-FT and PA-FT use the SAME parent_flow_ids
- PA-FT just adds perturbed variants from the same parents
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pandas as pd

from .manifest_builder import ManifestBuilder, Split


@dataclass
class SupportSet:
    """Few-shot support set."""
    parent_flow_ids: Set[str]
    k: int  # Number of shots per class
    seed: int
    label_balanced: bool

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame for saving."""
        df = pd.DataFrame({
            "parent_flow_id": sorted(self.parent_flow_ids),
        })
        return df

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, k: int, seed: int, label_balanced: bool) -> "SupportSet":
        """Load from DataFrame."""
        return cls(
            parent_flow_ids=set(df["parent_flow_id"]),
            k=k,
            seed=seed,
            label_balanced=label_balanced,
        )


class SupportSetGenerator:
    """
    Generate few-shot support sets from the train split.

    Usage:
        generator = SupportSetGenerator(seed=42)
        support_set = generator.generate(manifest, k=50)
        generator.save(support_set, output_path)
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def _group_by_label(self, manifest: ManifestBuilder) -> Dict[str, List[str]]:
        """Group train split parent flows by label."""
        label_groups: Dict[str, List[str]] = {}

        seen_parents: Set[str] = set()
        for entry in manifest.entries.values():
            if entry.split != Split.TRAIN.value:
                continue
            if entry.parent_flow_id in seen_parents:
                continue
            seen_parents.add(entry.parent_flow_id)
            label_groups.setdefault(entry.label, []).append(entry.parent_flow_id)

        # Shuffle each group
        for label in label_groups:
            self.rng.shuffle(label_groups[label])

        return label_groups

    def generate(
        self,
        manifest: ManifestBuilder,
        k: int = 50,
        label_balanced: bool = True,
    ) -> SupportSet:
        """
        Generate a support set.

        Args:
            manifest: The manifest
            k: Number of examples per class
            label_balanced: Whether to balance by label

        Returns:
            SupportSet object
        """
        label_groups = self._group_by_label(manifest)

        if not label_groups:
            raise ValueError("No train split flows found in manifest")

        if label_balanced:
            # Label-balanced sampling: k per class
            selected: Set[str] = set()
            for label, parent_ids in label_groups.items():
                # Take min(k, available)
                take = min(k, len(parent_ids))
                selected.update(parent_ids[:take])
        else:
            # Simple random sampling
            all_train_ids = [pid for pids in label_groups.values() for pid in pids]
            take = min(k * len(label_groups), len(all_train_ids))
            selected = set(self.rng.choice(all_train_ids, size=take, replace=False))

        return SupportSet(
            parent_flow_ids=selected,
            k=k,
            seed=self.seed,
            label_balanced=label_balanced,
        )

    def generate_multiple_seeds(
        self,
        manifest: ManifestBuilder,
        k: int,
        seeds: List[int],
        label_balanced: bool = True,
    ) -> List[SupportSet]:
        """Generate multiple support sets with different seeds."""
        sets = []
        original_seed = self.seed
        for seed in seeds:
            self.seed = seed
            self.rng = np.random.RandomState(seed)
            s = self.generate(manifest, k, label_balanced)
            sets.append(s)
        self.seed = original_seed
        self.rng = np.random.RandomState(original_seed)
        return sets

    def save(self, support_set: SupportSet, output_path: str) -> None:
        """Save support set to CSV."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df = support_set.to_dataframe()

        # Add metadata as comment
        metadata_lines = [
            f"# k: {support_set.k}",
            f"# seed: {support_set.seed}",
            f"# label_balanced: {support_set.label_balanced}",
        ]

        with open(output_path, "w") as f:
            for line in metadata_lines:
                f.write(line + "\n")
            df.to_csv(f, index=False)

    def load(self, input_path: str) -> SupportSet:
        """Load support set from CSV."""
        # Read metadata
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

        # Read data
        df = pd.read_csv(input_path, comment="#")

        return SupportSet.from_dataframe(
            df,
            k=int(metadata.get("k", "50")),
            seed=int(metadata.get("seed", "42")),
            label_balanced=metadata.get("label_balanced", "True") == "True",
        )
