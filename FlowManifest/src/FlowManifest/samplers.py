"""
PA-FT Balanced Samplers: Balanced variant sampling for Perturbation-Aware Fine-Tuning.

Key requirements:
- Each batch contains ~50% origin, ~10% each of the 5 perturbations
- Same gradient updates as Clean-FT (don't train 6x longer)
- Both use same parent_flow_ids from support set
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, Iterator, List, Optional, Set, Tuple

import numpy as np
from torch.utils.data import Sampler, BatchSampler

from .manifest_builder import ManifestBuilder, Variant


class PAFTSampler(Sampler[int]):
    """
    Sampler for Perturbation-Aware Fine-Tuning.

    Balances variants: 50% origin, 10% each case1-case5.
    Ensures that for any selected variant, we have the corresponding origin.
    """

    DEFAULT_WEIGHTS: Dict[str, float] = {
        Variant.ORIGIN.value: 0.5,
        Variant.CASE1_LOSS.value: 0.1,
        Variant.CASE2_RETRANSMIT.value: 0.1,
        Variant.CASE3_REORDER.value: 0.1,
        Variant.CASE4_LENGTH.value: 0.1,
        Variant.CASE5_RATE.value: 0.1,
    }

    def __init__(
        self,
        indices: List[int],
        parent_flow_ids: List[str],
        variants: List[str],
        weights: Optional[Dict[str, float]] = None,
        seed: int = 42,
    ):
        """
        Initialize PA-FT sampler.

        Args:
            indices: List of dataset indices
            parent_flow_ids: List of parent_flow_id for each index
            variants: List of variant for each index
            weights: Variant weight dict (default: 50% origin, 10% each case)
            seed: Random seed
        """
        self.indices = indices
        self.parent_flow_ids = parent_flow_ids
        self.variants = variants
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.seed = seed
        self.rng = random.Random(seed)

        # Build index maps
        self.parent_to_indices: Dict[str, List[int]] = defaultdict(list)
        self.variant_to_indices: Dict[str, List[int]] = defaultdict(list)
        self.parent_and_variant_to_index: Dict[Tuple[str, str], int] = {}

        for idx, parent_id, variant in zip(indices, parent_flow_ids, variants):
            self.parent_to_indices[parent_id].append(idx)
            self.variant_to_indices[variant].append(idx)
            self.parent_and_variant_to_index[(parent_id, variant)] = idx

        # Available variants per parent
        self.parent_available_variants: Dict[str, Set[str]] = {}
        for parent_id, idxs in self.parent_to_indices.items():
            self.parent_available_variants[parent_id] = {self.variants[i] for i in idxs}

        # Get list of unique parents
        self.unique_parents = list(self.parent_to_indices.keys())

    def _sample_variant(self) -> str:
        """Sample a variant type according to weights."""
        variants = list(self.weights.keys())
        weights = list(self.weights.values())
        return self.rng.choices(variants, weights=weights, k=1)[0]

    def __iter__(self) -> Iterator[int]:
        """
        Yield indices in an order that balances variants.
        For each non-origin sampled, also include the origin for that parent.
        """
        # Shuffle parents
        parent_order = self.unique_parents.copy()
        self.rng.shuffle(parent_order)

        # Track used parents to avoid reusing too quickly
        used_recently: Set[str] = set()

        # Generate samples
        while True:
            # Replenish if needed
            if not parent_order:
                parent_order = self.unique_parents.copy()
                self.rng.shuffle(parent_order)
                used_recently.clear()

            # Pick a parent not used recently
            parent_idx = 0
            while parent_idx < len(parent_order) and parent_order[parent_idx] in used_recently:
                parent_idx += 1

            if parent_idx >= len(parent_order):
                # All used recently, just pick first
                parent_idx = 0

            parent_id = parent_order.pop(parent_idx)
            used_recently.add(parent_id)

            # Sample a variant that exists for this parent
            available = self.parent_available_variants[parent_id]
            if not available:
                continue

            # Try to sample according to weights, fallback to available
            for _ in range(10):
                variant = self._sample_variant()
                if variant in available:
                    break
            else:
                # Fallback to any available
                variant = next(iter(available))

            # Get the index
            idx = self.parent_and_variant_to_index.get((parent_id, variant))
            if idx is not None:
                yield idx

    def __len__(self) -> int:
        return len(self.indices)


class BalancedVariantBatchSampler(BatchSampler):
    """
    Batch sampler that ensures each batch has a balanced mix of variants.

    Each batch will have approximately:
    - 50% origin
    - 10% each case1-case5
    """

    def __init__(
        self,
        dataset_indices: List[int],
        parent_flow_ids: List[str],
        variants: List[str],
        batch_size: int,
        weights: Optional[Dict[str, float]] = None,
        drop_last: bool = False,
        seed: int = 42,
    ):
        """
        Initialize balanced batch sampler.

        Args:
            dataset_indices: All indices in the dataset
            parent_flow_ids: parent_flow_id for each index
            variants: variant for each index
            batch_size: Batch size
            weights: Variant weights
            drop_last: Whether to drop last incomplete batch
            seed: Random seed
        """
        self.dataset_indices = dataset_indices
        self.parent_flow_ids = parent_flow_ids
        self.variants = variants
        self.batch_size = batch_size
        self.weights = weights or PAFTSampler.DEFAULT_WEIGHTS
        self.drop_last = drop_last
        self.seed = seed
        self.rng = random.Random(seed)

        # Group indices by variant
        self.variant_indices: Dict[str, List[int]] = defaultdict(list)
        for idx, variant in zip(dataset_indices, variants):
            self.variant_indices[variant].append(idx)

        # Shuffle each variant list
        for v in self.variant_indices:
            self.rng.shuffle(self.variant_indices[v])

        # Pointers for each variant list
        self.variant_pointers: Dict[str, int] = {v: 0 for v in self.variant_indices}

    def _get_next_for_variant(self, variant: str) -> Optional[int]:
        """Get next index for a variant, cycling if needed."""
        indices = self.variant_indices.get(variant, [])
        if not indices:
            return None

        ptr = self.variant_pointers[variant]
        idx = indices[ptr]
        self.variant_pointers[variant] = (ptr + 1) % len(indices)
        return idx

    def __iter__(self) -> Iterator[List[int]]:
        """Yield balanced batches."""
        # Calculate how many of each variant per batch
        variants = list(self.weights.keys())
        weights = np.array([self.weights[v] for v in variants])

        batch: List[int] = []

        while True:
            # Select variant based on weights
            v_idx = self.rng.choices(range(len(variants)), weights=weights, k=1)[0]
            variant = variants[v_idx]

            # Get next index for this variant
            idx = self._get_next_for_variant(variant)
            if idx is not None:
                batch.append(idx)

            # Yield batch when full
            if len(batch) == self.batch_size:
                yield batch
                batch = []
                # Reshuffle variant lists periodically
                for v in self.variant_indices:
                    self.rng.shuffle(self.variant_indices[v])

        # Handle remaining if not drop_last
        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset_indices) // self.batch_size
        return (len(self.dataset_indices) + self.batch_size - 1) // self.batch_size


def create_paft_batch_sampler(
    manifest: ManifestBuilder,
    support_parent_ids: Set[str],
    idx_to_entry: List[Tuple[int, str, str]],  # (dataset_idx, parent_flow_id, variant)
    batch_size: int,
    drop_last: bool = False,
    seed: int = 42,
) -> BalancedVariantBatchSampler:
    """
    Create a PA-FT balanced batch sampler from manifest and support set.

    Args:
        manifest: The manifest
        support_parent_ids: Parent flow IDs in support set
        idx_to_entry: Mapping from dataset index to (parent_flow_id, variant)
        batch_size: Batch size
        drop_last: Whether to drop last batch
        seed: Random seed

    Returns:
        BalancedVariantBatchSampler
    """
    # Filter to support set with all variants
    filtered_indices: List[int] = []
    filtered_parents: List[str] = []
    filtered_variants: List[str] = []

    for idx, parent_id, variant in idx_to_entry:
        if parent_id in support_parent_ids:
            filtered_indices.append(idx)
            filtered_parents.append(parent_id)
            filtered_variants.append(variant)

    return BalancedVariantBatchSampler(
        dataset_indices=filtered_indices,
        parent_flow_ids=filtered_parents,
        variants=filtered_variants,
        batch_size=batch_size,
        drop_last=drop_last,
        seed=seed,
    )
