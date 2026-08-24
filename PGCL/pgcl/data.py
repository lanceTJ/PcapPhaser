from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from torch.utils.data import Dataset, DataLoader


def load_csvs(paths: Sequence[str]) -> pd.DataFrame:
    """
    paths: list of csv files or directories containing .csv.
    """
    df_all = []
    for p in paths:
        if os.path.isdir(p):
            files = [os.path.join(p, f) for f in os.listdir(p) if f.endswith(".csv")]
            files.sort()
            for f in files:
                df_all.append(pd.read_csv(f))
        else:
            df_all.append(pd.read_csv(p))
    if not df_all:
        raise FileNotFoundError("No CSV files found from provided paths.")
    return pd.concat(df_all, ignore_index=True)


def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # mimic notebook: replace space/slash and other illegal chars
    df = df.copy()
    df.columns = df.columns.str.replace(r"[ /]", "_", regex=True)
    df.columns = df.columns.str.replace(r"[^a-zA-Z0-9_]", "_", regex=True)
    df.columns = df.columns.str.replace(r"_+", "_", regex=True)
    df.columns = df.columns.str.strip("_")
    return df


def drop_bad_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols_to_drop = [c for c in df.columns if "idle" in c.lower()]
    cols_to_drop += [c for c in df.columns if ("fwd_rst_flags" in c.lower() or "bwd_rst_flags" in c.lower())]
    cols_to_drop += [c for c in df.columns if ("ip" in c.lower() or "port" in c.lower())]
    if cols_to_drop:
        df = df.drop(columns=list(dict.fromkeys(cols_to_drop)), errors="ignore")
    return df


def replace_inf_and_drop_nan(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # drop all-nan columns
    df = df.dropna(axis=1, how="all")
    return df


def binarize_label(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """
    Notebook logic:
      Benign -> 'Benign'
      else   -> 'Other'
    """
    df = df.copy()
    if label_col not in df.columns:
        raise KeyError(f"label_col='{label_col}' not found in DataFrame.")
    df[label_col] = df[label_col].apply(lambda x: "Benign" if str(x).lower() == "benign" else "Other")
    return df


def drop_object_columns(df: pd.DataFrame, label_col: str, keep_cols: Optional[List[str]] = None) -> pd.DataFrame:
    keep_cols = keep_cols or []
    df = df.copy()
    object_columns = df.dtypes[df.dtypes == "object"].index.tolist()
    cols_to_drop = [c for c in object_columns if (c != label_col and c not in keep_cols)]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop, errors="ignore")
    return df


def handle_remaining_nan(df: pd.DataFrame, nan_row_ratio_threshold: float = 0.1) -> pd.DataFrame:
    """
    Notebook logic:
      If total NaN exists:
        If fraction of rows containing NaN < 10% -> drop those rows
        Else -> drop columns with NaN
    """
    df = df.copy()
    if df.isnull().sum().sum() == 0:
        return df
    total_rows = df.shape[0]
    nan_rows = df.isnull().any(axis=1).sum()
    if total_rows > 0 and (nan_rows / total_rows) < nan_row_ratio_threshold:
        df = df.dropna(axis=0)
    else:
        df = df.dropna(axis=1)
    return df


def drop_high_corr_features_by_phase(
    df: pd.DataFrame,
    label_col: str,
    threshold: float = 0.90,
    phase_prefix_pattern: str = r"^p(\d+)_",
    phase_suffix_pattern: str = r"_p(\d+)$",
) -> pd.DataFrame:
    """
    Re-implements the notebook's phase-wise high-correlation pruning:
      - Detect phases via prefix pN_ OR suffix _pN
      - Within each phase, compute abs(corr) and mark base features with corr>threshold
      - Only drop base features that are high-corr in ALL phases (global intersection)
    If no phase columns found, do nothing.
    """
    df = df.copy()
    cols = [c for c in df.columns if c != label_col]
    num_df = df[cols].select_dtypes(include=[np.number])
    if num_df.empty:
        return df

    prefix_re = re.compile(phase_prefix_pattern)
    suffix_re = re.compile(phase_suffix_pattern)

    phase_to_cols: Dict[int, List[str]] = {}
    for c in num_df.columns:
        m = prefix_re.match(c)
        if m:
            phase_to_cols.setdefault(int(m.group(1)), []).append(c)
            continue
        m = suffix_re.search(c)
        if m:
            phase_to_cols.setdefault(int(m.group(1)), []).append(c)

    if not phase_to_cols:
        return df

    # Compute base feature names for each phase
    phase_high_corr_bases: Dict[int, set] = {}
    for p, p_cols in phase_to_cols.items():
        if len(p_cols) < 2:
            phase_high_corr_bases[p] = set()
            continue
        corr = num_df[p_cols].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        high_pairs = np.where(upper > threshold)
        bases = set()
        for i, j in zip(*high_pairs):
            c1 = upper.index[i]
            c2 = upper.columns[j]
            # strip phase prefix/suffix to base
            b1 = prefix_re.sub("", c1)
            b1 = suffix_re.sub("", b1)
            b2 = prefix_re.sub("", c2)
            b2 = suffix_re.sub("", b2)
            bases.add(b1)
            bases.add(b2)
        phase_high_corr_bases[p] = bases

    phases_sorted = sorted(phase_to_cols.keys())
    global_bases = None
    for p in phases_sorted:
        if global_bases is None:
            global_bases = set(phase_high_corr_bases[p])
        else:
            global_bases &= set(phase_high_corr_bases[p])
    global_bases = global_bases or set()
    if not global_bases:
        return df

    # drop across phases (support both naming patterns)
    to_drop = []
    for p in phases_sorted:
        for base in global_bases:
            to_drop.append(f"p{p}_{base}")
            to_drop.append(f"{base}_p{p}")
    df = df.drop(columns=[c for c in to_drop if c in df.columns], errors="ignore")
    return df


def balanced_upsample(df: pd.DataFrame, label_col: str, random_state: int = 42, max_samples: int = 400) -> pd.DataFrame:
    majority_count = df[label_col].value_counts().max()
    sample_count = min(majority_count, max_samples)
    dfs = []
    for label in df[label_col].unique():
        class_df = df[df[label_col] == label]
        class_df_upsampled = resample(
            class_df,
            replace=True,
            n_samples=sample_count,
            random_state=random_state,
        )
        dfs.append(class_df_upsampled)
    return pd.concat(dfs, ignore_index=True)


class PhaseFlowDataset(Dataset):
    """
    PhaseFlowDataset:
    - auto-detect max phase K from column names (pN_ prefix or _pN suffix)
    - if none found => K=1, all numeric feature columns are phase 1
    - auto-infer feats_per_phase (must be equal across phases)
    - optional standardization (fit scaler when train_mode=True)
    """
    def __init__(
        self,
        df: pd.DataFrame,
        label_col: str = "Label",
        phase_prefix_pattern: str = r"^p(\d+)_",
        phase_suffix_pattern: str = r"_p(\d+)$",
        train_mode: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.label_col = label_col
        self.scaler: Optional[StandardScaler] = StandardScaler() if train_mode else None

        prefix_re = re.compile(phase_prefix_pattern)
        suffix_re = re.compile(phase_suffix_pattern)

        feature_cols = [c for c in self.df.columns if c != label_col]
        feature_cols = [c for c in feature_cols if c not in ["Flow ID", "src_ip", "dst_ip", "src_port", "dst_port", "t_start"]]
        num_cols = self.df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

        phase_columns: Dict[int, List[str]] = {}
        for c in num_cols:
            m = prefix_re.match(c)
            if m:
                phase_columns.setdefault(int(m.group(1)), []).append(c)
                continue
            m = suffix_re.search(c)
            if m:
                phase_columns.setdefault(int(m.group(1)), []).append(c)

        if not phase_columns:
            self.K = 1
            phase_columns = {1: num_cols}
        else:
            self.K = max(phase_columns.keys())

        # ensure phases 1..K exist (missing phases => empty)
        for p in range(1, self.K + 1):
            phase_columns.setdefault(p, [])
            phase_columns[p].sort()

        # infer feats_per_phase
        lens = [len(phase_columns[p]) for p in range(1, self.K + 1)]
        nonzero = [l for l in lens if l > 0]
        if not nonzero:
            raise ValueError("No numeric phase features found after preprocessing.")
        self.feats_per_phase = nonzero[0]
        for l in nonzero:
            if l != self.feats_per_phase:
                raise ValueError(f"feats_per_phase not consistent across phases: {lens}")

        self.phase_feature_names: Dict[int, List[str]] = {p: phase_columns[p] for p in range(1, self.K + 1)}

        # build phase data: (N, K, F)
        phase_data = []
        for p in range(1, self.K + 1):
            cols_p = self.phase_feature_names[p]
            if not cols_p:
                # pad missing phase with zeros
                phase_data.append(np.zeros((len(self.df), self.feats_per_phase), dtype=np.float32))
            else:
                arr = self.df[cols_p].to_numpy(dtype=np.float32, copy=True)
                phase_data.append(arr)

        X = np.stack(phase_data, axis=1)  # (N, K, F)

        # Standardize on flattened features (match notebook)
        if self.scaler is not None:
            flat = X.reshape(len(self.df), -1)
            self.scaler.fit(flat)
            flat = self.scaler.transform(flat).astype(np.float32, copy=False)
            X = flat.reshape(len(self.df), self.K, self.feats_per_phase)

        self.phase_data = X

        if self.label_col in self.df.columns:
            self.labels = self.df[self.label_col].astype(str).tolist()
        else:
            self.labels = None

    def __len__(self) -> int:
        return len(self.phase_data)

    def __getitem__(self, idx: int):
        V = torch.from_numpy(self.phase_data[idx])  # (K, F)
        if self.labels is None:
            return V, None
        return V, self.labels[idx]

    def get_K(self) -> int:
        return self.K

    def get_feats_per_phase(self) -> int:
        return self.feats_per_phase

    def get_scaler(self) -> Optional[StandardScaler]:
        return self.scaler

    def get_class_idx_mapping(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        if self.labels is None:
            return {}, {}
        classes = sorted(list(set(self.labels)))
        class_to_idx = {c: i for i, c in enumerate(classes)}
        idx_to_class = {i: c for c, i in class_to_idx.items()}
        return class_to_idx, idx_to_class


def encode_labels(labels: List[str], class_to_idx: Dict[str, int]) -> np.ndarray:
    return np.array([class_to_idx[l] for l in labels], dtype=np.int64)


class _LabelWrapper(Dataset):
    """
    Wrap PhaseFlowDataset that returns string labels and convert to int tensor.
    """
    def __init__(self, base: PhaseFlowDataset, class_to_idx: Dict[str, int]):
        self.base = base
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        V, y = self.base[idx]
        if y is None:
            return V, None
        return V, torch.tensor(self.class_to_idx[y], dtype=torch.long)


def make_dataloaders(
    df: pd.DataFrame,
    label_col: str,
    test_size: float,
    stratify: bool,
    seed: int,
    upsample: bool,
    max_samples_per_class: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
) -> Tuple[DataLoader, DataLoader, Dict[str, int], Dict[int, str], PhaseFlowDataset, PhaseFlowDataset]:
    strat = df[label_col] if stratify and (label_col in df.columns) else None
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=seed, stratify=strat)

    if upsample:
        train_df = balanced_upsample(train_df, label_col=label_col, random_state=seed, max_samples=max_samples_per_class)

    train_dataset = PhaseFlowDataset(train_df, label_col=label_col, train_mode=True)
    val_dataset = PhaseFlowDataset(val_df, label_col=label_col, train_mode=False)

    # apply train scaler to val (match notebook)
    scaler = train_dataset.get_scaler()
    if scaler is not None:
        flat = val_dataset.phase_data.reshape(len(val_dataset), -1)
        flat = scaler.transform(flat).astype(np.float32, copy=False)
        val_dataset.phase_data = flat.reshape(len(val_dataset), val_dataset.K, val_dataset.feats_per_phase)

    class_to_idx, idx_to_class = train_dataset.get_class_idx_mapping()

    train_wrapped = _LabelWrapper(train_dataset, class_to_idx)
    val_wrapped = _LabelWrapper(val_dataset, class_to_idx)

    train_loader = DataLoader(
        train_wrapped,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_wrapped,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    return train_loader, val_loader, class_to_idx, idx_to_class, train_dataset, val_dataset


def split_train_test(
    df: pd.DataFrame,
    label_col: str,
    test_size: float = 0.2,
    stratify: bool = True,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataframe into train_df and test_df.
    Test set is fixed and never participates in k-fold.
    """
    strat = df[label_col] if stratify and (label_col in df.columns) else None
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed, stratify=strat)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def make_kfold_dataloaders(
    train_df: pd.DataFrame,
    label_col: str,
    n_splits: int,
    fold_idx: int,
    stratify: bool = True,
    seed: int = 42,
    upsample: bool = True,
    max_samples_per_class: int = 400,
    batch_size: int = 512,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
) -> Tuple[DataLoader, DataLoader, Dict[str, int], Dict[int, str], PhaseFlowDataset, PhaseFlowDataset]:
    """
    Create train/val loaders for a specific fold from train_df using StratifiedKFold.

    Args:
        train_df: The training dataframe (test already removed)
        label_col: Name of the label column
        n_splits: Number of folds (k)
        fold_idx: Current fold index (0 to n_splits-1)
        stratify: Whether to stratify by label
        seed: Random seed for fold shuffling
        upsample: Whether to upsample minority classes in training fold
        max_samples_per_class: Max samples per class after upsampling
        batch_size: Batch size for both loaders
        num_workers: Number of DataLoader workers
        pin_memory: Whether to pin memory
        persistent_workers: Whether to use persistent workers

    Returns:
        train_loader, val_loader, class_to_idx, idx_to_class, train_dataset, val_dataset
    """
    from sklearn.model_selection import StratifiedKFold

    if fold_idx < 0 or fold_idx >= n_splits:
        raise ValueError(f"fold_idx must be in [0, {n_splits}), got {fold_idx}")

    # Use StratifiedKFold for balanced folds
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    y = train_df[label_col].values if stratify and (label_col in train_df.columns) else None
    if y is None:
        raise ValueError("StratifiedKFold requires labels. Set stratify=True or use non-stratified splitting.")

    splits = list(skf.split(train_df, y))
    train_indices, val_indices = splits[fold_idx]

    fold_train_df = train_df.iloc[train_indices].copy().reset_index(drop=True)
    fold_val_df = train_df.iloc[val_indices].copy().reset_index(drop=True)

    if upsample:
        fold_train_df = balanced_upsample(
            fold_train_df,
            label_col=label_col,
            random_state=seed + fold_idx,
            max_samples=max_samples_per_class,
        )

    # Build datasets (train_mode=True for train to fit scaler)
    train_dataset = PhaseFlowDataset(fold_train_df, label_col=label_col, train_mode=True)
    val_dataset = PhaseFlowDataset(fold_val_df, label_col=label_col, train_mode=False)

    # Apply train scaler to val
    scaler = train_dataset.get_scaler()
    if scaler is not None:
        flat = val_dataset.phase_data.reshape(len(val_dataset), -1)
        flat = scaler.transform(flat).astype(np.float32, copy=False)
        val_dataset.phase_data = flat.reshape(len(val_dataset), val_dataset.K, val_dataset.feats_per_phase)

    class_to_idx, idx_to_class = train_dataset.get_class_idx_mapping()

    train_wrapped = _LabelWrapper(train_dataset, class_to_idx)
    val_wrapped = _LabelWrapper(val_dataset, class_to_idx)

    train_loader = DataLoader(
        train_wrapped,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_wrapped,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    return train_loader, val_loader, class_to_idx, idx_to_class, train_dataset, val_dataset


def make_dataloaders_from_train_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_col: str,
    upsample: bool = True,
    max_samples_per_class: int = 400,
    batch_size: int = 512,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
) -> Tuple[DataLoader, DataLoader, Dict[str, int], Dict[int, str], PhaseFlowDataset, PhaseFlowDataset]:
    """
    Create train and test loaders from pre-split dataframes.
    Used for final retraining on full train set after k-fold hyperparameter selection.

    Returns:
        train_loader, test_loader, class_to_idx, idx_to_class, train_dataset, test_dataset
    """
    if upsample:
        train_df = balanced_upsample(
            train_df,
            label_col=label_col,
            random_state=42,
            max_samples=max_samples_per_class,
        )

    train_dataset = PhaseFlowDataset(train_df, label_col=label_col, train_mode=True)
    test_dataset = PhaseFlowDataset(test_df, label_col=label_col, train_mode=False)

    # Apply train scaler to test
    scaler = train_dataset.get_scaler()
    if scaler is not None:
        flat = test_dataset.phase_data.reshape(len(test_dataset), -1)
        flat = scaler.transform(flat).astype(np.float32, copy=False)
        test_dataset.phase_data = flat.reshape(len(test_dataset), test_dataset.K, test_dataset.feats_per_phase)

    class_to_idx, idx_to_class = train_dataset.get_class_idx_mapping()

    train_wrapped = _LabelWrapper(train_dataset, class_to_idx)
    test_wrapped = _LabelWrapper(test_dataset, class_to_idx)

    train_loader = DataLoader(
        train_wrapped,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    test_loader = DataLoader(
        test_wrapped,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    return train_loader, test_loader, class_to_idx, idx_to_class, train_dataset, test_dataset


def make_dataloaders_from_split_csvs(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_col: str,
    upsample: bool = True,
    max_samples_per_class: int = 400,
    batch_size: int = 512,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int], Dict[int, str], PhaseFlowDataset, PhaseFlowDataset, PhaseFlowDataset]:
    """
    Create train/val/test loaders from pre-split dataframes.
    Used when splits are provided externally (e.g., from FlowManifest).

    Returns:
        train_loader, val_loader, test_loader, class_to_idx, idx_to_class,
        train_dataset, val_dataset, test_dataset
    """
    if upsample:
        train_df = balanced_upsample(
            train_df,
            label_col=label_col,
            random_state=42,
            max_samples=max_samples_per_class,
        )

    train_dataset = PhaseFlowDataset(train_df, label_col=label_col, train_mode=True)
    val_dataset = PhaseFlowDataset(val_df, label_col=label_col, train_mode=False)
    test_dataset = PhaseFlowDataset(test_df, label_col=label_col, train_mode=False)

    # Apply train scaler to val and test
    scaler = train_dataset.get_scaler()
    if scaler is not None:
        for ds in (val_dataset, test_dataset):
            flat = ds.phase_data.reshape(len(ds), -1)
            flat = scaler.transform(flat).astype(np.float32, copy=False)
            ds.phase_data = flat.reshape(len(ds), ds.K, ds.feats_per_phase)

    class_to_idx, idx_to_class = train_dataset.get_class_idx_mapping()

    train_wrapped = _LabelWrapper(train_dataset, class_to_idx)
    val_wrapped = _LabelWrapper(val_dataset, class_to_idx)
    test_wrapped = _LabelWrapper(test_dataset, class_to_idx)

    train_loader = DataLoader(
        train_wrapped,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_wrapped,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    test_loader = DataLoader(
        test_wrapped,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    return train_loader, val_loader, test_loader, class_to_idx, idx_to_class, train_dataset, val_dataset, test_dataset
