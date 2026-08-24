import os
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from safetensors.torch import save_file


def set_seed(seed: int) -> None:
    """Best-effort determinism."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # CuDNN deterministic can hurt perf; keep it optional via env.
    if os.environ.get("CUDNN_DETERMINISTIC", "0") == "1":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_ckpt_atomic(
    path: str,
    encoder: torch.nn.Module,
    proj_head: torch.nn.Module,
    feature_cols: List[str],
    feat_scaler,
    config: Dict[str, Any],
    label_name: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save checkpoint as a single .safetensors file plus a sidecar json metadata.
    Keeps original notebook behavior (atomic write) but avoids partial files.
    """
    path = str(path)
    Path(os.path.dirname(path) or ".").mkdir(parents=True, exist_ok=True)

    state_dict = {**encoder.state_dict(), **{f"proj.{k}": v for k, v in proj_head.state_dict().items()}}

    feature_cols = list(map(str, feature_cols))
    assert hasattr(feat_scaler, "mean_") and hasattr(feat_scaler, "scale_"), "feat_scaler must be a fitted StandardScaler"

    meta = {
        "feature_cols": feature_cols,
        "scaler_mean": feat_scaler.mean_.tolist(),
        "scaler_scale": feat_scaler.scale_.tolist(),
        "config": config,
        "label_name": label_name,
        "extra": extra or {},
    }

    # Atomic write: write to temp then replace
    tmp_dir = Path(path).parent
    with tempfile.NamedTemporaryFile(dir=tmp_dir, delete=False, suffix=".safetensors") as tf:
        tmp_path = tf.name
    try:
        save_file(state_dict, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    meta_path = os.path.splitext(path)[0] + ".meta.json"
    with tempfile.NamedTemporaryFile(dir=tmp_dir, delete=False, suffix=".json", mode="w", encoding="utf-8") as tf:
        tmp_meta = tf.name
        json.dump(meta, tf, ensure_ascii=False, indent=2)
    os.replace(tmp_meta, meta_path)


def write_json_atomic(path: str, payload: Dict[str, Any]) -> None:
    """Atomically write a JSON result file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output_path.parent,
        delete=False,
        suffix=".json",
        mode="w",
        encoding="utf-8",
    ) as handle:
        temp_path = handle.name
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(temp_path, output_path)


def save_classifier_ckpt_atomic(
    path: str,
    model: torch.nn.Module,
    feature_cols: List[str],
    feat_scaler,
    config: Dict[str, Any],
    label_name: str,
    class_to_idx: Dict[str, int],
) -> None:
    """Save the restored best fine-tuned encoder and classification head."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
    }

    with tempfile.NamedTemporaryFile(
        dir=output_path.parent,
        delete=False,
        suffix=".safetensors",
    ) as handle:
        temp_path = handle.name
    try:
        save_file(state_dict, temp_path)
        os.replace(temp_path, output_path)
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass

    meta = {
        "checkpoint_type": "fine_tuned_classifier",
        "feature_cols": list(map(str, feature_cols)),
        "scaler_mean": feat_scaler.mean_.tolist(),
        "scaler_scale": feat_scaler.scale_.tolist(),
        "config": config,
        "label_name": label_name,
        "class_to_idx": class_to_idx,
    }
    write_json_atomic(str(output_path.with_suffix(".meta.json")), meta)
