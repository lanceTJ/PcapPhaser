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
