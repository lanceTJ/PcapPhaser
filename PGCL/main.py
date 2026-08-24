from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

from torch.utils.data import DataLoader

from pgcl.data import (
    load_csvs,
    sanitize_columns,
    drop_bad_columns,
    replace_inf_and_drop_nan,
    drop_high_corr_features_by_phase,
    binarize_label,
    drop_object_columns,
    handle_remaining_nan,
    make_dataloaders,
    split_train_test,
    make_kfold_dataloaders,
    make_dataloaders_from_train_test,
    make_dataloaders_from_split_csvs,
    PhaseFlowDataset,
    _LabelWrapper,
)
from pgcl.model import Encoder, ProjectionHead, DownstreamClassifier
from pgcl.train import pretrain_pgcl, fine_tune_two_stage, evaluate_downstream_metrics
from pgcl.utils import (
    save_ckpt_atomic,
    save_classifier_ckpt_atomic,
    set_seed,
    write_json_atomic,
)


def load_train_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith((".yaml", ".yml")):
            return yaml.safe_load(f)
        raise ValueError("Only YAML config is supported in this template.")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("PGCL phase training (notebook -> python refactor)")

    # config
    p.add_argument("--train-config", type=str, required=True, help="Path to YAML train config.")
    p.add_argument("--seed", type=int, default=None, help="Override config seed.")

    # data
    p.add_argument("--csv-paths", type=str, nargs="+", default=None, help="CSV file(s) or directory(ies) containing CSVs.")
    p.add_argument("--train-csv", type=str, default=None, help="Pre-split train CSV (external split mode).")
    p.add_argument("--val-csv", type=str, default=None, help="Pre-split val CSV (external split mode).")
    p.add_argument("--test-csv", type=str, default=None, help="Pre-split test CSV (external split mode).")
    p.add_argument("--label-col", type=str, default=None, help="Override config.data.label_col.")
    p.add_argument("--no-binarize", action="store_true", help="Disable Benign/Other binarization.")
    p.add_argument("--no-upsampling", action="store_true", help="Disable balanced upsampling for training split.")
    p.add_argument("--corr-threshold", type=float, default=None, help="Override config.data.corr_threshold.")

    # output
    p.add_argument("--output-dir", type=str, required=True, help="Output directory for logs and checkpoints.")
    p.add_argument("--run-name", type=str, default="run", help="Used in output filenames.")
    p.add_argument("--ckpt-name", type=str, default="best_pgcl_phase.safetensors", help="Checkpoint filename.")
    p.add_argument("--log-name", type=str, default="train_pgcl_phase.csv", help="Training log csv filename.")

    # device
    p.add_argument("--device", type=str, default=None, help="e.g. cuda, cuda:0, cpu. Default: auto.")

    # k-fold cross-validation
    p.add_argument("--k-fold", type=int, default=None, help="Enable k-fold CV with K folds. Default: None (use fixed train/val split)")
    p.add_argument("--k-fold-seed", type=int, default=None, help="Seed for k-fold fold shuffling. Default: use main seed")

    # finetune
    p.add_argument("--skip-finetune", action="store_true", help="Skip downstream finetuning stage.")
    return p


def _run_pretrain(
    encoder: Encoder,
    proj_head: ProjectionHead,
    train_loader,
    val_loader,
    device: torch.device,
    K: int,
    pcfg: Dict[str, Any],
    log_csv_path: str,
    ckpt_path: str,
    feat_scaler,
    feature_cols: list,
    class_to_idx: Dict[str, int],
    feats_per_phase: int,
    encoder_out_dim: int,
    proj_out_dim: int,
    dropout: float,
    label_col: str,
) -> None:
    """Run pretrain stage."""

    def _on_best(encoder_, proj_head_):
        save_ckpt_atomic(
            ckpt_path,
            encoder=encoder_,
            proj_head=proj_head_,
            feature_cols=feature_cols,
            feat_scaler=feat_scaler,
            config={
                "feats_per_phase": feats_per_phase,
                "K": K,
                "encoder_out_dim": encoder_out_dim,
                "proj_out_dim": proj_out_dim,
                "dropout": dropout,
            },
            label_name=label_col,
            extra={"class_to_idx": class_to_idx},
        )

    pretrain_pgcl(
        encoder=encoder,
        proj_head=proj_head,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        K=K,
        epochs=int(pcfg.get("epochs", 50)),
        lr=float(pcfg.get("lr", 5e-4)),
        weight_decay=float(pcfg.get("weight_decay", 1e-4)),
        temperature=float(pcfg.get("temperature", 0.1)),
        noise_std=float(pcfg.get("noise_std", 0.05)),
        feat_drop_prob=float(pcfg.get("feat_drop_prob", 0.1)),
        patience=int(pcfg.get("patience", 6)),
        max_grad_norm=float(pcfg.get("max_grad_norm", 1.0)),
        use_amp=bool(pcfg.get("use_amp", True)),
        log_csv_path=log_csv_path,
        scheduler_t_max=int(pcfg.get("scheduler", {}).get("t_max", int(pcfg.get("epochs", 50)))),
        on_best_ckpt=_on_best,
    )


def _run_finetune(
    encoder: Encoder,
    train_loader,
    val_loader,
    device: torch.device,
    class_to_idx: Dict[str, int],
    encoder_out_dim: int,
    ft_cfg: Dict[str, Any],
    pcfg: Dict[str, Any],
) -> Tuple[float, DownstreamClassifier]:
    """Run finetune stage and return the best validation F1 and trained model."""
    num_classes = len(class_to_idx)
    classifier = DownstreamClassifier(
        encoder=encoder, num_classes=num_classes, encoder_out_dim=encoder_out_dim
    ).to(device)

    stage_a = ft_cfg.get("stage_a", {})
    stage_b = ft_cfg.get("stage_b", {})

    results = fine_tune_two_stage(
        train_loader=train_loader,
        val_loader=val_loader,
        model=classifier,
        device=device,
        stage_a_epochs=int(stage_a.get("epochs", 10)),
        stage_a_lr=float(stage_a.get("lr", 1e-3)),
        stage_a_weight_decay=float(stage_a.get("weight_decay", 1e-4)),
        stage_a_t_max=int(stage_a.get("t_max", 10)),
        stage_b_enabled=bool(stage_b.get("enabled", False)),
        stage_b_epochs=int(stage_b.get("epochs", 10)),
        stage_b_lr=float(stage_b.get("lr", 1e-5)),
        stage_b_weight_decay=float(stage_b.get("weight_decay", 1e-4)),
        stage_b_t_max=int(stage_b.get("t_max", 10)),
        use_amp=bool(pcfg.get("use_amp", True)),
        max_grad_norm=float(pcfg.get("max_grad_norm", 1.0)),
    )

    # fine_tune_two_stage restores the model state with the best validation F1.
    return float(results["best_val_f1"]), classifier


def _run_single_pipeline(
    train_loader,
    val_loader,
    device: torch.device,
    K: int,
    feats_per_phase: int,
    encoder_out_dim: int,
    proj_out_dim: int,
    dropout: float,
    feature_cols: list,
    feat_scaler,
    class_to_idx: Dict[str, int],
    label_col: str,
    pcfg: Dict[str, Any],
    ft_cfg: Dict[str, Any],
    log_csv_path: str,
    ckpt_path: str,
    skip_finetune: bool,
    test_loader=None,
) -> float:
    """Run one complete pretrain + finetune pipeline. Returns best val F1."""
    # Create fresh models
    encoder = Encoder(
        feats_per_phase=feats_per_phase,
        hidden_dim=encoder_out_dim,
        out_dim=encoder_out_dim,
        dropout=dropout,
    )
    proj_head = ProjectionHead(in_dim=encoder_out_dim, out_dim=proj_out_dim)

    _run_pretrain(
        encoder=encoder,
        proj_head=proj_head,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        K=K,
        pcfg=pcfg,
        log_csv_path=log_csv_path,
        ckpt_path=ckpt_path,
        feat_scaler=feat_scaler,
        feature_cols=feature_cols,
        class_to_idx=class_to_idx,
        feats_per_phase=feats_per_phase,
        encoder_out_dim=encoder_out_dim,
        proj_out_dim=proj_out_dim,
        dropout=dropout,
        label_col=label_col,
    )

    if skip_finetune or (not bool(ft_cfg.get("enabled", True))):
        return 0.0

    val_f1, classifier = _run_finetune(
        encoder=encoder,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        class_to_idx=class_to_idx,
        encoder_out_dim=encoder_out_dim,
        ft_cfg=ft_cfg,
        pcfg=pcfg,
    )

    val_acc, restored_val_f1, val_precision, val_recall = evaluate_downstream_metrics(
        classifier, val_loader, device
    )
    metrics = {
        "best_val_f1": val_f1,
        "validation": {
            "accuracy": val_acc,
            "macro_f1": restored_val_f1,
            "macro_precision": val_precision,
            "macro_recall": val_recall,
        },
        "test": None,
    }

    # Optional test evaluation (external split and final k-fold modes).
    if test_loader is not None:
        print(f"\n{'-'*60}")
        print("Evaluating on held-out test set...")
        print(f"{'-'*60}")
        acc, f1, p, r = evaluate_downstream_metrics(classifier, test_loader, device)
        print(f"Test  Acc: {acc:.4f}  F1: {f1:.4f}  P: {p:.4f}  R: {r:.4f}")
        metrics["test"] = {
            "accuracy": acc,
            "macro_f1": f1,
            "macro_precision": p,
            "macro_recall": r,
        }

    checkpoint_path = Path(ckpt_path)
    fine_tuned_path = checkpoint_path.with_name(
        f"{checkpoint_path.stem}_finetuned.safetensors"
    )
    metrics_path = checkpoint_path.with_name(
        f"{checkpoint_path.stem}_metrics.json"
    )
    save_classifier_ckpt_atomic(
        str(fine_tuned_path),
        model=classifier,
        feature_cols=feature_cols,
        feat_scaler=feat_scaler,
        config={
            "K": K,
            "feats_per_phase": feats_per_phase,
            "encoder_out_dim": encoder_out_dim,
            "proj_out_dim": proj_out_dim,
            "dropout": dropout,
            "pretrain": pcfg,
            "finetune": ft_cfg,
        },
        label_name=label_col,
        class_to_idx=class_to_idx,
    )
    write_json_atomic(str(metrics_path), metrics)
    print(f"Fine-tuned checkpoint saved to: {fine_tuned_path}")
    print(f"Evaluation metrics saved to: {metrics_path}")

    return val_f1


def _preprocess_df(df: pd.DataFrame, label_col: str, no_binarize: bool, corr_threshold: float) -> pd.DataFrame:
    """Apply standard preprocessing pipeline to a dataframe."""
    df = drop_bad_columns(df)
    df = replace_inf_and_drop_nan(df)
    df = sanitize_columns(df)
    df = drop_high_corr_features_by_phase(df, label_col=label_col, threshold=corr_threshold)
    if not no_binarize:
        df = binarize_label(df, label_col)
    df = handle_remaining_nan(df, nan_row_ratio_threshold=0.1)
    df = drop_object_columns(df, label_col=label_col, keep_cols=["Flow ID"])
    return df


def main() -> None:
    args = build_argparser().parse_args()

    cfg = load_train_config(args.train_config)

    # Resolve overrides
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 42))
    set_seed(seed)

    label_col = args.label_col if args.label_col is not None else cfg["data"].get("label_col", "Label")
    corr_threshold = args.corr_threshold if args.corr_threshold is not None else float(cfg["data"].get("corr_threshold", 0.90))

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine data loading mode
    use_external_split = args.train_csv is not None and args.val_csv is not None and args.test_csv is not None
    use_csv_paths = args.csv_paths is not None

    if not use_external_split and not use_csv_paths:
        raise ValueError("Must provide either --csv-paths OR --train-csv + --val-csv + --test-csv")

    # ---------- Config ----------
    dl_cfg = cfg.get("dataloader", {})
    data_cfg = cfg.get("data", {})
    pcfg = cfg.get("pretrain", {})
    ft_cfg = cfg.get("finetune", {})

    upsample = not args.no_upsampling
    max_samples_per_class = int(data_cfg.get("max_samples_per_class", 400))
    batch_size = int(dl_cfg.get("batch_size", 512))
    num_workers = int(dl_cfg.get("num_workers", 0))
    pin_memory = bool(dl_cfg.get("pin_memory", False))
    persistent_workers = bool(dl_cfg.get("persistent_workers", False))

    # ---------- Device ----------
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # ---------- Model Config ----------
    mcfg = cfg.get("model", {})
    encoder_out_dim = int(mcfg.get("encoder_out_dim", 256))
    proj_out_dim = int(mcfg.get("proj_out_dim", 128))
    dropout = float(mcfg.get("dropout", 0.1))

    if args.k_fold is not None:
        # ============================================================
        # K-Fold Cross-Validation Mode
        # ============================================================
        k = args.k_fold
        kfold_seed = args.k_fold_seed if args.k_fold_seed is not None else seed
        test_size = float(data_cfg.get("test_size", 0.2))
        stratify = bool(data_cfg.get("stratify", True))

        print(f"\n{'='*60}")
        print(f"K-Fold Cross-Validation: K={k}")
        print(f"  Fixed test size: {test_size}")
        print(f"  K-fold seed: {kfold_seed}")
        print(f"{'='*60}\n")

        if use_external_split:
            raise ValueError("External split mode (--train-csv/--val-csv/--test-csv) is not supported with k-fold. Use --csv-paths instead.")

        df = load_csvs(args.csv_paths)
        df = _preprocess_df(df, label_col, args.no_binarize, corr_threshold)

        # Step 1: Fixed train/test split
        train_df, test_df = split_train_test(
            df=df,
            label_col=label_col,
            test_size=test_size,
            stratify=stratify,
            seed=seed,
        )
        print(f"Train set: {len(train_df)} samples")
        print(f"Test set:  {len(test_df)} samples")

        # Step 2: K-fold cross validation on train_df
        fold_val_f1s = []
        for fold_idx in range(k):
            print(f"\n{'-'*60}")
            print(f"Fold {fold_idx + 1}/{k}")
            print(f"{'-'*60}")

            set_seed(kfold_seed + fold_idx)

            # Create fold-specific loaders
            fold_train_loader, fold_val_loader, class_to_idx, idx_to_class, fold_train_ds, fold_val_ds = make_kfold_dataloaders(
                train_df=train_df,
                label_col=label_col,
                n_splits=k,
                fold_idx=fold_idx,
                stratify=stratify,
                seed=kfold_seed,
                upsample=upsample,
                max_samples_per_class=max_samples_per_class,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
            )

            K = fold_train_ds.get_K()
            feats_per_phase = fold_train_ds.get_feats_per_phase()
            feat_scaler = fold_train_ds.get_scaler()
            feature_cols = sum([fold_train_ds.phase_feature_names[p] for p in range(1, K + 1)], [])

            log_csv_path = str(out_dir / f"{args.run_name}_fold{fold_idx}_{K}_{args.log_name}")
            ckpt_path = str(out_dir / f"{args.run_name}_fold{fold_idx}_{K}_{args.ckpt_name}")

            # Run complete pipeline for this fold
            val_f1 = _run_single_pipeline(
                train_loader=fold_train_loader,
                val_loader=fold_val_loader,
                device=device,
                K=K,
                feats_per_phase=feats_per_phase,
                encoder_out_dim=encoder_out_dim,
                proj_out_dim=proj_out_dim,
                dropout=dropout,
                feature_cols=feature_cols,
                feat_scaler=feat_scaler,
                class_to_idx=class_to_idx,
                label_col=label_col,
                pcfg=pcfg,
                ft_cfg=ft_cfg,
                log_csv_path=log_csv_path,
                ckpt_path=ckpt_path,
                skip_finetune=args.skip_finetune,
            )
            fold_val_f1s.append(val_f1)
            print(f"Fold {fold_idx + 1} val F1: {val_f1:.4f}")

        # Step 3: Report k-fold results
        avg_val_f1 = float(np.mean(fold_val_f1s))
        std_val_f1 = float(np.std(fold_val_f1s))
        print(f"\n{'='*60}")
        print(f"K-Fold Results (K={k})")
        print(f"  Average val F1: {avg_val_f1:.4f} (+/- {std_val_f1:.4f})")
        print(f"  Per-fold F1s: {[f'{f:.4f}' for f in fold_val_f1s]}")
        print(f"{'='*60}")

        # Step 4: Retrain final model on full train_df
        # Split train_df into final_train (90%) and final_val (10%) for early stopping.
        # test_df is reserved for final evaluation ONLY.
        print(f"\n{'-'*60}")
        print("Retraining final model on full train set...")
        print("  (train_df split: 90% train / 10% val for early stopping)")
        print(f"{'-'*60}")

        set_seed(seed)
        final_train_df, final_val_df = split_train_test(
            df=train_df,
            label_col=label_col,
            test_size=0.1,  # 10% of train_df for validation
            stratify=stratify,
            seed=seed,
        )

        final_train_loader, final_val_loader, class_to_idx, idx_to_class, final_train_ds, final_val_ds = make_dataloaders_from_train_test(
            train_df=final_train_df,
            test_df=final_val_df,
            label_col=label_col,
            upsample=upsample,
            max_samples_per_class=max_samples_per_class,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

        # Also create a test loader (no scaler fitting, apply train scaler)
        test_dataset = PhaseFlowDataset(test_df, label_col=label_col, train_mode=False)
        scaler = final_train_ds.get_scaler()
        if scaler is not None:
            flat = test_dataset.phase_data.reshape(len(test_dataset), -1)
            flat = scaler.transform(flat).astype(np.float32, copy=False)
            test_dataset.phase_data = flat.reshape(len(test_dataset), test_dataset.K, test_dataset.feats_per_phase)
        test_wrapped = _LabelWrapper(test_dataset, class_to_idx)
        test_loader = DataLoader(
            test_wrapped,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

        K = final_train_ds.get_K()
        feats_per_phase = final_train_ds.get_feats_per_phase()
        feat_scaler = final_train_ds.get_scaler()
        feature_cols = sum([final_train_ds.phase_feature_names[p] for p in range(1, K + 1)], [])

        log_csv_path = str(out_dir / f"{args.run_name}_final_{K}_{args.log_name}")
        ckpt_path = str(out_dir / f"{args.run_name}_final_{K}_{args.ckpt_name}")

        final_val_f1 = _run_single_pipeline(
            train_loader=final_train_loader,
            val_loader=final_val_loader,  # Use validation split for early stopping, NOT test
            device=device,
            K=K,
            feats_per_phase=feats_per_phase,
            encoder_out_dim=encoder_out_dim,
            proj_out_dim=proj_out_dim,
            dropout=dropout,
            feature_cols=feature_cols,
            feat_scaler=feat_scaler,
            class_to_idx=class_to_idx,
            label_col=label_col,
            pcfg=pcfg,
            ft_cfg=ft_cfg,
            log_csv_path=log_csv_path,
            ckpt_path=ckpt_path,
            skip_finetune=args.skip_finetune,
            test_loader=test_loader,
        )

        # Step 5: Report final validation and held-out test evaluation.
        # The test metrics are emitted by _run_single_pipeline using the restored
        # best-validation fine-tuned model.
        print(f"\n{'='*60}")
        print("Final Model Summary")
        print(f"{'='*60}")
        print(f"K-fold average val F1: {avg_val_f1:.4f} (+/- {std_val_f1:.4f})")
        print(f"Final val F1 (early stopping): {final_val_f1:.4f}")
        print(f"{'='*60}\n")

    else:
        # ============================================================
        # Standard Mode: Fixed train/val/test split
        # Supports both internal split (--csv-paths) and external split
        # (--train-csv/--val-csv/--test-csv, e.g. from FlowManifest)
        # ============================================================
        if use_external_split:
            print(f"\n{'='*60}")
            print("External Split Mode (FlowManifest-compatible)")
            print(f"  Train: {args.train_csv}")
            print(f"  Val:   {args.val_csv}")
            print(f"  Test:  {args.test_csv}")
            print(f"{'='*60}\n")

            train_df = pd.read_csv(args.train_csv)
            val_df = pd.read_csv(args.val_csv)
            test_df = pd.read_csv(args.test_csv)

            # Apply identical preprocessing to all three splits
            train_df = _preprocess_df(train_df, label_col, args.no_binarize, corr_threshold)
            val_df = _preprocess_df(val_df, label_col, args.no_binarize, corr_threshold)
            test_df = _preprocess_df(test_df, label_col, args.no_binarize, corr_threshold)

            train_loader, val_loader, test_loader, class_to_idx, idx_to_class, train_dataset, val_dataset, test_dataset = make_dataloaders_from_split_csvs(
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                label_col=label_col,
                upsample=upsample,
                max_samples_per_class=max_samples_per_class,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
            )

            K = train_dataset.get_K()
            feats_per_phase = train_dataset.get_feats_per_phase()
            feat_scaler = train_dataset.get_scaler()
            feature_cols = sum([train_dataset.phase_feature_names[p] for p in range(1, K + 1)], [])

            log_csv_path = str(out_dir / f"{args.run_name}_{K}_{args.log_name}")
            ckpt_path = str(out_dir / f"{args.run_name}_{K}_{args.ckpt_name}")

            _run_single_pipeline(
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                K=K,
                feats_per_phase=feats_per_phase,
                encoder_out_dim=encoder_out_dim,
                proj_out_dim=proj_out_dim,
                dropout=dropout,
                feature_cols=feature_cols,
                feat_scaler=feat_scaler,
                class_to_idx=class_to_idx,
                label_col=label_col,
                pcfg=pcfg,
                ft_cfg=ft_cfg,
                log_csv_path=log_csv_path,
                ckpt_path=ckpt_path,
                skip_finetune=args.skip_finetune,
                test_loader=test_loader,
            )
        else:
            # Internal split mode (original behavior)
            df = load_csvs(args.csv_paths)
            df = _preprocess_df(df, label_col, args.no_binarize, corr_threshold)

            train_loader, val_loader, class_to_idx, idx_to_class, train_dataset, _ = make_dataloaders(
                df=df,
                label_col=label_col,
                test_size=float(data_cfg.get("test_size", 0.2)),
                stratify=bool(data_cfg.get("stratify", True)),
                seed=seed,
                upsample=upsample,
                max_samples_per_class=max_samples_per_class,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
            )

            K = train_dataset.get_K()
            feats_per_phase = train_dataset.get_feats_per_phase()
            feat_scaler = train_dataset.get_scaler()
            feature_cols = sum([train_dataset.phase_feature_names[p] for p in range(1, K + 1)], [])

            log_csv_path = str(out_dir / f"{args.run_name}_{K}_{args.log_name}")
            ckpt_path = str(out_dir / f"{args.run_name}_{K}_{args.ckpt_name}")

            _run_single_pipeline(
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                K=K,
                feats_per_phase=feats_per_phase,
                encoder_out_dim=encoder_out_dim,
                proj_out_dim=proj_out_dim,
                dropout=dropout,
                feature_cols=feature_cols,
                feat_scaler=feat_scaler,
                class_to_idx=class_to_idx,
                label_col=label_col,
                pcfg=pcfg,
                ft_cfg=ft_cfg,
                log_csv_path=log_csv_path,
                ckpt_path=ckpt_path,
                skip_finetune=args.skip_finetune,
            )


if __name__ == "__main__":
    main()
