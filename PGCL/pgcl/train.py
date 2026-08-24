from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


def augment_views(V: torch.Tensor, noise_std: float = 0.05, feat_drop_prob: float = 0.1, training: bool = True) -> torch.Tensor:
    """
    V: (B, K, F)
    - Gaussian jitter: N(0, noise_std)
    - Feature dropout: each element masked to 0 with probability feat_drop_prob
    """
    if (not training) or (noise_std <= 0 and feat_drop_prob <= 0):
        return V
    if noise_std > 0:
        V = V + torch.randn_like(V) * noise_std
    if feat_drop_prob > 0:
        drop_mask = (torch.rand_like(V) < feat_drop_prob)
        V = V.masked_fill(drop_mask, 0.0)
    return V


def build_multi_views(
    V_batch: torch.Tensor,
    K: int,
    encoder: nn.Module,
    proj_head: nn.Module,
    device: torch.device,
    noise_std: float,
    feat_drop_prob: float,
    training: bool = True,
) -> torch.Tensor:
    """
    Vectorized: one forward to get orig + K masked views.
    Returns z_flat: (B*(K+1), D_proj)
    """
    B = V_batch.size(0)
    V_batch = V_batch.to(device)
    V_batch = augment_views(V_batch, noise_std=noise_std, feat_drop_prob=feat_drop_prob, training=training)

    h_orig = encoder(V_batch)  # (B, D)

    # (K,K) eye => mask which phase to zero; expand to (B,K,K,1)
    mask = torch.eye(K, device=device).view(1, K, K, 1).expand(B, K, K, 1)
    V_expanded = V_batch.unsqueeze(1)  # (B,1,K,F)
    V_masked = V_expanded * (1.0 - mask)  # (B,K,K,F)

    # flatten B*K and run encoder
    V_flat = V_masked.flatten(0, 1)  # (B*K, K, F)
    h_masked = encoder(V_flat).view(B, K, -1)  # (B,K,D)

    h_all = torch.cat([h_masked, h_orig.unsqueeze(1)], dim=1)  # (B,K+1,D)
    z_flat = proj_head(h_all.flatten(0, 1))  # (B*(K+1), D_proj)
    return z_flat


def pgcl_loss(z_stage: torch.Tensor, K: int, temperature: float = 0.5) -> torch.Tensor:
    """
    z_stage : (B*(K+1), D)
    """
    device = z_stage.device
    N_total = z_stage.shape[0]
    K_plus_1 = K + 1
    batch_size = N_total // K_plus_1

    z_norm = F.normalize(z_stage, dim=1)
    sim_matrix = torch.mm(z_norm, z_norm.t()) / temperature

    positives_mask = torch.zeros_like(sim_matrix, dtype=torch.bool)
    for i in range(batch_size):
        start = i * K_plus_1
        end = start + K_plus_1
        idxs = torch.arange(start, end, device=device)
        mesh_i, mesh_j = torch.meshgrid(idxs, idxs, indexing="ij")
        positives_mask[mesh_i, mesh_j] = True

    self_mask = torch.eye(N_total, dtype=torch.bool, device=device)
    positives_mask = positives_mask & (~self_mask)

    # exp(sim) with stability
    sim_matrix = sim_matrix - sim_matrix.max(dim=1, keepdim=True).values
    exp_sim = torch.exp(sim_matrix)

    denom = exp_sim.masked_fill(self_mask, 0.0).sum(dim=1)  # exclude self
    pos_sum = exp_sim.masked_fill(~positives_mask, 0.0).sum(dim=1)

    loss = -torch.log((pos_sum + 1e-12) / (denom + 1e-12))
    return loss.mean()


@torch.no_grad()
def evaluate_pgcl(
    encoder: nn.Module,
    proj_head: nn.Module,
    dataloader,
    device: torch.device,
    K: int,
    temperature: float,
    noise_std: float,
    feat_drop_prob: float,
) -> float:
    encoder.eval()
    proj_head.eval()
    total_loss = 0.0
    total_samples = 0
    for V_batch, _ in dataloader:
        B = V_batch.size(0)
        z_flat = build_multi_views(
            V_batch, K, encoder, proj_head, device,
            noise_std=noise_std, feat_drop_prob=feat_drop_prob, training=True
        )
        loss = pgcl_loss(z_flat, K, temperature=temperature)
        total_loss += float(loss.item()) * B
        total_samples += B
    return total_loss / max(total_samples, 1)


def pretrain_pgcl(
    encoder: nn.Module,
    proj_head: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    K: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    temperature: float,
    noise_std: float,
    feat_drop_prob: float,
    patience: int,
    max_grad_norm: float,
    use_amp: bool,
    log_csv_path: str,
    scheduler_t_max: Optional[int] = None,
    on_best_ckpt=None,
) -> Dict[str, float]:
    """
    Returns summary dict. If on_best_ckpt is provided, it will be called as:
      on_best_ckpt(encoder, proj_head)
    """
    log_path = Path(log_csv_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", newline="") as f:
        csv.writer(f).writerow(["epoch", "loss", "val_loss", "lr", "temperature"])

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(proj_head.parameters()), lr=lr, weight_decay=weight_decay)

    if scheduler_t_max is None:
        scheduler_t_max = epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=scheduler_t_max)

    amp_enabled = use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_val = float("inf")
    best_encoder_state = None
    best_projection_state = None
    stalled = 0

    encoder.to(device)
    proj_head.to(device)

    for ep in range(1, epochs + 1):
        encoder.train()
        proj_head.train()
        loss_sum = 0.0
        total = 0

        for V_batch, _ in train_loader:
            B = V_batch.size(0)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                z_flat = build_multi_views(
                    V_batch, K, encoder, proj_head, device,
                    noise_std=noise_std, feat_drop_prob=feat_drop_prob, training=True
                )
                loss = pgcl_loss(z_flat, K, temperature=temperature)

            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss detected")

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(proj_head.parameters()), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            loss_sum += float(loss.item()) * B
            total += B

        scheduler.step()
        train_loss = loss_sum / max(total, 1)
        val_loss = evaluate_pgcl(encoder, proj_head, val_loader, device, K, temperature, noise_std, feat_drop_prob)
        cur_lr = optimizer.param_groups[0]["lr"]

        with log_path.open("a", newline="") as f:
            csv.writer(f).writerow([ep, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{cur_lr:.6e}", f"{temperature:.4f}"])

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_encoder_state = {
                key: value.detach().cpu().clone()
                for key, value in encoder.state_dict().items()
            }
            best_projection_state = {
                key: value.detach().cpu().clone()
                for key, value in proj_head.state_dict().items()
            }
            stalled = 0
            if on_best_ckpt is not None:
                on_best_ckpt(encoder, proj_head)
        else:
            stalled += 1
            if stalled >= patience:
                break

    if best_encoder_state is not None and best_projection_state is not None:
        encoder.load_state_dict(best_encoder_state)
        proj_head.load_state_dict(best_projection_state)

    return {"best_val_loss": best_val}


@torch.no_grad()
def evaluate_downstream_metrics(model: nn.Module, dataloader, device: torch.device) -> Tuple[float, float, float, float]:
    model.eval()
    all_preds, all_labels = [], []
    for V_batch, labels in dataloader:
        V_batch = V_batch.to(device)
        labels = labels.to(device)
        logits = model(V_batch)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    p, r, _, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro")
    return float(acc), float(f1), float(p), float(r)


def fine_tune_classifier(
    train_loader,
    val_loader,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scheduler,
    device: torch.device,
    epochs: int,
    use_amp: bool = True,
    max_grad_norm: float = 1.0,
    unfreeze_encoder: bool = False,
) -> Dict[str, float]:
    """
    Fine-tune the downstream classifier.
    If unfreeze_encoder is False, caller should already freeze encoder params.
    """
    amp_enabled = use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_f1 = -1.0
    best_model_state = None
    for ep in range(1, epochs + 1):
        model.train()
        if not unfreeze_encoder:
            model.encoder.eval()

        for V_batch, labels in train_loader:
            V_batch = V_batch.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = model(V_batch)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

        if scheduler is not None:
            scheduler.step()

        acc, f1, p, r = evaluate_downstream_metrics(model, val_loader, device)
        if f1 > best_f1:
            best_f1 = f1
            best_model_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return {"best_val_f1": best_f1}


def set_encoder_requires_grad(model: nn.Module, requires_grad: bool) -> None:
    """
    Set requires_grad for all encoder parameters.
    """
    for p in model.encoder.parameters():
        p.requires_grad = requires_grad


def fine_tune_two_stage(
    train_loader,
    val_loader,
    model: nn.Module,
    device: torch.device,
    stage_a_epochs: int,
    stage_a_lr: float,
    stage_a_weight_decay: float,
    stage_a_t_max: int,
    stage_b_enabled: bool = False,
    stage_b_epochs: int = 10,
    stage_b_lr: float = 1e-5,
    stage_b_weight_decay: float = 1e-4,
    stage_b_t_max: int = 10,
    use_amp: bool = True,
    max_grad_norm: float = 1.0,
) -> Dict[str, float]:
    """
    Two-stage fine-tuning:
      Stage A: freeze encoder, train only classifier head
      Stage B (optional): unfreeze encoder, train entire model with smaller LR
    """
    criterion = torch.nn.CrossEntropyLoss()
    best_f1 = -1.0

    # ========== Stage A: Freeze encoder, train classifier ==========
    set_encoder_requires_grad(model, requires_grad=False)

    optimizer_a = torch.optim.Adam(
        model.fc.parameters(),
        lr=stage_a_lr,
        weight_decay=stage_a_weight_decay,
    )
    scheduler_a = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_a, T_max=stage_a_t_max)

    results_a = fine_tune_classifier(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        optimizer=optimizer_a,
        criterion=criterion,
        scheduler=scheduler_a,
        device=device,
        epochs=stage_a_epochs,
        use_amp=use_amp,
        max_grad_norm=max_grad_norm,
        unfreeze_encoder=False,
    )
    best_f1 = results_a["best_val_f1"]
    best_model_state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }

    # ========== Stage B: Unfreeze encoder, joint fine-tuning ==========
    if stage_b_enabled:
        set_encoder_requires_grad(model, requires_grad=True)

        optimizer_b = torch.optim.Adam(
            model.parameters(),
            lr=stage_b_lr,
            weight_decay=stage_b_weight_decay,
        )
        scheduler_b = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_b, T_max=stage_b_t_max)

        results_b = fine_tune_classifier(
            train_loader=train_loader,
            val_loader=val_loader,
            model=model,
            optimizer=optimizer_b,
            criterion=criterion,
            scheduler=scheduler_b,
            device=device,
            epochs=stage_b_epochs,
            use_amp=use_amp,
            max_grad_norm=max_grad_norm,
            unfreeze_encoder=True,
        )
        if results_b["best_val_f1"] > best_f1:
            best_f1 = results_b["best_val_f1"]
            best_model_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    model.load_state_dict(best_model_state)

    return {"best_val_f1": best_f1}
