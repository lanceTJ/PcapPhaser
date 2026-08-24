import math
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

PGCL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PGCL_ROOT))

from pgcl.model import DownstreamClassifier, Encoder, ProjectionHead
from pgcl.train import (
    evaluate_downstream_metrics,
    fine_tune_two_stage,
    pretrain_pgcl,
)


class TrainingSmokeTests(unittest.TestCase):
    def test_cpu_pretraining_and_finetuning_with_amp_enabled(self) -> None:
        torch.manual_seed(42)
        features = torch.randn(16, 3, 4)
        labels = torch.tensor([0, 1] * 8)
        loader = DataLoader(TensorDataset(features, labels), batch_size=4)
        device = torch.device("cpu")

        encoder = Encoder(
            feats_per_phase=4,
            hidden_dim=8,
            out_dim=8,
            dropout=0.0,
        )
        projection = ProjectionHead(in_dim=8, out_dim=4)

        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "pretrain.csv"
            pretrain_result = pretrain_pgcl(
                encoder=encoder,
                proj_head=projection,
                train_loader=loader,
                val_loader=loader,
                device=device,
                K=3,
                epochs=2,
                lr=1e-3,
                weight_decay=0.0,
                temperature=0.5,
                noise_std=0.0,
                feat_drop_prob=0.0,
                patience=2,
                max_grad_norm=1.0,
                use_amp=True,
                log_csv_path=str(log_path),
            )
            self.assertTrue(log_path.exists())

        self.assertTrue(math.isfinite(pretrain_result["best_val_loss"]))

        classifier = DownstreamClassifier(
            encoder=encoder,
            num_classes=2,
            encoder_out_dim=8,
        )
        finetune_result = fine_tune_two_stage(
            train_loader=loader,
            val_loader=loader,
            model=classifier,
            device=device,
            stage_a_epochs=1,
            stage_a_lr=1e-2,
            stage_a_weight_decay=0.0,
            stage_a_t_max=1,
            stage_b_enabled=True,
            stage_b_epochs=1,
            stage_b_lr=1e-3,
            stage_b_weight_decay=0.0,
            stage_b_t_max=1,
            use_amp=True,
        )

        metrics = evaluate_downstream_metrics(classifier, loader, device)
        self.assertTrue(0.0 <= finetune_result["best_val_f1"] <= 1.0)
        self.assertTrue(all(0.0 <= metric <= 1.0 for metric in metrics))


if __name__ == "__main__":
    unittest.main()
