from __future__ import annotations

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    1D Conv encoder. Input: V (B, K, F)
      - treat features as channels, phases as sequence length:
        x = V.transpose(1, 2) => (B, F, K)
    Output: h (B, out_dim)
    """
    def __init__(self, feats_per_phase: int, hidden_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(feats_per_phase, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(hidden_dim, out_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, V: torch.Tensor) -> torch.Tensor:
        x = V.transpose(1, 2)  # (B, F, K)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x).squeeze(-1)  # (B, out_dim)
        return x


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class DownstreamClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, num_classes: int, encoder_out_dim: int):
        super().__init__()
        self.encoder = encoder
        self.fc = nn.Linear(encoder_out_dim, num_classes)

        nn.init.kaiming_normal_(self.fc.weight, mode="fan_out", nonlinearity="relu")
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(self, V: torch.Tensor) -> torch.Tensor:
        h = self.encoder(V)
        return self.fc(h)
