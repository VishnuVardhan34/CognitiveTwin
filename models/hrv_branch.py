"""HRV branch: 1D-CNN + GRU for HRV feature sequences."""

from __future__ import annotations

import torch
import torch.nn as nn


class HRVNet(nn.Module):
    """1D-CNN + GRU network for HRV feature sequences.

    Args:
        n_features: Number of HRV features per timestep. Default 10.
        embedding_dim: Output embedding dimension. Default 32.
        hidden_size: GRU hidden state size. Default 64.
        n_gru_layers: Number of GRU layers. Default 2.
        dropout: Dropout probability. Default 0.3.
    """

    def __init__(
        self,
        n_features: int = 10,
        embedding_dim: int = 32,
        hidden_size: int = 64,
        n_gru_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        # 1D-CNN feature extraction
        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        # GRU temporal modelling
        self.gru = nn.GRU(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=n_gru_layers,
            batch_first=True,
            dropout=dropout if n_gru_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, n_features).

        Returns:
            Embedding tensor of shape (batch, 32).
        """
        # CNN expects (batch, channels, seq_len)
        x = x.permute(0, 2, 1)       # (B, F, T)
        x = self.cnn(x)               # (B, 64, T)
        x = x.permute(0, 2, 1)       # (B, T, 64)

        # GRU: take last hidden state
        _, h_n = self.gru(x)          # h_n: (n_layers, B, hidden_size)
        h_last = h_n[-1]              # (B, hidden_size)
        h_last = self.dropout(h_last)
        return self.fc(h_last)        # (B, 32)
