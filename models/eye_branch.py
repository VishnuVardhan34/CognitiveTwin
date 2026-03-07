"""Eye-tracking branch: 1D-CNN + BiLSTM + attention pooling."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EyeTrackingNet(nn.Module):
    """1D-CNN + Bidirectional LSTM + attention pooling for eye-tracking sequences.

    Args:
        n_features: Number of input features per timestep. Default 7.
        embedding_dim: Output embedding dimension. Default 64.
        hidden_size: LSTM hidden state size. Default 64.
        n_lstm_layers: Number of LSTM layers. Default 2.
        dropout: Dropout probability. Default 0.3.
    """

    def __init__(
        self,
        n_features: int = 7,
        embedding_dim: int = 64,
        hidden_size: int = 64,
        n_lstm_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        # 1D-CNN feature extraction
        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout),
        )

        # Bidirectional LSTM
        self.bilstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=n_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_lstm_layers > 1 else 0.0,
        )

        # Attention pooling
        self.attn_fc = nn.Linear(hidden_size * 2, 1)

        # Output projection
        self.fc = nn.Linear(hidden_size * 2, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, n_features).

        Returns:
            Embedding tensor of shape (batch, 64).
        """
        # CNN expects (batch, channels, seq_len)
        x = x.permute(0, 2, 1)           # (B, F, T)
        x = self.cnn(x)                   # (B, 64, T//2)
        x = x.permute(0, 2, 1)           # (B, T//2, 64)

        # BiLSTM
        lstm_out, _ = self.bilstm(x)      # (B, T//2, 128)

        # Attention pooling
        attn_weights = F.softmax(self.attn_fc(lstm_out), dim=1)  # (B, T//2, 1)
        pooled = (attn_weights * lstm_out).sum(dim=1)             # (B, 128)

        pooled = self.dropout(pooled)
        return self.fc(pooled)             # (B, 64)
