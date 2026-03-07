"""Cross-modal transformer attention for fusing EEG, eye, and HRV embeddings."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class CrossModalAttention(nn.Module):
    """Cross-modal self-attention over EEG, eye-tracking, and HRV tokens.

    Projects three modality embeddings to a common dimension, stacks them as
    a 3-token sequence, applies multi-head self-attention with a feed-forward
    network, and returns the attended tokens together with the attention
    weight matrix.

    Args:
        eeg_dim: Input dimension of the EEG embedding. Default 64.
        eye_dim: Input dimension of the eye-tracking embedding. Default 64.
        hrv_dim: Input dimension of the HRV embedding. Default 32.
        common_dim: Common projected dimension. Default 64.
        n_heads: Number of attention heads. Default 4.
        dropout: Dropout probability. Default 0.1.
    """

    def __init__(
        self,
        eeg_dim: int = 64,
        eye_dim: int = 64,
        hrv_dim: int = 32,
        common_dim: int = 64,
        n_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.common_dim = common_dim

        # Projection layers
        self.proj_eeg = nn.Linear(eeg_dim, common_dim)
        self.proj_eye = nn.Linear(eye_dim, common_dim)
        self.proj_hrv = nn.Linear(hrv_dim, common_dim)

        # Multi-head self-attention (operates on the 3-token sequence)
        self.mha = nn.MultiheadAttention(
            embed_dim=common_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(common_dim)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(common_dim, common_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(common_dim * 4, common_dim),
        )
        self.norm2 = nn.LayerNorm(common_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        eeg_emb: torch.Tensor,
        eye_emb: torch.Tensor,
        hrv_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            eeg_emb: EEG embedding of shape (batch, eeg_dim).
            eye_emb: Eye-tracking embedding of shape (batch, eye_dim).
            hrv_emb: HRV embedding of shape (batch, hrv_dim).

        Returns:
            Tuple:
                - ``tokens``: Attended token tensor of shape (batch, 3, common_dim).
                - ``attn_weights``: Attention weight matrix (batch, 3, 3).
        """
        # Project to common dimension: (B, common_dim) each
        e = self.proj_eeg(eeg_emb)
        y = self.proj_eye(eye_emb)
        h = self.proj_hrv(hrv_emb)

        # Stack as sequence: (B, 3, common_dim)
        tokens = torch.stack([e, y, h], dim=1)

        # Multi-head self-attention with residual
        attended, attn_weights = self.mha(tokens, tokens, tokens)
        tokens = self.norm1(tokens + self.dropout(attended))

        # FFN with residual
        tokens = self.norm2(tokens + self.dropout(self.ffn(tokens)))

        return tokens, attn_weights
