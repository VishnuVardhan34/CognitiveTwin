"""EEG branch: EEGNet-inspired 2D CNN operating on CWT scalograms."""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Tuple


class EEGWaveletCNN(nn.Module):
    """EEGNet-inspired 2D CNN for scalogram-based EEG feature extraction.

    Takes CWT power scalogram tensors and produces a 64-dimensional embedding.

    Architecture:
        1. Temporal conv (1 × 32 kernel) + BatchNorm
        2. Depthwise conv (n_channels × 1 kernel, across frequency dimension) + BatchNorm + ELU
        3. Average pooling (1 × 4)
        4. Separable conv (1 × 8 kernel, pointwise 1 × 1) + BatchNorm + ELU
        5. Average pooling (1 × 8)
        6. Dropout
        7. Flatten → Linear(64)

    Args:
        n_channels: Number of EEG channels. Default 32.
        n_freqs: Number of frequency bins in the scalogram. Default 64.
        n_time: Number of time steps in the scalogram. Default 256.
        embedding_dim: Output embedding dimension. Default 64.
        dropout: Dropout probability. Default 0.3.
    """

    def __init__(
        self,
        n_channels: int = 32,
        n_freqs: int = 64,
        n_time: int = 256,
        embedding_dim: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.n_freqs = n_freqs
        self.n_time = n_time
        self.embedding_dim = embedding_dim

        F1, D, F2 = 8, 2, 16  # EEGNet filter counts

        # Block 1: Temporal convolution
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, F1, (1, 32), padding=(0, 16), bias=False),
            nn.BatchNorm2d(F1),
        )

        # Block 2: Depthwise spatial (frequency) convolution
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (n_freqs, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )

        # Block 3: Separable convolution
        self.separable_conv = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )

        # Compute flat size with a dummy forward pass
        self._flat_size = self._get_flat_size(n_channels, n_freqs, n_time)
        self.fc = nn.Linear(self._flat_size, embedding_dim)

    def _get_flat_size(self, n_channels: int, n_freqs: int, n_time: int) -> int:
        """Determine flattened feature dimension via dummy forward pass."""
        with torch.no_grad():
            dummy = torch.zeros(1, n_channels, n_freqs, n_time)
            out = self._forward_conv(dummy)
            return int(out.numel())

    def _forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        """Convolutional feature extraction (without the final linear layer).

        Args:
            x: Input tensor of shape (batch, n_channels, n_freqs, n_time).

        Returns:
            Flattened feature tensor.
        """
        # Aggregate EEG channels → (B, 1, F, T) so the CNN sees a single 2D scalogram image
        x = x.mean(dim=1, keepdim=True)  # (B, 1, F, T)
        x = self.temporal_conv(x)         # (B, F1, F, T')
        x = self.depthwise_conv(x)        # (B, F1*D, 1, T'')
        x = self.separable_conv(x)        # (B, F2, 1, T''')
        return x.flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Scalogram tensor of shape (batch, n_channels, n_freqs, n_time).

        Returns:
            Embedding tensor of shape (batch, 64).
        """
        feat = self._forward_conv(x)
        return self.fc(feat)
