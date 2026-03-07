"""Modality confidence gating for adaptive fusion weighting."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalityConfidenceGate(nn.Module):
    """Per-modality confidence scorer that produces normalised fusion weights.

    Each modality embedding is independently scored through a two-layer MLP.
    The three scalar scores are then passed through a softmax to produce
    normalised weights ``[w_eeg, w_eye, w_hrv]``.

    Args:
        eeg_dim: Dimension of the EEG embedding. Default 64.
        eye_dim: Dimension of the eye-tracking embedding. Default 64.
        hrv_dim: Dimension of the HRV embedding. Default 32.
    """

    def __init__(
        self,
        eeg_dim: int = 64,
        eye_dim: int = 64,
        hrv_dim: int = 32,
    ) -> None:
        super().__init__()

        self.eeg_scorer = nn.Sequential(
            nn.Linear(eeg_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.eye_scorer = nn.Sequential(
            nn.Linear(eye_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.hrv_scorer = nn.Sequential(
            nn.Linear(hrv_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(
        self,
        eeg_emb: torch.Tensor,
        eye_emb: torch.Tensor,
        hrv_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Compute normalised confidence weights for each modality.

        Args:
            eeg_emb: EEG embedding of shape (batch, eeg_dim).
            eye_emb: Eye-tracking embedding of shape (batch, eye_dim).
            hrv_emb: HRV embedding of shape (batch, hrv_dim).

        Returns:
            Weight tensor of shape (batch, 3): ``[w_eeg, w_eye, w_hrv]``.
        """
        s_eeg = self.eeg_scorer(eeg_emb)   # (B, 1)
        s_eye = self.eye_scorer(eye_emb)   # (B, 1)
        s_hrv = self.hrv_scorer(hrv_emb)   # (B, 1)

        scores = torch.cat([s_eeg, s_eye, s_hrv], dim=1)  # (B, 3)
        weights = F.softmax(scores, dim=1)                  # (B, 3)
        return weights
