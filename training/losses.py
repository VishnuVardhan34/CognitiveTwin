"""Multi-task uncertainty-weighted loss for CognitiveTwin (Kendall et al. 2018)."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class CognitiveTwinLoss(nn.Module):
    """Uncertainty-weighted multi-task loss combining classification,
    regression, auxiliary branch supervision, and modal agreement terms.

    Based on:
        Kendall, A., Gal, Y., & Cipolla, R. (2018). Multi-Task Learning
        Using Uncertainty to Weigh Losses for Scene Geometry and Semantics.
        CVPR 2018.

    Learnable parameters ``log_sigma_1`` … ``log_sigma_4`` are the log
    standard deviations for each task.  The final loss is:

    .. math::

        \\mathcal{L} = \\frac{L_{cls}}{2 \\sigma_1^2}
                     + \\frac{L_{reg}}{2 \\sigma_2^2}
                     + \\frac{L_{aux}}{2 \\sigma_3^2}
                     + \\frac{L_{agree}}{2 \\sigma_4^2}
                     + \\log \\sigma_1 + \\log \\sigma_2
                       + \\log \\sigma_3 + \\log \\sigma_4

    Args:
        n_classes: Number of classification classes. Default 4.
        lambda_reg: Weight for regression loss before uncertainty scaling.
            Default 1.0.
        lambda_aux: Weight for auxiliary classification loss. Default 0.3.
        lambda_agree: Weight for modal agreement (KL) loss. Default 0.1.
    """

    def __init__(
        self,
        n_classes: int = 4,
        lambda_reg: float = 1.0,
        lambda_aux: float = 0.3,
        lambda_agree: float = 0.1,
    ) -> None:
        super().__init__()
        self.lambda_reg = lambda_reg
        self.lambda_aux = lambda_aux
        self.lambda_agree = lambda_agree

        # Learnable log-variance parameters: log_var_i = log(σ_i²), initialised to 0.
        # The regularisation term 0.5 * log_var = 0.5 * log(σ²) = log(σ) matches the
        # Kendall et al. 2018 formulation exactly.
        self.log_var1 = nn.Parameter(torch.zeros(1))   # classification
        self.log_var2 = nn.Parameter(torch.zeros(1))   # regression
        self.log_var3 = nn.Parameter(torch.zeros(1))   # auxiliary
        self.log_var4 = nn.Parameter(torch.zeros(1))   # agreement

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        av_targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute the total multi-task loss.

        Args:
            outputs: Dictionary returned by ``CognitiveTwinFusionModel.forward()``.
            labels: Ground-truth class labels of shape (batch,), dtype int64.
            av_targets: Arousal-valence targets of shape (batch, 2), range [-1, 1].

        Returns:
            Dictionary with keys:
                - ``total``          : scalar total loss
                - ``cls``            : classification loss
                - ``reg``            : regression loss
                - ``aux``            : auxiliary classification loss
                - ``agree``          : agreement (KL) loss
                - ``task_weights``   : dict with per-task precision weights
        """
        fused_logits   = outputs["fused_logits"]    # (B, C)
        eeg_logits     = outputs["eeg_logits"]      # (B, C)
        eye_logits     = outputs["eye_logits"]      # (B, C)
        hrv_logits     = outputs["hrv_logits"]      # (B, C)
        av_pred        = outputs["arousal_valence"] # (B, 2)

        # ---- L_cls: cross-entropy on fused logits ----
        L_cls = F.cross_entropy(fused_logits, labels)

        # ---- L_reg: MSE on arousal-valence ----
        L_reg = F.mse_loss(av_pred, av_targets)

        # ---- L_aux: auxiliary cross-entropy on per-modality logits ----
        L_eeg = F.cross_entropy(eeg_logits, labels)
        L_eye = F.cross_entropy(eye_logits, labels)
        L_hrv = F.cross_entropy(hrv_logits, labels)
        L_aux = (L_eeg + L_eye + L_hrv) / 3.0

        # ---- L_agree: KL divergence between fused and per-branch probs ----
        log_p_fused = F.log_softmax(fused_logits, dim=1)
        p_eeg = F.softmax(eeg_logits, dim=1)
        p_eye = F.softmax(eye_logits, dim=1)
        p_hrv = F.softmax(hrv_logits, dim=1)

        # KL(p_branch || p_fused)
        kl_eeg = F.kl_div(log_p_fused, p_eeg, reduction="batchmean")
        kl_eye = F.kl_div(log_p_fused, p_eye, reduction="batchmean")
        kl_hrv = F.kl_div(log_p_fused, p_hrv, reduction="batchmean")
        L_agree = (kl_eeg + kl_eye + kl_hrv) / 3.0

        # ---- Uncertainty-weighted combination ----
        # σ² = exp(log_var)
        w1 = torch.exp(-self.log_var1)  # 1/σ²
        w2 = torch.exp(-self.log_var2)
        w3 = torch.exp(-self.log_var3)
        w4 = torch.exp(-self.log_var4)

        total = (
            w1 * L_cls    + self.log_var1 * 0.5 +
            w2 * self.lambda_reg * L_reg  + self.log_var2 * 0.5 +
            w3 * self.lambda_aux * L_aux  + self.log_var3 * 0.5 +
            w4 * self.lambda_agree * L_agree + self.log_var4 * 0.5
        )

        task_weights = {
            "cls":   float(w1.item()),
            "reg":   float(w2.item()),
            "aux":   float(w3.item()),
            "agree": float(w4.item()),
        }

        return {
            "total": total,
            "cls":   L_cls,
            "reg":   L_reg,
            "aux":   L_aux,
            "agree": L_agree,
            "task_weights": task_weights,
        }
