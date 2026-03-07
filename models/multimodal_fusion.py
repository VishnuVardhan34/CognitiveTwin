"""Multimodal fusion model combining EEG, eye tracking, and HRV branches."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .eeg_branch import EEGWaveletCNN
from .eye_branch import EyeTrackingNet
from .hrv_branch import HRVNet
from .cross_attention import CrossModalAttention
from .confidence_gate import ModalityConfidenceGate


CLASS_NAMES: List[str] = ["Underload", "Optimal", "Overload", "Fatigue"]


class CognitiveTwinFusionModel(nn.Module):
    """Full multimodal fusion model for cognitive-load estimation.

    Combines EEG scalogram features (2D-CNN), eye-tracking features (1D-CNN +
    BiLSTM + Attention), and HRV features (1D-CNN + GRU) via cross-modal
    transformer attention and a confidence gate.  Produces four-class
    cognitive-load predictions and a continuous arousal-valence estimate.

    Decision fusion:
        final_probs = softmax(decision_weights) ⊙
                      [P(fused), P(eeg), P(eye), P(hrv)]

    Args:
        n_channels_eeg: Number of EEG channels. Default 32.
        n_freqs: Scalogram frequency bins. Default 64.
        n_time: Scalogram time steps. Default 256.
        n_eye_features: Eye feature dimension. Default 7.
        n_hrv_features: HRV feature dimension. Default 10.
        n_classes: Number of cognitive-load classes. Default 4.
        dropout: Dropout probability. Default 0.3.
        cross_attn_heads: Multi-head attention heads. Default 4.
        initial_decision_weights: Initial weights for
            [fused, eeg, eye, hrv] decision fusion. Default [0.5, 0.2, 0.2, 0.1].
    """

    def __init__(
        self,
        n_channels_eeg: int = 32,
        n_freqs: int = 64,
        n_time: int = 256,
        n_eye_features: int = 7,
        n_hrv_features: int = 10,
        n_classes: int = 4,
        dropout: float = 0.3,
        cross_attn_heads: int = 4,
        initial_decision_weights: Optional[List[float]] = None,
    ) -> None:
        super().__init__()

        if initial_decision_weights is None:
            initial_decision_weights = [0.5, 0.2, 0.2, 0.1]

        # ---- Modality branches ----
        self.eeg_branch = EEGWaveletCNN(
            n_channels=n_channels_eeg,
            n_freqs=n_freqs,
            n_time=n_time,
            embedding_dim=64,
            dropout=dropout,
        )
        self.eye_branch = EyeTrackingNet(
            n_features=n_eye_features,
            embedding_dim=64,
            dropout=dropout,
        )
        self.hrv_branch = HRVNet(
            n_features=n_hrv_features,
            embedding_dim=32,
            dropout=dropout,
        )

        # ---- Fusion modules ----
        self.cross_attention = CrossModalAttention(
            eeg_dim=64,
            eye_dim=64,
            hrv_dim=32,
            common_dim=64,
            n_heads=cross_attn_heads,
            dropout=dropout,
        )
        self.confidence_gate = ModalityConfidenceGate(
            eeg_dim=64,
            eye_dim=64,
            hrv_dim=32,
        )

        # ---- Per-modality classification heads ----
        self.eeg_classifier = nn.Linear(64, n_classes)
        self.eye_classifier = nn.Linear(64, n_classes)
        self.hrv_classifier = nn.Linear(32, n_classes)

        # ---- Fused classification head ----
        self.fused_classifier = nn.Sequential(
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

        # ---- Arousal-valence regression head ----
        self.av_regressor = nn.Sequential(
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
            nn.Tanh(),
        )

        # ---- Learnable decision fusion weights [fused, eeg, eye, hrv] ----
        # Raw (unconstrained) values; softmax in forward() guarantees they sum to 1.
        self.decision_weights = nn.Parameter(
            torch.tensor(initial_decision_weights, dtype=torch.float32)
        )

        self.n_classes = n_classes
        self.class_names = CLASS_NAMES[:n_classes]

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        eeg_scalogram: torch.Tensor,
        eye_seq: torch.Tensor,
        hrv_seq: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the full fusion model.

        Args:
            eeg_scalogram: Shape (batch, n_channels, n_freqs, n_time).
            eye_seq:        Shape (batch, seq_len, n_eye_features).
            hrv_seq:        Shape (batch, seq_len, n_hrv_features).

        Returns:
            Dictionary containing:
                - ``fused_logits``      : (batch, n_classes)
                - ``eeg_logits``        : (batch, n_classes)
                - ``eye_logits``        : (batch, n_classes)
                - ``hrv_logits``        : (batch, n_classes)
                - ``arousal_valence``   : (batch, 2) in [−1, 1] (Tanh output)
                - ``fused_embedding``   : (batch, 64)
                - ``confidence_weights``: (batch, 3)
                - ``attention_weights`` : (batch, 3, 3)
                - ``final_probs``       : (batch, n_classes)
                - ``decision_weights``  : (n_classes_fusion,) – raw learnable weights
        """
        # ---- 1. Per-modality embeddings ----
        eeg_emb = self.eeg_branch(eeg_scalogram)   # (B, 64)
        eye_emb = self.eye_branch(eye_seq)           # (B, 64)
        hrv_emb = self.hrv_branch(hrv_seq)           # (B, 32)

        # ---- 2. Per-modality logits ----
        eeg_logits = self.eeg_classifier(eeg_emb)   # (B, C)
        eye_logits = self.eye_classifier(eye_emb)   # (B, C)
        hrv_logits = self.hrv_classifier(hrv_emb)   # (B, C)

        # ---- 3. Cross-modal attention ----
        tokens, attn_weights = self.cross_attention(eeg_emb, eye_emb, hrv_emb)
        # tokens: (B, 3, 64)

        # ---- 4. Confidence gate ----
        conf_weights = self.confidence_gate(eeg_emb, eye_emb, hrv_emb)  # (B, 3)

        # ---- 5. Weighted fusion of attended tokens ----
        # conf_weights: (B, 3) → expand to (B, 3, 1)
        fused_emb = (tokens * conf_weights.unsqueeze(-1)).sum(dim=1)  # (B, 64)

        # ---- 6. Fused classifier + AV regressor ----
        fused_logits = self.fused_classifier(fused_emb)  # (B, C)
        av = self.av_regressor(fused_emb)                 # (B, 2)

        # ---- 7. Weighted decision fusion ----
        # Normalise decision weights
        dw = F.softmax(self.decision_weights, dim=0)  # (4,)

        p_fused = F.softmax(fused_logits, dim=1)   # (B, C)
        p_eeg   = F.softmax(eeg_logits,   dim=1)   # (B, C)
        p_eye   = F.softmax(eye_logits,   dim=1)   # (B, C)
        p_hrv   = F.softmax(hrv_logits,   dim=1)   # (B, C)

        # Stack probabilities: (B, 4, C) then weighted sum → (B, C)
        all_probs = torch.stack([p_fused, p_eeg, p_eye, p_hrv], dim=1)   # (B, 4, C)
        final_probs = (dw.view(1, 4, 1) * all_probs).sum(dim=1)           # (B, C)

        return {
            "fused_logits": fused_logits,
            "eeg_logits": eeg_logits,
            "eye_logits": eye_logits,
            "hrv_logits": hrv_logits,
            "arousal_valence": av,
            "fused_embedding": fused_emb,
            "confidence_weights": conf_weights,
            "attention_weights": attn_weights,
            "final_probs": final_probs,
            "decision_weights": self.decision_weights,
        }

    # ------------------------------------------------------------------
    # State declaration
    # ------------------------------------------------------------------

    def declare_state(
        self, outputs: Dict[str, torch.Tensor]
    ) -> List[Dict[str, object]]:
        """Interpret model outputs into rich per-sample state dictionaries.

        Args:
            outputs: Dictionary returned by ``forward()``.

        Returns:
            List of per-sample state dictionaries, each containing:
                - ``predicted_class``         : int – argmax of final_probs
                - ``class_name``              : str
                - ``confidence``              : float – max probability
                - ``class_probabilities``     : dict mapping class name → probability
                - ``arousal``                 : float – normalised [0, 1] from Tanh
                - ``valence``                 : float – normalised [0, 1] from Tanh
                - ``modality_contributions``  : dict – per-modality confidence weights
                - ``per_modality_predictions``: dict – per-branch predicted class
                - ``modality_agreement``      : float – fraction of branches agreeing
                - ``decision_fusion_weights`` : dict – normalised fusion weights
        """
        final_probs = outputs["final_probs"].detach().cpu()          # (B, C)
        av = outputs["arousal_valence"].detach().cpu()                # (B, 2)
        conf_w = outputs["confidence_weights"].detach().cpu()         # (B, 3)
        dw = F.softmax(outputs["decision_weights"].detach().cpu(), dim=0)  # (4,)

        eeg_probs = F.softmax(outputs["eeg_logits"].detach().cpu(), dim=1)
        eye_probs = F.softmax(outputs["eye_logits"].detach().cpu(), dim=1)
        hrv_probs = F.softmax(outputs["hrv_logits"].detach().cpu(), dim=1)

        B = final_probs.shape[0]
        results = []
        for i in range(B):
            pred_class = int(final_probs[i].argmax().item())
            confidence = float(final_probs[i].max().item())

            class_probs = {
                self.class_names[c]: float(final_probs[i, c].item())
                for c in range(self.n_classes)
            }

            # Arousal / valence from Tanh → [0, 1]
            arousal = float((av[i, 0].item() + 1.0) / 2.0)
            valence = float((av[i, 1].item() + 1.0) / 2.0)

            modality_contributions = {
                "eeg": float(conf_w[i, 0].item()),
                "eye": float(conf_w[i, 1].item()),
                "hrv": float(conf_w[i, 2].item()),
            }

            per_modality = {
                "eeg": self.class_names[int(eeg_probs[i].argmax().item())],
                "eye": self.class_names[int(eye_probs[i].argmax().item())],
                "hrv": self.class_names[int(hrv_probs[i].argmax().item())],
            }

            branch_preds = [
                int(eeg_probs[i].argmax().item()),
                int(eye_probs[i].argmax().item()),
                int(hrv_probs[i].argmax().item()),
            ]
            agreement = sum(p == pred_class for p in branch_preds) / len(branch_preds)

            fusion_weights = {
                "fused": float(dw[0].item()),
                "eeg": float(dw[1].item()),
                "eye": float(dw[2].item()),
                "hrv": float(dw[3].item()),
            }

            results.append({
                "predicted_class": pred_class,
                "class_name": self.class_names[pred_class],
                "confidence": confidence,
                "class_probabilities": class_probs,
                "arousal": arousal,
                "valence": valence,
                "modality_contributions": modality_contributions,
                "per_modality_predictions": per_modality,
                "modality_agreement": agreement,
                "decision_fusion_weights": fusion_weights,
            })

        return results
