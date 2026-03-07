"""Neural network models for CognitiveTwin multimodal fusion."""
from .eeg_branch import EEGWaveletCNN
from .eye_branch import EyeTrackingNet
from .hrv_branch import HRVNet
from .cross_attention import CrossModalAttention
from .confidence_gate import ModalityConfidenceGate
from .multimodal_fusion import CognitiveTwinFusionModel

__all__ = [
    "EEGWaveletCNN",
    "EyeTrackingNet",
    "HRVNet",
    "CrossModalAttention",
    "ModalityConfidenceGate",
    "CognitiveTwinFusionModel",
]
