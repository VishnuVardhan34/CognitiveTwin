"""Preprocessing modules for EEG, eye tracking, and HRV signals."""
from .eeg_preprocessor import EEGPreprocessor
from .eye_preprocessor import EyeFeatureExtractor
from .hrv_preprocessor import HRVFeatureExtractor
from .wavelet_transform import compute_scalogram, eeg_to_scalogram_tensor

__all__ = [
    "EEGPreprocessor",
    "EyeFeatureExtractor",
    "HRVFeatureExtractor",
    "compute_scalogram",
    "eeg_to_scalogram_tensor",
]
