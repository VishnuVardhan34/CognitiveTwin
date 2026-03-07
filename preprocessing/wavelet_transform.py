"""Continuous Wavelet Transform (CWT) scalogram computation."""

from __future__ import annotations

import numpy as np
import pywt


def compute_scalogram(
    eeg_channel: np.ndarray,
    sfreq: float = 128.0,
    wavelet: str = "cmor1.5-1.0",
    n_freqs: int = 64,
) -> np.ndarray:
    """Compute a CWT Morlet scalogram for a single EEG channel.

    Args:
        eeg_channel: 1-D EEG signal of shape (n_samples,).
        sfreq: Sampling frequency in Hz. Default 128.
        wavelet: PyWavelets CWT wavelet name. Default ``'cmor1.5-1.0'``
            (complex Morlet).
        n_freqs: Number of frequency bins (scales). Default 64.

    Returns:
        Power scalogram of shape (n_freqs, n_samples), float32.
    """
    n_samples = len(eeg_channel)

    # Map frequency range 0.5–45 Hz to scales
    f_min, f_max = 0.5, 45.0
    freqs = np.linspace(f_min, f_max, n_freqs)

    # Central frequency of the wavelet
    central_freq = pywt.central_frequency(wavelet)
    scales = central_freq * sfreq / freqs  # convert freq → scale

    coeffs, _ = pywt.cwt(eeg_channel.astype(np.float64), scales, wavelet, 1.0 / sfreq)
    power = np.abs(coeffs) ** 2  # (n_freqs, n_samples)
    return power.astype(np.float32)


def eeg_to_scalogram_tensor(
    eeg: np.ndarray,
    sfreq: float = 128.0,
    n_freqs: int = 64,
    wavelet: str = "cmor1.5-1.0",
) -> np.ndarray:
    """Convert a multi-channel EEG array to a scalogram tensor.

    Args:
        eeg: EEG array of shape (n_channels, n_samples).
        sfreq: Sampling frequency in Hz. Default 128.
        n_freqs: Number of frequency bins. Default 64.
        wavelet: PyWavelets CWT wavelet name.

    Returns:
        Scalogram tensor of shape (n_channels, n_freqs, n_samples), float32.
    """
    n_channels, n_samples = eeg.shape
    scalograms = np.zeros((n_channels, n_freqs, n_samples), dtype=np.float32)
    for ch in range(n_channels):
        scalograms[ch] = compute_scalogram(eeg[ch], sfreq=sfreq, wavelet=wavelet, n_freqs=n_freqs)
    return scalograms
