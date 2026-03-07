"""EEG preprocessing: bandpass filter, ICA artifact removal, z-score normalisation."""

from __future__ import annotations

import numpy as np
import scipy.signal as ss
from typing import Optional


class EEGPreprocessor:
    """Preprocess raw EEG signals.

    Applies an FIR bandpass filter, removes ocular/muscle artefacts via
    Independent Component Analysis (ICA), and z-score normalises each channel.

    Args:
        sfreq: Sampling frequency in Hz. Default 128.
        n_channels: Number of EEG channels. Default 32.
        bandpass_low: Lower cutoff frequency (Hz). Default 0.5.
        bandpass_high: Upper cutoff frequency (Hz). Default 45.0.
        ica_components: Number of ICA components to retain. Default 14.
    """

    def __init__(
        self,
        sfreq: float = 128.0,
        n_channels: int = 32,
        bandpass_low: float = 0.5,
        bandpass_high: float = 45.0,
        ica_components: int = 14,
    ) -> None:
        self.sfreq = sfreq
        self.n_channels = n_channels
        self.bandpass_low = bandpass_low
        self.bandpass_high = bandpass_high
        self.ica_components = ica_components

        # Pre-compute FIR filter coefficients
        self._b = self._design_fir_bandpass()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _design_fir_bandpass(self) -> np.ndarray:
        """Design a linear-phase FIR bandpass filter.

        Returns:
            1-D filter coefficient array ``b``.
        """
        nyq = self.sfreq / 2.0
        numtaps = int(self.sfreq) | 1  # ensure odd length
        b = ss.firwin(
            numtaps,
            [self.bandpass_low / nyq, self.bandpass_high / nyq],
            pass_zero=False,
            window="hamming",
        )
        return b

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def bandpass_filter(self, eeg: np.ndarray) -> np.ndarray:
        """Apply FIR bandpass filter to each EEG channel.

        Args:
            eeg: EEG array of shape ``(n_channels, n_samples)``.

        Returns:
            Filtered EEG of the same shape.
        """
        if eeg.ndim == 1:
            return ss.filtfilt(self._b, [1.0], eeg).astype(np.float32)
        filtered = np.zeros_like(eeg)
        for ch in range(eeg.shape[0]):
            filtered[ch] = ss.filtfilt(self._b, [1.0], eeg[ch])
        return filtered.astype(np.float32)

    def remove_artifacts_ica(self, eeg: np.ndarray) -> np.ndarray:
        """Remove artefacts using FastICA.

        Decomposes the signal into independent components (ICs), flags
        components with high kurtosis (likely artefacts), zeroes them out,
        and reconstructs the signal.

        Args:
            eeg: EEG array of shape ``(n_channels, n_samples)``.

        Returns:
            Artefact-reduced EEG of the same shape.
        """
        try:
            from sklearn.decomposition import FastICA
        except ImportError:
            return eeg  # skip ICA if sklearn unavailable

        n_ch, n_samples = eeg.shape
        n_comp = min(self.ica_components, n_ch)

        ica = FastICA(n_components=n_comp, random_state=42, max_iter=500)
        # ICA expects (n_samples, n_features)
        sources = ica.fit_transform(eeg.T)  # (n_samples, n_comp)

        # Detect artefact components by excess kurtosis threshold
        kurtoses = np.array(
            [
                float(np.mean((sources[:, i] - sources[:, i].mean()) ** 4)
                      / (sources[:, i].std() ** 4 + 1e-8))
                for i in range(n_comp)
            ]
        )
        artefact_mask = kurtoses > 5.0  # threshold (empirical)
        sources[:, artefact_mask] = 0.0

        # Reconstruct: mixing_matrix is (n_features, n_components)
        mixing = ica.mixing_  # (n_ch, n_comp)
        reconstructed = (sources @ mixing.T + ica.mean_).T  # (n_ch, n_samples)
        return reconstructed.astype(np.float32)

    def normalize(self, eeg: np.ndarray) -> np.ndarray:
        """Z-score normalise each EEG channel independently.

        Args:
            eeg: EEG array of shape ``(n_channels, n_samples)``.

        Returns:
            Normalised EEG of the same shape.
        """
        mean = eeg.mean(axis=1, keepdims=True)
        std = eeg.std(axis=1, keepdims=True) + 1e-8
        return ((eeg - mean) / std).astype(np.float32)

    def process(self, eeg: np.ndarray) -> np.ndarray:
        """Full preprocessing pipeline: filter → ICA → normalise.

        Args:
            eeg: Raw EEG array of shape ``(n_channels, n_samples)``.

        Returns:
            Preprocessed EEG of the same shape.
        """
        eeg = self.bandpass_filter(eeg)
        eeg = self.remove_artifacts_ica(eeg)
        eeg = self.normalize(eeg)
        return eeg
