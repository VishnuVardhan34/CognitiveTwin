"""HRV feature extraction from PPG signals."""

from __future__ import annotations

import numpy as np
import scipy.signal as ss
import scipy.interpolate as si
from typing import Optional, Tuple


class HRVFeatureExtractor:
    """Extract Heart Rate Variability (HRV) features from a PPG signal.

    Produces a 10-dimensional feature vector:
    ``[mean_rr, sdnn, rmssd, pnn50, mean_hr, lf_power, hf_power,
       vlf_power, lf_hf_ratio, total_power]``

    Args:
        sfreq: PPG sampling frequency in Hz. Default 128.
        window_sec: Analysis window duration in seconds. Default 30.
        step_sec: Step between successive windows in seconds. Default 5.
    """

    N_FEATURES: int = 10
    # Minimum inter-beat interval: 0.4 s ≈ 150 bpm maximum heart rate
    MIN_IBI_SEC: float = 0.4
    # Frequency bands (Hz)
    VLF_BAND: Tuple[float, float] = (0.003, 0.04)
    LF_BAND: Tuple[float, float] = (0.04, 0.15)
    HF_BAND: Tuple[float, float] = (0.15, 0.4)

    def __init__(
        self,
        sfreq: float = 128.0,
        window_sec: float = 30.0,
        step_sec: float = 5.0,
    ) -> None:
        self.sfreq = sfreq
        self.window_sec = window_sec
        self.step_sec = step_sec

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _band_power(
        self,
        freqs: np.ndarray,
        psd: np.ndarray,
        fmin: float,
        fmax: float,
    ) -> float:
        """Integrate PSD between fmin and fmax using the trapezoidal rule.

        Args:
            freqs: Frequency axis (Hz).
            psd: Power spectral density.
            fmin: Lower frequency bound.
            fmax: Upper frequency bound.

        Returns:
            Band power as a float.
        """
        mask = (freqs >= fmin) & (freqs <= fmax)
        if mask.sum() < 2:
            return 0.0
        return float(np.trapz(psd[mask], freqs[mask]))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_rr_intervals(self, ppg: np.ndarray) -> np.ndarray:
        """Detect peaks in a PPG signal and return RR intervals in ms.

        Args:
            ppg: PPG waveform of shape (n_samples,).

        Returns:
            RR interval array in milliseconds. Shape (n_intervals,).
        """
        ppg = ppg.astype(np.float64)
        # Minimum distance between peaks: ~40 bpm → 1.5 s
        min_distance = int(self.sfreq * self.MIN_IBI_SEC)
        peaks, _ = ss.find_peaks(
            ppg,
            distance=min_distance,
            height=np.percentile(ppg, 50),
        )
        if len(peaks) < 2:
            return np.array([], dtype=np.float64)
        rr = np.diff(peaks) / self.sfreq * 1000.0  # ms
        # Physiological RR range: 300–2000 ms (30–200 bpm)
        rr = rr[(rr > 300) & (rr < 2000)]
        return rr.astype(np.float64)

    def compute_time_domain(self, rr: np.ndarray) -> np.ndarray:
        """Compute time-domain HRV features.

        Args:
            rr: RR interval array in milliseconds.

        Returns:
            Feature array [mean_rr, sdnn, rmssd, pnn50, mean_hr].
        """
        if len(rr) < 2:
            return np.zeros(5, dtype=np.float32)

        mean_rr = np.mean(rr)
        sdnn = np.std(rr)
        diff_rr = np.diff(rr)
        rmssd = np.sqrt(np.mean(diff_rr ** 2))
        pnn50 = np.mean(np.abs(diff_rr) > 50.0) * 100.0  # percent
        mean_hr = 60000.0 / (mean_rr + 1e-6)

        return np.array([mean_rr, sdnn, rmssd, pnn50, mean_hr], dtype=np.float32)

    def compute_frequency_domain(self, rr: np.ndarray) -> np.ndarray:
        """Compute frequency-domain HRV features via Welch's method.

        Args:
            rr: RR interval array in milliseconds.

        Returns:
            Feature array [lf_power, hf_power, vlf_power, lf_hf_ratio, total_power].
        """
        if len(rr) < 4:
            return np.zeros(5, dtype=np.float32)

        # Interpolate to uniform 4 Hz grid
        t_orig = np.cumsum(rr) / 1000.0  # seconds
        t_uniform = np.arange(t_orig[0], t_orig[-1], 0.25)
        if len(t_uniform) < 4:
            return np.zeros(5, dtype=np.float32)

        interp = si.interp1d(t_orig, rr, kind="linear", fill_value="extrapolate")
        rr_uniform = interp(t_uniform)

        fs_uniform = 4.0  # Hz
        freqs, psd = ss.welch(
            rr_uniform,
            fs=fs_uniform,
            nperseg=min(256, len(rr_uniform)),
        )

        vlf = self._band_power(freqs, psd, *self.VLF_BAND)
        lf = self._band_power(freqs, psd, *self.LF_BAND)
        hf = self._band_power(freqs, psd, *self.HF_BAND)
        total = vlf + lf + hf
        lf_hf = lf / (hf + 1e-8)

        return np.array([lf, hf, vlf, lf_hf, total], dtype=np.float32)

    def extract_features(self, ppg: np.ndarray) -> np.ndarray:
        """Extract the full 10-dim HRV feature vector from a PPG segment.

        Args:
            ppg: PPG waveform of shape (n_samples,).

        Returns:
            Feature vector of shape (10,).
        """
        rr = self.extract_rr_intervals(ppg)
        time_features = self.compute_time_domain(rr)    # (5,)
        freq_features = self.compute_frequency_domain(rr)  # (5,)
        return np.concatenate([time_features, freq_features]).astype(np.float32)

    def extract_windowed(
        self,
        ppg: np.ndarray,
        window_sec: Optional[float] = None,
        step_sec: Optional[float] = None,
    ) -> np.ndarray:
        """Extract HRV features over a sliding window.

        Args:
            ppg: PPG waveform of shape (n_samples,).
            window_sec: Override window duration in seconds.
            step_sec: Override step size in seconds.

        Returns:
            Feature matrix of shape (n_windows, 10).
        """
        win_sec = window_sec if window_sec is not None else self.window_sec
        stp_sec = step_sec if step_sec is not None else self.step_sec

        win_samples = int(win_sec * self.sfreq)
        step_samples = int(stp_sec * self.sfreq)
        n = len(ppg)

        features = []
        start = 0
        while start + win_samples <= n:
            segment = ppg[start : start + win_samples]
            features.append(self.extract_features(segment))
            start += step_samples

        if not features:
            return np.empty((0, self.N_FEATURES), dtype=np.float32)
        return np.stack(features, axis=0)
