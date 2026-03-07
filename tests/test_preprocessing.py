"""Unit tests for preprocessing modules."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preprocessing.eeg_preprocessor import EEGPreprocessor
from preprocessing.eye_preprocessor import EyeFeatureExtractor
from preprocessing.hrv_preprocessor import HRVFeatureExtractor
from preprocessing.wavelet_transform import compute_scalogram, eeg_to_scalogram_tensor


# ---------------------------------------------------------------------------
# EEG Preprocessor
# ---------------------------------------------------------------------------

class TestEEGPreprocessor:
    def setup_method(self):
        self.sfreq = 128
        self.n_channels = 32
        self.n_samples = 512
        self.prep = EEGPreprocessor(sfreq=self.sfreq, n_channels=self.n_channels)
        self.eeg = np.random.randn(self.n_channels, self.n_samples).astype(np.float32)

    def test_bandpass_filter_shape(self):
        filtered = self.prep.bandpass_filter(self.eeg)
        assert filtered.shape == self.eeg.shape, "Bandpass filter must preserve shape"

    def test_bandpass_filter_dtype(self):
        filtered = self.prep.bandpass_filter(self.eeg)
        assert filtered.dtype == np.float32

    def test_bandpass_filter_attenuates_dc(self):
        # A DC signal should be strongly attenuated by the bandpass filter
        dc_signal = np.ones((1, self.n_samples), dtype=np.float32)
        filtered = self.prep.bandpass_filter(dc_signal)
        assert np.abs(filtered).mean() < 0.1, "DC component should be attenuated"

    def test_normalize_zero_mean(self):
        normed = self.prep.normalize(self.eeg)
        channel_means = normed.mean(axis=1)
        np.testing.assert_allclose(channel_means, 0.0, atol=1e-5)

    def test_normalize_unit_std(self):
        normed = self.prep.normalize(self.eeg)
        channel_stds = normed.std(axis=1)
        np.testing.assert_allclose(channel_stds, 1.0, atol=1e-3)

    def test_normalize_shape(self):
        normed = self.prep.normalize(self.eeg)
        assert normed.shape == self.eeg.shape

    def test_ica_shape(self):
        filtered = self.prep.bandpass_filter(self.eeg)
        cleaned = self.prep.remove_artifacts_ica(filtered)
        assert cleaned.shape == filtered.shape

    def test_process_pipeline_shape(self):
        result = self.prep.process(self.eeg)
        assert result.shape == self.eeg.shape


# ---------------------------------------------------------------------------
# Eye Feature Extractor
# ---------------------------------------------------------------------------

class TestEyeFeatureExtractor:
    def setup_method(self):
        self.n_samples = 240
        self.extractor = EyeFeatureExtractor(sfreq=120.0)
        rng = np.random.default_rng(0)
        self.gaze_x = rng.standard_normal(self.n_samples).astype(np.float32)
        self.gaze_y = rng.standard_normal(self.n_samples).astype(np.float32)
        self.pupil  = np.abs(rng.standard_normal(self.n_samples)).astype(np.float32) + 2.0

    def test_extract_from_gaze_shape(self):
        feats = self.extractor.extract_from_gaze(
            self.gaze_x, self.gaze_y, self.pupil, self.pupil
        )
        assert feats.shape == (self.n_samples, 7)

    def test_extract_from_gaze_dtype(self):
        feats = self.extractor.extract_from_gaze(
            self.gaze_x, self.gaze_y, self.pupil, self.pupil
        )
        assert feats.dtype == np.float32

    def test_fixation_saccade_binary(self):
        feats = self.extractor.extract_from_gaze(
            self.gaze_x, self.gaze_y, self.pupil, self.pupil
        )
        fixation = feats[:, 5]
        saccade  = feats[:, 6]
        assert np.all(np.isin(fixation, [0.0, 1.0])), "Fixation must be binary"
        assert np.all(np.isin(saccade, [0.0, 1.0])), "Saccade must be binary"
        np.testing.assert_allclose(fixation + saccade, 1.0, atol=1e-6)

    def test_extract_from_deap_eog_shape(self):
        feats = self.extractor.extract_from_deap_eog(self.gaze_x, self.gaze_y)
        assert feats.shape == (self.n_samples, 7)

    def test_detrend_pupil_shape(self):
        signal = np.random.randn(self.n_samples).astype(np.float32)
        detrended = self.extractor.detrend_pupil(signal)
        assert detrended.shape == signal.shape


# ---------------------------------------------------------------------------
# HRV Feature Extractor
# ---------------------------------------------------------------------------

class TestHRVFeatureExtractor:
    def setup_method(self):
        self.sfreq = 128.0
        self.extractor = HRVFeatureExtractor(sfreq=self.sfreq)
        # Synthetic PPG: sinusoidal at ~1 Hz (60 bpm) for 60 seconds
        t = np.arange(0, 60, 1.0 / self.sfreq)
        self.ppg = (np.sin(2 * np.pi * 1.0 * t) + 0.5 * np.random.randn(len(t))).astype(np.float32)

    def test_extract_rr_returns_array(self):
        rr = self.extractor.extract_rr_intervals(self.ppg)
        assert isinstance(rr, np.ndarray)

    def test_rr_physiological_range(self):
        rr = self.extractor.extract_rr_intervals(self.ppg)
        if len(rr) > 0:
            assert rr.min() > 300 and rr.max() < 2000, "RR intervals out of physiological range"

    def test_time_domain_shape(self):
        rr = self.extractor.extract_rr_intervals(self.ppg)
        td = self.extractor.compute_time_domain(rr)
        assert td.shape == (5,)

    def test_frequency_domain_shape(self):
        rr = self.extractor.extract_rr_intervals(self.ppg)
        fd = self.extractor.compute_frequency_domain(rr)
        assert fd.shape == (5,)

    def test_extract_features_shape(self):
        feats = self.extractor.extract_features(self.ppg)
        assert feats.shape == (10,)
        assert feats.dtype == np.float32

    def test_extract_windowed_shape(self):
        feats = self.extractor.extract_windowed(self.ppg, window_sec=30.0, step_sec=5.0)
        assert feats.ndim == 2
        assert feats.shape[1] == 10


# ---------------------------------------------------------------------------
# Wavelet Transform
# ---------------------------------------------------------------------------

class TestWaveletTransform:
    def setup_method(self):
        self.sfreq = 128.0
        self.n_samples = 256
        self.n_freqs = 32
        self.n_channels = 4
        rng = np.random.default_rng(1)
        self.signal = rng.standard_normal(self.n_samples).astype(np.float32)
        self.eeg = rng.standard_normal((self.n_channels, self.n_samples)).astype(np.float32)

    def test_scalogram_shape(self):
        scalo = compute_scalogram(self.signal, sfreq=self.sfreq, n_freqs=self.n_freqs)
        assert scalo.shape == (self.n_freqs, self.n_samples)

    def test_scalogram_dtype(self):
        scalo = compute_scalogram(self.signal, sfreq=self.sfreq, n_freqs=self.n_freqs)
        assert scalo.dtype == np.float32

    def test_scalogram_non_negative(self):
        scalo = compute_scalogram(self.signal, sfreq=self.sfreq, n_freqs=self.n_freqs)
        assert np.all(scalo >= 0), "Power scalogram must be non-negative"

    def test_eeg_to_scalogram_tensor_shape(self):
        tensor = eeg_to_scalogram_tensor(self.eeg, sfreq=self.sfreq, n_freqs=self.n_freqs)
        assert tensor.shape == (self.n_channels, self.n_freqs, self.n_samples)

    def test_eeg_to_scalogram_tensor_dtype(self):
        tensor = eeg_to_scalogram_tensor(self.eeg, sfreq=self.sfreq, n_freqs=self.n_freqs)
        assert tensor.dtype == np.float32
