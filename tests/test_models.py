"""Unit tests for individual neural network branch models."""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.eeg_branch import EEGWaveletCNN
from models.eye_branch import EyeTrackingNet
from models.hrv_branch import HRVNet
from models.cross_attention import CrossModalAttention
from models.confidence_gate import ModalityConfidenceGate


class TestEEGWaveletCNN:
    def setup_method(self):
        self.n_channels = 4
        self.n_freqs = 16
        self.n_time = 64
        self.batch = 2
        self.model = EEGWaveletCNN(
            n_channels=self.n_channels,
            n_freqs=self.n_freqs,
            n_time=self.n_time,
            embedding_dim=64,
        )
        self.model.eval()

    def test_output_shape(self):
        x = torch.randn(self.batch, self.n_channels, self.n_freqs, self.n_time)
        with torch.no_grad():
            out = self.model(x)
        assert out.shape == (self.batch, 64)

    def test_output_dtype(self):
        x = torch.randn(self.batch, self.n_channels, self.n_freqs, self.n_time)
        with torch.no_grad():
            out = self.model(x)
        assert out.dtype == torch.float32

    def test_no_nan(self):
        x = torch.randn(self.batch, self.n_channels, self.n_freqs, self.n_time)
        with torch.no_grad():
            out = self.model(x)
        assert not torch.isnan(out).any()

    def test_batch_invariance(self):
        """Single sample and batched should give the same result."""
        x = torch.randn(1, self.n_channels, self.n_freqs, self.n_time)
        with torch.no_grad():
            out1 = self.model(x)
            out2 = self.model(x.repeat(3, 1, 1, 1))
        torch.testing.assert_close(out1, out2[:1], atol=1e-4, rtol=1e-4)


class TestEyeTrackingNet:
    def setup_method(self):
        self.batch = 2
        self.seq_len = 64
        self.n_features = 7
        self.model = EyeTrackingNet(n_features=self.n_features, embedding_dim=64)
        self.model.eval()

    def test_output_shape(self):
        x = torch.randn(self.batch, self.seq_len, self.n_features)
        with torch.no_grad():
            out = self.model(x)
        assert out.shape == (self.batch, 64)

    def test_no_nan(self):
        x = torch.randn(self.batch, self.seq_len, self.n_features)
        with torch.no_grad():
            out = self.model(x)
        assert not torch.isnan(out).any()


class TestHRVNet:
    def setup_method(self):
        self.batch = 2
        self.seq_len = 8
        self.n_features = 10
        self.model = HRVNet(n_features=self.n_features, embedding_dim=32)
        self.model.eval()

    def test_output_shape(self):
        x = torch.randn(self.batch, self.seq_len, self.n_features)
        with torch.no_grad():
            out = self.model(x)
        assert out.shape == (self.batch, 32)

    def test_no_nan(self):
        x = torch.randn(self.batch, self.seq_len, self.n_features)
        with torch.no_grad():
            out = self.model(x)
        assert not torch.isnan(out).any()


class TestCrossModalAttention:
    def setup_method(self):
        self.batch = 2
        self.model = CrossModalAttention(eeg_dim=64, eye_dim=64, hrv_dim=32, common_dim=64, n_heads=4)
        self.model.eval()

    def test_tokens_shape(self):
        eeg = torch.randn(self.batch, 64)
        eye = torch.randn(self.batch, 64)
        hrv = torch.randn(self.batch, 32)
        with torch.no_grad():
            tokens, attn = self.model(eeg, eye, hrv)
        assert tokens.shape == (self.batch, 3, 64)

    def test_attention_weights_shape(self):
        eeg = torch.randn(self.batch, 64)
        eye = torch.randn(self.batch, 64)
        hrv = torch.randn(self.batch, 32)
        with torch.no_grad():
            tokens, attn = self.model(eeg, eye, hrv)
        assert attn.shape == (self.batch, 3, 3)


class TestModalityConfidenceGate:
    def setup_method(self):
        self.batch = 2
        self.gate = ModalityConfidenceGate(eeg_dim=64, eye_dim=64, hrv_dim=32)
        self.gate.eval()

    def test_output_shape(self):
        eeg = torch.randn(self.batch, 64)
        eye = torch.randn(self.batch, 64)
        hrv = torch.randn(self.batch, 32)
        with torch.no_grad():
            weights = self.gate(eeg, eye, hrv)
        assert weights.shape == (self.batch, 3)

    def test_weights_sum_to_one(self):
        eeg = torch.randn(self.batch, 64)
        eye = torch.randn(self.batch, 64)
        hrv = torch.randn(self.batch, 32)
        with torch.no_grad():
            weights = self.gate(eeg, eye, hrv)
        torch.testing.assert_close(weights.sum(dim=1), torch.ones(self.batch), atol=1e-5, rtol=1e-5)

    def test_weights_positive(self):
        eeg = torch.randn(self.batch, 64)
        eye = torch.randn(self.batch, 64)
        hrv = torch.randn(self.batch, 32)
        with torch.no_grad():
            weights = self.gate(eeg, eye, hrv)
        assert (weights >= 0).all()
