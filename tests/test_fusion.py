"""Unit tests for the full CognitiveTwinFusionModel."""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.multimodal_fusion import CognitiveTwinFusionModel


class TestCognitiveTwinFusionModel:
    def setup_method(self):
        self.n_channels = 4
        self.n_freqs = 16
        self.n_time = 64
        self.batch = 2
        self.model = CognitiveTwinFusionModel(
            n_channels_eeg=self.n_channels,
            n_freqs=self.n_freqs,
            n_time=self.n_time,
            n_eye_features=7,
            n_hrv_features=10,
            n_classes=4,
        )
        self.model.eval()

    def _dummy_inputs(self):
        eeg = torch.randn(self.batch, self.n_channels, self.n_freqs, self.n_time)
        eye = torch.randn(self.batch, 64, 7)
        hrv = torch.randn(self.batch, 8, 10)
        return eeg, eye, hrv

    def test_forward_keys(self):
        eeg, eye, hrv = self._dummy_inputs()
        with torch.no_grad():
            out = self.model(eeg, eye, hrv)
        required = [
            "fused_logits", "eeg_logits", "eye_logits", "hrv_logits",
            "arousal_valence", "fused_embedding", "confidence_weights",
            "attention_weights", "final_probs", "decision_weights",
        ]
        for key in required:
            assert key in out, f"Missing key: {key}"

    def test_fused_logits_shape(self):
        eeg, eye, hrv = self._dummy_inputs()
        with torch.no_grad():
            out = self.model(eeg, eye, hrv)
        assert out["fused_logits"].shape == (self.batch, 4)

    def test_final_probs_sum_to_one(self):
        eeg, eye, hrv = self._dummy_inputs()
        with torch.no_grad():
            out = self.model(eeg, eye, hrv)
        probs_sum = out["final_probs"].sum(dim=1)
        torch.testing.assert_close(probs_sum, torch.ones(self.batch), atol=1e-5, rtol=1e-5)

    def test_arousal_valence_shape(self):
        eeg, eye, hrv = self._dummy_inputs()
        with torch.no_grad():
            out = self.model(eeg, eye, hrv)
        assert out["arousal_valence"].shape == (self.batch, 2)

    def test_arousal_valence_in_tanh_range(self):
        eeg, eye, hrv = self._dummy_inputs()
        with torch.no_grad():
            out = self.model(eeg, eye, hrv)
        av = out["arousal_valence"]
        assert av.min() >= -1.0 - 1e-5 and av.max() <= 1.0 + 1e-5

    def test_confidence_weights_shape(self):
        eeg, eye, hrv = self._dummy_inputs()
        with torch.no_grad():
            out = self.model(eeg, eye, hrv)
        assert out["confidence_weights"].shape == (self.batch, 3)

    def test_no_nan_in_outputs(self):
        eeg, eye, hrv = self._dummy_inputs()
        with torch.no_grad():
            out = self.model(eeg, eye, hrv)
        for key, val in out.items():
            if isinstance(val, torch.Tensor):
                assert not torch.isnan(val).any(), f"NaN in {key}"

    def test_declare_state_structure(self):
        eeg, eye, hrv = self._dummy_inputs()
        with torch.no_grad():
            out = self.model(eeg, eye, hrv)
        states = self.model.declare_state(out)
        assert len(states) == self.batch
        required_keys = [
            "predicted_class", "class_name", "confidence",
            "class_probabilities", "arousal", "valence",
            "modality_contributions", "per_modality_predictions",
            "modality_agreement", "decision_fusion_weights",
        ]
        for state in states:
            for key in required_keys:
                assert key in state, f"declare_state missing key: {key}"

    def test_declare_state_predicted_class_valid(self):
        eeg, eye, hrv = self._dummy_inputs()
        with torch.no_grad():
            out = self.model(eeg, eye, hrv)
        states = self.model.declare_state(out)
        for state in states:
            assert 0 <= state["predicted_class"] < 4

    def test_declare_state_confidence_in_range(self):
        eeg, eye, hrv = self._dummy_inputs()
        with torch.no_grad():
            out = self.model(eeg, eye, hrv)
        states = self.model.declare_state(out)
        for state in states:
            assert 0.0 <= state["confidence"] <= 1.0

    def test_declare_state_agreement_in_range(self):
        eeg, eye, hrv = self._dummy_inputs()
        with torch.no_grad():
            out = self.model(eeg, eye, hrv)
        states = self.model.declare_state(out)
        for state in states:
            assert 0.0 <= state["modality_agreement"] <= 1.0
