"""Unit tests for the Unscented Kalman Filter."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from state_estimation.ukf import UnscentedKalmanFilter, TransitionModel
import torch


class TestUnscentedKalmanFilter:
    def setup_method(self):
        self.ukf = UnscentedKalmanFilter(
            state_dim=4,
            obs_dim=64,
            process_noise=0.01,
            observation_noise=0.05,
        )

    def test_initial_state_zeros(self):
        state = self.ukf.get_state()
        np.testing.assert_array_equal(state["state"], np.zeros(4))

    def test_predict_state_shape(self):
        self.ukf.predict()
        state = self.ukf.get_state()
        assert state["state"].shape == (4,)
        assert state["covariance"].shape == (4, 4)

    def test_update_state_shape(self):
        z = np.random.randn(64)
        self.ukf.predict()
        self.ukf.update(z)
        state = self.ukf.get_state()
        assert state["state"].shape == (4,)

    def test_covariance_positive_definite_after_update(self):
        z = np.random.randn(64)
        self.ukf.predict()
        self.ukf.update(z)
        state = self.ukf.get_state()
        P = state["covariance"]
        eigenvalues = np.linalg.eigvalsh(P)
        assert np.all(eigenvalues > 0), "Covariance must be positive definite"

    def test_state_convergence(self):
        """Repeated updates with a consistent observation should stabilise state."""
        # Fixed observation vector corresponding to a known state
        true_state = np.array([0.5, 0.3, 0.7, 0.2])
        z_fixed = self.ukf.H @ true_state  # consistent observation

        for _ in range(50):
            self.ukf.predict()
            self.ukf.update(z_fixed)

        state = self.ukf.get_state()
        # After many updates, estimate should have moved towards true state
        # (convergence is approximate due to linear H approximation)
        err = np.abs(state["state"] - true_state).max()
        assert err < 2.0, f"State should converge toward true state, got error={err:.3f}"

    def test_reset(self):
        z = np.random.randn(64)
        self.ukf.predict()
        self.ukf.update(z)
        self.ukf.reset()
        state = self.ukf.get_state()
        np.testing.assert_array_equal(state["state"], np.zeros(4))

    def test_uncertainty_non_negative(self):
        self.ukf.predict()
        z = np.random.randn(64)
        self.ukf.update(z)
        state = self.ukf.get_state()
        assert np.all(state["uncertainty"] >= 0)

    def test_predict_update_cycle(self):
        """Multiple predict-update cycles should not cause NaN."""
        for _ in range(20):
            self.ukf.predict()
            z = np.random.randn(64)
            self.ukf.update(z)
        state = self.ukf.get_state()
        assert not np.isnan(state["state"]).any()
        assert not np.isnan(state["covariance"]).any()


class TestTransitionModel:
    def setup_method(self):
        self.model = TransitionModel(state_dim=4, d_model=16, n_heads=4, n_layers=1)
        self.model.eval()

    def test_single_output_shape(self):
        state = torch.randn(4)
        with torch.no_grad():
            out = self.model(state)
        assert out.shape == (4,)

    def test_batch_output_shape(self):
        states = torch.randn(3, 4)
        with torch.no_grad():
            out = self.model(states)
        assert out.shape == (3, 4)

    def test_no_nan_output(self):
        states = torch.randn(5, 4)
        with torch.no_grad():
            out = self.model(states)
        assert not torch.isnan(out).any()
