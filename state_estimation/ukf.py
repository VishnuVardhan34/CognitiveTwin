"""Unscented Kalman Filter (UKF) for cognitive state tracking."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Transition model
# ---------------------------------------------------------------------------

class TransitionModel(nn.Module):
    """Small Transformer encoder that predicts the next cognitive state.

    Takes a 4-D state vector and returns a predicted 4-D next state.

    Args:
        state_dim: Dimensionality of the state vector. Default 4.
        d_model: Internal Transformer dimension. Default 32.
        n_heads: Number of attention heads. Default 4.
        n_layers: Number of Transformer encoder layers. Default 2.
    """

    def __init__(
        self,
        state_dim: int = 4,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.input_proj = nn.Linear(state_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(d_model, state_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Predict next state.

        Args:
            state: Current state tensor of shape (batch, state_dim) or (state_dim,).

        Returns:
            Predicted next state of the same shape.
        """
        squeeze = state.dim() == 1
        if squeeze:
            state = state.unsqueeze(0)   # (1, state_dim)
        x = self.input_proj(state).unsqueeze(1)  # (B, 1, d_model)
        x = self.transformer(x)                   # (B, 1, d_model)
        out = self.output_proj(x.squeeze(1))      # (B, state_dim)
        if squeeze:
            out = out.squeeze(0)
        return out


# ---------------------------------------------------------------------------
# Unscented Kalman Filter
# ---------------------------------------------------------------------------

class UnscentedKalmanFilter:
    """Unscented Kalman Filter for tracking a 4-D cognitive state.

    State vector: [cognitive_load, arousal, valence, fatigue_index]
    Observations: 64-D fused embedding from the neural model.

    The transition function is either the learned ``TransitionModel`` or a
    simple mean-reverting linear model.  The observation function is a linear
    projection from the state to the 64-D embedding space.

    Args:
        state_dim: Dimensionality of the state vector. Default 4.
        obs_dim: Dimensionality of observations. Default 64.
        alpha: Spread of sigma points. Default 0.001. Small values keep sigma
            points close to the mean, reducing sensitivity to nonlinearities
            in the low-dimensional (4-D) state space.
        beta: Prior distribution parameter. Default 2.0.
        kappa: Secondary scaling parameter. Default 0.0.
        process_noise: Process noise variance (diagonal). Default 0.01.
        observation_noise: Observation noise variance (diagonal). Default 0.05.
        transition_model: Optional learned ``TransitionModel`` instance.
    """

    def __init__(
        self,
        state_dim: int = 4,
        obs_dim: int = 64,
        alpha: float = 0.001,
        beta: float = 2.0,
        kappa: float = 0.0,
        process_noise: float = 0.01,
        observation_noise: float = 0.05,
        transition_model: Optional[TransitionModel] = None,
    ) -> None:
        self.n = state_dim
        self.obs_dim = obs_dim
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

        # UKF scaling parameters
        lam = alpha ** 2 * (state_dim + kappa) - state_dim
        self.lam = lam

        # Sigma point weights
        n_sigma = 2 * state_dim + 1
        self.Wm = np.full(n_sigma, 1.0 / (2.0 * (state_dim + lam)))
        self.Wc = np.full(n_sigma, 1.0 / (2.0 * (state_dim + lam)))
        self.Wm[0] = lam / (state_dim + lam)
        self.Wc[0] = lam / (state_dim + lam) + (1 - alpha ** 2 + beta)

        # Process and observation noise covariances
        self.Q = np.eye(state_dim) * process_noise
        self.R = np.eye(obs_dim) * observation_noise

        # Linear observation matrix H: (obs_dim, state_dim)
        # Maps state → observation space via random projection (fixed)
        rng = np.random.default_rng(seed=42)
        self.H = rng.standard_normal((obs_dim, state_dim)).astype(np.float64)
        self.H /= np.linalg.norm(self.H, axis=0, keepdims=True) + 1e-8

        # State estimate and covariance
        self.x = np.zeros(state_dim, dtype=np.float64)
        self.P = np.eye(state_dim, dtype=np.float64) * 0.1

        self.transition_model = transition_model

    # ------------------------------------------------------------------
    # Sigma points
    # ------------------------------------------------------------------

    def _sigma_points(self) -> np.ndarray:
        """Generate sigma points from current state and covariance.

        Returns:
            Sigma point matrix of shape (2n+1, n).
        """
        n = self.n
        try:
            L = np.linalg.cholesky((n + self.lam) * self.P)
        except np.linalg.LinAlgError:
            # Fallback: regularise P
            L = np.linalg.cholesky((n + self.lam) * (self.P + 1e-6 * np.eye(n)))

        sigma = np.zeros((2 * n + 1, n), dtype=np.float64)
        sigma[0] = self.x
        for i in range(n):
            sigma[i + 1]     = self.x + L[:, i]
            sigma[n + i + 1] = self.x - L[:, i]
        return sigma

    # ------------------------------------------------------------------
    # Transition function
    # ------------------------------------------------------------------

    def _transition(self, sigma: np.ndarray) -> np.ndarray:
        """Propagate sigma points through the transition function.

        Args:
            sigma: Sigma points of shape (2n+1, n).

        Returns:
            Propagated sigma points of the same shape.
        """
        if self.transition_model is not None:
            with torch.no_grad():
                s_t = torch.tensor(sigma, dtype=torch.float32)
                out = self.transition_model(s_t).numpy().astype(np.float64)
            return out

        # Simple mean-reverting linear transition: x_{t+1} ≈ 0.95 * x_t
        return 0.95 * sigma

    # ------------------------------------------------------------------
    # Observation function
    # ------------------------------------------------------------------

    def _observe(self, sigma_pred: np.ndarray) -> np.ndarray:
        """Map predicted sigma points to observation space.

        Args:
            sigma_pred: Propagated sigma points of shape (2n+1, n).

        Returns:
            Predicted observations of shape (2n+1, obs_dim).
        """
        return sigma_pred @ self.H.T  # (2n+1, obs_dim)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self) -> None:
        """Predict step: propagate state and covariance through transition."""
        sigma = self._sigma_points()                       # (2n+1, n)
        sigma_pred = self._transition(sigma)               # (2n+1, n)

        # Predicted mean
        x_pred = (self.Wm[:, None] * sigma_pred).sum(axis=0)  # (n,)

        # Predicted covariance
        diff = sigma_pred - x_pred[None, :]                    # (2n+1, n)
        P_pred = (
            self.Wc[:, None, None] * (diff[:, :, None] @ diff[:, None, :])
        ).sum(axis=0) + self.Q  # (n, n)

        self.x = x_pred
        self.P = P_pred
        self._sigma_pred = sigma_pred  # cache for update step

    def update(self, z: np.ndarray) -> None:
        """Update step: incorporate a new observation.

        Args:
            z: Observation vector of shape (obs_dim,).
        """
        if not hasattr(self, "_sigma_pred"):
            self.predict()

        sigma_pred = self._sigma_pred                      # (2n+1, n)
        z_sigma = self._observe(sigma_pred)                # (2n+1, obs_dim)

        # Predicted observation mean
        z_hat = (self.Wm[:, None] * z_sigma).sum(axis=0)  # (obs_dim,)

        # Innovation covariance
        dz = z_sigma - z_hat[None, :]                      # (2n+1, obs_dim)
        S = (
            self.Wc[:, None, None] * (dz[:, :, None] @ dz[:, None, :])
        ).sum(axis=0) + self.R  # (obs_dim, obs_dim)

        # Cross-covariance
        dx = sigma_pred - self.x[None, :]                  # (2n+1, n)
        Pxz = (
            self.Wc[:, None, None] * (dx[:, :, None] @ dz[:, None, :])
        ).sum(axis=0)  # (n, obs_dim)

        # Kalman gain
        K = Pxz @ np.linalg.inv(S)                        # (n, obs_dim)

        # State update
        innovation = z - z_hat                             # (obs_dim,)
        self.x = self.x + K @ innovation                  # (n,)
        self.P = self.P - K @ S @ K.T                     # (n, n)

        # Ensure symmetry and positive-definiteness
        self.P = (self.P + self.P.T) / 2.0
        self.P += 1e-6 * np.eye(self.n)

    def get_state(self) -> Dict[str, np.ndarray]:
        """Return the current state estimate and uncertainty.

        Returns:
            Dictionary with:
                - ``state``: (n,) current state estimate
                - ``covariance``: (n, n) state covariance
                - ``uncertainty``: (n,) diagonal of covariance (std dev)
        """
        return {
            "state": self.x.copy(),
            "covariance": self.P.copy(),
            "uncertainty": np.sqrt(np.maximum(np.diag(self.P), 0.0)),
        }

    def reset(self) -> None:
        """Reset filter to initial state."""
        self.x = np.zeros(self.n, dtype=np.float64)
        self.P = np.eye(self.n, dtype=np.float64) * 0.1
        if hasattr(self, "_sigma_pred"):
            del self._sigma_pred
