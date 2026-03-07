"""Eye-tracking feature extraction from gaze, EOG, or facial landmark data."""

from __future__ import annotations

import numpy as np
from typing import Optional


class EyeFeatureExtractor:
    """Extract eye-tracking features from multiple input modalities.

    Produces a 7-dimensional feature vector per time step:
    ``[gaze_x, gaze_y, pupil_detrended, pupil_rate, velocity,
       fixation_flag, saccade_flag]``

    Args:
        sfreq: Sampling frequency in Hz. Default 120.
        velocity_threshold: Velocity threshold (deg/s or a.u.) to classify
            saccades. Default 30.0.
        ear_threshold: Eye Aspect Ratio threshold for blink detection.
            Default 0.2.
    """

    N_FEATURES: int = 7
    DETREND_WINDOW_SEC: float = 2.0  # moving-median window for pupil detrending

    def __init__(
        self,
        sfreq: float = 120.0,
        velocity_threshold: float = 30.0,
        ear_threshold: float = 0.2,
    ) -> None:
        self.sfreq = sfreq
        self.velocity_threshold = velocity_threshold
        self.ear_threshold = ear_threshold

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_velocity(
        self, x: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        """Compute gaze velocity as Euclidean speed between consecutive frames.

        Args:
            x: Horizontal gaze signal (n_samples,).
            y: Vertical gaze signal (n_samples,).

        Returns:
            Velocity array of shape (n_samples,).
        """
        dx = np.diff(x, prepend=x[0])
        dy = np.diff(y, prepend=y[0])
        velocity = np.sqrt(dx ** 2 + dy ** 2) * self.sfreq  # per second
        return velocity.astype(np.float32)

    def _detect_fixation_saccade(
        self, velocity: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Classify each sample as fixation or saccade based on velocity.

        Args:
            velocity: Velocity array (n_samples,).

        Returns:
            Tuple of (fixation_flag, saccade_flag), each shape (n_samples,).
        """
        saccade = (velocity > self.velocity_threshold).astype(np.float32)
        fixation = 1.0 - saccade
        return fixation, saccade

    def detrend_pupil(self, signal: np.ndarray) -> np.ndarray:
        """Remove slow luminance-driven drift via moving-median subtraction.

        Args:
            signal: Pupil diameter signal (n_samples,).

        Returns:
            Detrended signal of the same shape.
        """
        n = len(signal)
        win = max(1, int(self.sfreq * self.DETREND_WINDOW_SEC))
        out = np.empty(n, dtype=np.float32)
        for i in range(n):
            lo = max(0, i - win // 2)
            hi = min(n, i + win // 2 + 1)
            out[i] = signal[i] - np.median(signal[lo:hi])
        return out

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_from_gaze(
        self,
        gaze_x: np.ndarray,
        gaze_y: np.ndarray,
        pupil_l: np.ndarray,
        pupil_r: np.ndarray,
    ) -> np.ndarray:
        """Extract 7-dim feature matrix from raw gaze and pupil signals.

        Args:
            gaze_x: Horizontal gaze position (n_samples,).
            gaze_y: Vertical gaze position (n_samples,).
            pupil_l: Left pupil diameter (n_samples,).
            pupil_r: Right pupil diameter (n_samples,).

        Returns:
            Feature matrix of shape (n_samples, 7).
        """
        gaze_x = gaze_x.astype(np.float32)
        gaze_y = gaze_y.astype(np.float32)
        pupil = ((pupil_l + pupil_r) / 2.0).astype(np.float32)

        pupil_detrended = self.detrend_pupil(pupil)
        pupil_rate = np.diff(pupil, prepend=pupil[0]) * self.sfreq

        velocity = self._compute_velocity(gaze_x, gaze_y)
        fixation, saccade = self._detect_fixation_saccade(velocity)

        # Z-score normalise continuous features
        def _zscore(a: np.ndarray) -> np.ndarray:
            return ((a - a.mean()) / (a.std() + 1e-8)).astype(np.float32)

        features = np.stack(
            [
                _zscore(gaze_x),
                _zscore(gaze_y),
                _zscore(pupil_detrended),
                _zscore(pupil_rate),
                _zscore(velocity),
                fixation,
                saccade,
            ],
            axis=1,
        )
        return features  # (n_samples, 7)

    def extract_from_deap_eog(
        self,
        eog_h: np.ndarray,
        eog_v: np.ndarray,
    ) -> np.ndarray:
        """Extract eye features from DEAP EOG channels as gaze proxy.

        Uses EOG horizontal / vertical as pseudo gaze-x / gaze-y, and
        substitutes constant pupil signals (EOG does not contain pupil data).

        Args:
            eog_h: Horizontal EOG channel (n_samples,).
            eog_v: Vertical EOG channel (n_samples,).

        Returns:
            Feature matrix of shape (n_samples, 7).
        """
        dummy_pupil = np.zeros_like(eog_h, dtype=np.float32)
        return self.extract_from_gaze(eog_h, eog_v, dummy_pupil, dummy_pupil)

    def extract_from_landmarks(
        self, landmarks: np.ndarray
    ) -> np.ndarray:
        """Extract eye features from 68-point facial landmarks (DROZY).

        Computes EAR (eye aspect ratio), PERCLOS (percentage of eye closure),
        and blink rate.  Pads the 3 facial features to the 7-feature standard
        with zeros to maintain a uniform feature dimensionality.

        Args:
            landmarks: Array of shape (n_frames, 68, 2) with (x, y) coords.

        Returns:
            Feature matrix of shape (n_frames, 7).
        """
        def _ear(pts: np.ndarray) -> np.ndarray:
            A = np.linalg.norm(pts[:, 1] - pts[:, 5], axis=-1)
            B = np.linalg.norm(pts[:, 2] - pts[:, 4], axis=-1)
            C = np.linalg.norm(pts[:, 0] - pts[:, 3], axis=-1)
            return (A + B) / (2.0 * C + 1e-6)

        n = landmarks.shape[0]
        left_pts = landmarks[:, 36:42, :]
        right_pts = landmarks[:, 42:48, :]
        ear = (_ear(left_pts) + _ear(right_pts)) / 2.0  # (n,)

        closed = (ear < self.ear_threshold).astype(np.float32)
        win = min(60, n)
        perclos = np.convolve(closed, np.ones(win) / win, mode="same")

        fps = int(self.sfreq)
        blinks = np.maximum(np.diff(closed.astype(np.int32), prepend=0), 0).astype(np.float32)
        blink_rate = np.convolve(blinks, np.ones(fps) / fps, mode="same")

        padding = np.zeros((n, 4), dtype=np.float32)
        features = np.concatenate(
            [
                ear[:, None],
                perclos[:, None],
                blink_rate[:, None],
                padding,
            ],
            axis=1,
        )
        return features  # (n_frames, 7)
