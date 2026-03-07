"""End-to-end real-time cognitive load estimation pipeline."""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocessing.eeg_preprocessor import EEGPreprocessor
from preprocessing.eye_preprocessor import EyeFeatureExtractor
from preprocessing.hrv_preprocessor import HRVFeatureExtractor
from preprocessing.wavelet_transform import eeg_to_scalogram_tensor
from state_estimation.ukf import UnscentedKalmanFilter
from backend.websocket_server import DigitalTwinServer


class CognitiveTwinPipeline:
    """End-to-end pipeline for real-time cognitive load estimation.

    Integrates:
    1. Signal preprocessing (EEG, eye tracking, HRV)
    2. Neural model inference (via ONNX or PyTorch)
    3. Unscented Kalman Filter state smoothing
    4. WebSocket broadcast

    Args:
        onnx_path: Path to ONNX model file. If None, uses PyTorch model.
        checkpoint_path: Path to PyTorch checkpoint (used if onnx_path is None).
        sfreq_eeg: EEG sampling frequency. Default 128.
        n_channels_eeg: Number of EEG channels. Default 32.
        n_freqs: CWT frequency bins. Default 64.
        n_time: Time steps per window. Default 256.
        server_host: WebSocket server host. Default 'localhost'.
        server_port: WebSocket server port. Default 8765.
        max_latency_ms: Maximum acceptable latency in ms. Default 200.
    """

    def __init__(
        self,
        onnx_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        sfreq_eeg: float = 128.0,
        n_channels_eeg: int = 32,
        n_freqs: int = 64,
        n_time: int = 256,
        server_host: str = "localhost",
        server_port: int = 8765,
        max_latency_ms: float = 200.0,
    ) -> None:
        self.sfreq_eeg = sfreq_eeg
        self.n_channels_eeg = n_channels_eeg
        self.n_freqs = n_freqs
        self.n_time = n_time
        self.max_latency_ms = max_latency_ms

        # Preprocessors
        self.eeg_preprocessor = EEGPreprocessor(
            sfreq=sfreq_eeg, n_channels=n_channels_eeg
        )
        self.eye_extractor = EyeFeatureExtractor(sfreq=sfreq_eeg)
        self.hrv_extractor = HRVFeatureExtractor(sfreq=sfreq_eeg)

        # UKF
        self.ukf = UnscentedKalmanFilter(state_dim=4, obs_dim=64)

        # Load model
        self._model = None
        self._ort_session = None
        if onnx_path and Path(onnx_path).exists():
            self._load_onnx(onnx_path)
        elif checkpoint_path and Path(checkpoint_path).exists():
            self._load_pytorch(checkpoint_path, n_channels_eeg, n_freqs, n_time)

        # WebSocket server (optional)
        self.server = DigitalTwinServer(host=server_host, port=server_port)

    def _load_onnx(self, path: str) -> None:
        """Load ONNX model for inference."""
        try:
            import onnxruntime as ort
            self._ort_session = ort.InferenceSession(path)
            print(f"[Pipeline] Loaded ONNX model from {path}")
        except Exception as e:
            print(f"[Pipeline] Failed to load ONNX: {e}")

    def _load_pytorch(
        self, path: str, n_ch: int, n_freqs: int, n_time: int
    ) -> None:
        """Load PyTorch model for inference."""
        import torch
        from models.multimodal_fusion import CognitiveTwinFusionModel

        model = CognitiveTwinFusionModel(
            n_channels_eeg=n_ch, n_freqs=n_freqs, n_time=n_time
        )
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        self._model = model
        print(f"[Pipeline] Loaded PyTorch model from {path}")

    def _infer_pytorch(
        self,
        scalogram: np.ndarray,
        eye_seq: np.ndarray,
        hrv_seq: np.ndarray,
    ) -> Dict[str, Any]:
        """Run PyTorch model inference."""
        import torch
        with torch.no_grad():
            s = torch.tensor(scalogram[None], dtype=torch.float32)
            e = torch.tensor(eye_seq[None], dtype=torch.float32)
            h = torch.tensor(hrv_seq[None], dtype=torch.float32)
            outputs = self._model(s, e, h)
        states = self._model.declare_state(outputs)
        emb = outputs["fused_embedding"][0].numpy()
        return states[0], emb

    def _infer_onnx(
        self,
        scalogram: np.ndarray,
        eye_seq: np.ndarray,
        hrv_seq: np.ndarray,
    ) -> Dict[str, Any]:
        """Run ONNX model inference."""
        import torch
        import torch.nn.functional as F

        ort_inputs = {
            "eeg_scalogram": scalogram[None].astype(np.float32),
            "eye_seq": eye_seq[None].astype(np.float32),
            "hrv_seq": hrv_seq[None].astype(np.float32),
        }
        ort_outputs = self._ort_session.run(None, ort_inputs)
        # Output order: fused_logits(0), eeg(1), eye(2), hrv(3), av(4), emb(5), conf(6), attn(7), final_probs(8), dw(9)
        final_probs = ort_outputs[8][0]
        av = ort_outputs[4][0]
        emb = ort_outputs[5][0]
        conf_w = ort_outputs[6][0]
        pred_class = int(np.argmax(final_probs))
        class_names = ["Underload", "Optimal", "Overload", "Fatigue"]
        state = {
            "predicted_class": pred_class,
            "class_name": class_names[pred_class],
            "confidence": float(np.max(final_probs)),
            "class_probabilities": {class_names[i]: float(final_probs[i]) for i in range(4)},
            "arousal": float((av[0] + 1.0) / 2.0),
            "valence": float((av[1] + 1.0) / 2.0),
            "modality_contributions": {
                "eeg": float(conf_w[0]),
                "eye": float(conf_w[1]),
                "hrv": float(conf_w[2]),
            },
        }
        return state, emb

    def process_window(
        self,
        eeg: np.ndarray,
        eog_h: np.ndarray,
        eog_v: np.ndarray,
        ppg: np.ndarray,
    ) -> Dict[str, Any]:
        """Process a single time window end-to-end.

        Args:
            eeg: EEG array of shape (n_channels, n_samples).
            eog_h: Horizontal EOG (n_samples,).
            eog_v: Vertical EOG (n_samples,).
            ppg: PPG signal (n_samples,).

        Returns:
            Cognitive state dictionary from ``declare_state``.
        """
        t_start = time.perf_counter()

        # --- Preprocessing ---
        eeg_filtered = self.eeg_preprocessor.bandpass_filter(eeg)
        eeg_norm = self.eeg_preprocessor.normalize(eeg_filtered)

        scalogram = eeg_to_scalogram_tensor(
            eeg_norm, sfreq=self.sfreq_eeg, n_freqs=self.n_freqs
        )
        # Pad/trim time axis
        T = scalogram.shape[-1]
        if T > self.n_time:
            scalogram = scalogram[:, :, :self.n_time]
        elif T < self.n_time:
            pad = np.zeros((*scalogram.shape[:2], self.n_time - T), dtype=np.float32)
            scalogram = np.concatenate([scalogram, pad], axis=-1)

        eye_seq = self.eye_extractor.extract_from_deap_eog(eog_h, eog_v)
        hrv_features = self.hrv_extractor.extract_features(ppg)
        # 8 steps match the HRV LSTM encoder's expected sequence length.
        hrv_seq = np.tile(hrv_features, (8, 1))  # (8, 10)

        # --- Model inference ---
        if self._ort_session is not None:
            state, embedding = self._infer_onnx(scalogram, eye_seq, hrv_seq)
        elif self._model is not None:
            state, embedding = self._infer_pytorch(scalogram, eye_seq, hrv_seq)
        else:
            # Fallback: return dummy state
            state = {
                "predicted_class": 1,
                "class_name": "Optimal",
                "confidence": 0.5,
                "arousal": 0.5,
                "valence": 0.5,
            }
            embedding = np.zeros(64, dtype=np.float32)

        # --- UKF predict + update ---
        self.ukf.predict()
        self.ukf.update(embedding)
        ukf_state = self.ukf.get_state()

        state["ukf_state"] = ukf_state["state"].tolist()
        state["ukf_uncertainty"] = ukf_state["uncertainty"].tolist()

        # --- Broadcast via WebSocket ---
        self.server.broadcast_state(state)

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        state["processing_time_ms"] = elapsed_ms

        if elapsed_ms > self.max_latency_ms:
            print(
                f"[Pipeline] WARNING: processing latency {elapsed_ms:.1f} ms "
                f"exceeds {self.max_latency_ms} ms target."
            )

        return state
