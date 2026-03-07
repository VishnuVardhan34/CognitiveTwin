"""Export CognitiveTwinFusionModel to ONNX format."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.multimodal_fusion import CognitiveTwinFusionModel


def export_to_onnx(
    checkpoint_path: str,
    output_path: str = "checkpoints/cognitivetwin.onnx",
    n_channels_eeg: int = 32,
    n_freqs: int = 64,
    n_time: int = 256,
    n_eye_features: int = 7,
    eye_seq_len: int = 128,
    n_hrv_features: int = 10,
    hrv_seq_len: int = 8,
    device: Optional[torch.device] = None,
) -> None:
    """Load a checkpoint and export the model to ONNX.

    Args:
        checkpoint_path: Path to the saved .pth checkpoint.
        output_path: Destination ONNX file path.
        n_channels_eeg: Number of EEG channels.
        n_freqs: Scalogram frequency bins.
        n_time: Scalogram time steps.
        n_eye_features: Eye feature dimension.
        eye_seq_len: Eye sequence length.
        n_hrv_features: HRV feature dimension.
        hrv_seq_len: HRV sequence length.
        device: Torch device.
    """
    if device is None:
        device = torch.device("cpu")

    model = CognitiveTwinFusionModel(
        n_channels_eeg=n_channels_eeg,
        n_freqs=n_freqs,
        n_time=n_time,
        n_eye_features=n_eye_features,
        n_hrv_features=n_hrv_features,
    ).to(device)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Dummy inputs for tracing
    dummy_eeg = torch.randn(1, n_channels_eeg, n_freqs, n_time, device=device)
    dummy_eye = torch.randn(1, eye_seq_len, n_eye_features, device=device)
    dummy_hrv = torch.randn(1, hrv_seq_len, n_hrv_features, device=device)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        (dummy_eeg, dummy_eye, dummy_hrv),
        output_path,
        input_names=["eeg_scalogram", "eye_seq", "hrv_seq"],
        output_names=[
            "fused_logits", "eeg_logits", "eye_logits", "hrv_logits",
            "arousal_valence", "fused_embedding", "confidence_weights",
            "attention_weights", "final_probs", "decision_weights",
        ],
        dynamic_axes={
            "eeg_scalogram": {0: "batch_size"},
            "eye_seq": {0: "batch_size"},
            "hrv_seq": {0: "batch_size"},
            "fused_logits": {0: "batch_size"},
            "final_probs": {0: "batch_size"},
        },
        opset_version=17,
    )
    print(f"[ONNX] Model exported to {output_path}")

    # Verify ONNX model
    try:
        import onnx
        import onnxruntime as ort

        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("[ONNX] Model check passed.")

        ort_session = ort.InferenceSession(output_path)
        ort_inputs = {
            "eeg_scalogram": dummy_eeg.numpy(),
            "eye_seq": dummy_eye.numpy(),
            "hrv_seq": dummy_hrv.numpy(),
        }
        ort_outputs = ort_session.run(None, ort_inputs)
        print(f"[ONNX] ONNX inference output shape: {ort_outputs[0].shape}")

        # Compare PyTorch vs ONNX final_probs
        with torch.no_grad():
            pt_outputs = model(dummy_eeg, dummy_eye, dummy_hrv)
        pt_probs = pt_outputs["final_probs"].numpy()
        onnx_probs = ort_outputs[8]  # final_probs index
        max_diff = float(np.abs(pt_probs - onnx_probs).max())
        print(f"[ONNX] Max output difference (PyTorch vs ONNX): {max_diff:.2e}")
        if max_diff >= 1e-4:
            print(f"[ONNX] WARNING: output mismatch exceeds threshold ({max_diff:.2e} >= 1e-4). "
                  "Check model export settings.")
        else:
            print("[ONNX] Output verification passed.")
    except ImportError:
        print("[ONNX] onnx/onnxruntime not installed, skipping verification.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export CognitiveTwin to ONNX")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="checkpoints/cognitivetwin.onnx")
    args = parser.parse_args()
    export_to_onnx(args.checkpoint, args.output)
