"""Full training pipeline for CognitiveTwin (phases 1–4)."""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset_loaders import DEAPDataset, SEEDIVDataset
from models.eeg_branch import EEGWaveletCNN
from models.multimodal_fusion import CognitiveTwinFusionModel
from preprocessing.eeg_preprocessor import EEGPreprocessor
from preprocessing.eye_preprocessor import EyeFeatureExtractor
from preprocessing.hrv_preprocessor import HRVFeatureExtractor
from preprocessing.wavelet_transform import compute_scalogram, eeg_to_scalogram_tensor
from training.losses import CognitiveTwinLoss


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def _set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class SEEDIVPretrainDataset(Dataset):
    """EEG-only pre-training dataset built from SEED-IV data.

    Args:
        data_dir: Path to SEED-IV dataset root.
        sfreq: EEG sampling frequency.
        n_freqs: Number of CWT frequency bins.
        n_time: Fixed time dimension for scalogram (samples).
        subject_ids: Subjects to include (1-based).
    """

    def __init__(
        self,
        data_dir: str,
        sfreq: int = 200,
        n_freqs: int = 64,
        n_time: int = 400,
        subject_ids: Optional[List[int]] = None,
    ) -> None:
        loader = SEEDIVDataset(data_dir, sfreq=sfreq, window_samples=n_time)
        data = loader.load_all_as_cognitive(subject_ids=subject_ids)

        self.eeg = data["eeg"]           # (N, 62, n_time)
        self.labels = data["cognitive_label"]  # (N,)
        self.sfreq = sfreq
        self.n_freqs = n_freqs
        self.n_time = n_time
        self.preprocessor = EEGPreprocessor(sfreq=sfreq, n_channels=62)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        eeg = self.preprocessor.bandpass_filter(self.eeg[idx])  # (62, n_time)
        eeg = self.preprocessor.normalize(eeg)
        scalogram = eeg_to_scalogram_tensor(eeg, sfreq=self.sfreq, n_freqs=self.n_freqs)
        # Truncate or pad time dimension
        T = scalogram.shape[-1]
        if T > self.n_time:
            scalogram = scalogram[:, :, :self.n_time]
        elif T < self.n_time:
            pad = np.zeros((scalogram.shape[0], scalogram.shape[1], self.n_time - T), dtype=np.float32)
            scalogram = np.concatenate([scalogram, pad], axis=-1)
        return torch.tensor(scalogram, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)


class DEAPMultimodalDataset(Dataset):
    """Multimodal dataset built from DEAP data.

    Segments each 1-minute trial into 2-second windows and pre-computes
    scalograms, eye features, and HRV windows per segment.

    Args:
        data_dir: Path to DEAP data_preprocessed_python directory.
        subject_ids: Subjects to include.
        sfreq: EEG sampling frequency (default 128).
        seg_samples: Samples per 2-second segment at 128 Hz (default 256).
        n_freqs: CWT frequency bins (default 64).
    """

    def __init__(
        self,
        data_dir: str,
        subject_ids: Optional[List[int]] = None,
        sfreq: int = 128,
        seg_samples: int = 256,
        n_freqs: int = 64,
    ) -> None:
        loader = DEAPDataset(data_dir)
        data = loader.load_all(subject_ids=subject_ids)

        self.sfreq = sfreq
        self.seg_samples = seg_samples
        self.n_freqs = n_freqs

        self.eeg_preprocessor = EEGPreprocessor(sfreq=sfreq, n_channels=32)
        self.eye_extractor = EyeFeatureExtractor(sfreq=sfreq)
        self.hrv_extractor = HRVFeatureExtractor(sfreq=sfreq)

        # Build sample index: (trial_idx, segment_start)
        self.samples: List[Tuple[int, int]] = []
        self._data = data
        n_trials = data["eeg"].shape[0]
        n_total_samples = data["eeg"].shape[-1]
        n_segs = n_total_samples // seg_samples
        for trial in range(n_trials):
            for seg in range(n_segs):
                self.samples.append((trial, seg * seg_samples))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        trial_idx, start = self.samples[idx]
        end = start + self.seg_samples
        data = self._data

        # EEG segment
        eeg_seg = data["eeg"][trial_idx, :, start:end]  # (32, 256)
        eeg_seg = self.eeg_preprocessor.bandpass_filter(eeg_seg)
        eeg_seg = self.eeg_preprocessor.normalize(eeg_seg)
        scalogram = eeg_to_scalogram_tensor(eeg_seg, sfreq=self.sfreq, n_freqs=self.n_freqs)
        # scalogram: (32, n_freqs, 256)

        # Eye features (from EOG channels 0=h, 1=v in periph)
        eog_h = data["periph"][trial_idx, 0, start:end]
        eog_v = data["periph"][trial_idx, 1, start:end]
        eye_seq = self.eye_extractor.extract_from_deap_eog(eog_h, eog_v)
        # eye_seq: (256, 7)

        # HRV features (from PPG, use full trial for proper windowing)
        ppg_full = data["ppg"][trial_idx]
        hrv_features = self.hrv_extractor.extract_features(ppg_full)  # (10,)
        # Tile to create a pseudo-sequence of length 8.
        # 8 steps match the HRV LSTM encoder's expected sequence length (set in
        # models/multimodal_fusion.py) while keeping memory overhead minimal for
        # a single-window HRV feature vector.
        hrv_seq = np.tile(hrv_features, (8, 1))  # (8, 10)

        label = data["cognitive_label"][trial_idx]
        av = data["arousal_valence"][trial_idx]
        sid = data["subject_id"][trial_idx]

        return (
            torch.tensor(scalogram, dtype=torch.float32),
            torch.tensor(eye_seq, dtype=torch.float32),
            torch.tensor(hrv_seq, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
            torch.tensor(av, dtype=torch.float32),
            torch.tensor(sid, dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# Phase 1: Pre-train EEG branch on SEED-IV
# ---------------------------------------------------------------------------

def pretrain_eeg_branch_on_seediv(
    data_dir: str,
    output_path: str = "checkpoints/eeg_pretrained.pth",
    n_freqs: int = 64,
    n_time: int = 400,
    n_epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
) -> EEGWaveletCNN:
    """Pre-train EEGWaveletCNN on SEED-IV for cognitive-state classification.

    Args:
        data_dir: SEED-IV root directory.
        output_path: Path to save pre-trained weights.
        n_freqs: CWT frequency bins.
        n_time: Scalogram time steps.
        n_epochs: Number of training epochs.
        batch_size: Batch size.
        lr: Learning rate.
        device: Torch device. Defaults to CUDA if available.

    Returns:
        Pre-trained ``EEGWaveletCNN`` instance.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[Phase 1] Pre-training EEG branch on SEED-IV ({device})")
    dataset = SEEDIVPretrainDataset(data_dir, n_freqs=n_freqs, n_time=n_time)

    if len(dataset) == 0:
        print("[Phase 1] No SEED-IV data found, skipping pre-training.")
        return EEGWaveletCNN(n_channels=62, n_freqs=n_freqs, n_time=n_time).to(device)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = EEGWaveletCNN(n_channels=62, n_freqs=n_freqs, n_time=n_time, embedding_dim=64).to(device)
    head = nn.Linear(64, 4).to(device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(head.parameters()), lr=lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    for epoch in range(1, n_epochs + 1):
        model.train()
        head.train()
        total_loss, correct, total = 0.0, 0, 0
        for scalograms, labels in loader:
            scalograms, labels = scalograms.to(device), labels.to(device)
            optimizer.zero_grad()
            emb = model(scalograms)
            logits = head(emb)
            loss = nn.functional.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            total += len(labels)
        scheduler.step()
        acc = correct / max(total, 1)
        print(f"  [Phase 1] Epoch {epoch:3d}/{n_epochs}  loss={total_loss/max(total,1):.4f}  acc={acc:.3f}")

    os.makedirs(Path(output_path).parent, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"[Phase 1] Saved pre-trained EEG branch → {output_path}")
    return model


# ---------------------------------------------------------------------------
# Phase 2-3: Full multimodal training on DEAP
# ---------------------------------------------------------------------------

def train_full_model(
    data_dir: str,
    pretrained_eeg: Optional[EEGWaveletCNN] = None,
    output_path: str = "checkpoints/cognitivetwin_model.pth",
    n_freqs: int = 64,
    n_time: int = 256,
    n_epochs: int = 50,
    batch_size: int = 32,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    device: Optional[torch.device] = None,
    subject_ids: Optional[List[int]] = None,
) -> CognitiveTwinFusionModel:
    """Train the full multimodal fusion model on DEAP.

    Args:
        data_dir: DEAP data directory.
        pretrained_eeg: Optional pre-trained EEGWaveletCNN to initialise the
            EEG branch (compatible weights are copied).
        output_path: Path to save the trained model checkpoint.
        n_freqs: Scalogram frequency bins.
        n_time: Scalogram time steps.
        n_epochs: Training epochs.
        batch_size: Batch size.
        lr: Learning rate.
        weight_decay: AdamW weight decay.
        grad_clip: Gradient clipping max-norm.
        device: Torch device.
        subject_ids: DEAP subject IDs to use.

    Returns:
        Trained ``CognitiveTwinFusionModel``.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[Phase 2-3] Training full model on DEAP ({device})")
    dataset = DEAPMultimodalDataset(
        data_dir, subject_ids=subject_ids, n_freqs=n_freqs, seg_samples=n_time
    )

    if len(dataset) == 0:
        print("[Phase 2-3] No DEAP data found, skipping full training.")
        model = CognitiveTwinFusionModel(n_channels_eeg=32, n_freqs=n_freqs, n_time=n_time)
        return model.to(device)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = CognitiveTwinFusionModel(
        n_channels_eeg=32,
        n_freqs=n_freqs,
        n_time=n_time,
        n_eye_features=7,
        n_hrv_features=10,
    ).to(device)

    # Transfer EEG branch weights if pre-trained
    if pretrained_eeg is not None:
        try:
            model.eeg_branch.load_state_dict(pretrained_eeg.state_dict())
            print("[Phase 2-3] Transferred EEG branch weights from pre-training.")
        except RuntimeError as e:
            print(f"[Phase 2-3] Weight transfer skipped (incompatible): {e}")

    criterion = CognitiveTwinLoss()
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss, epoch_cls, epoch_reg, epoch_aux, correct, total = (
            0.0, 0.0, 0.0, 0.0, 0, 0
        )
        for scalograms, eye_seqs, hrv_seqs, labels, av_targets, _ in loader:
            scalograms = scalograms.to(device)
            eye_seqs = eye_seqs.to(device)
            hrv_seqs = hrv_seqs.to(device)
            labels = labels.to(device)
            av_targets = av_targets.to(device)

            # Map AV from [0,1] to [-1,1] (Tanh range)
            av_targets_scaled = av_targets * 2.0 - 1.0

            optimizer.zero_grad()
            outputs = model(scalograms, eye_seqs, hrv_seqs)
            loss_dict = criterion(outputs, labels, av_targets_scaled)
            loss = loss_dict["total"]
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            epoch_loss += loss.item() * len(labels)
            epoch_cls += loss_dict["cls"].item() * len(labels)
            epoch_reg += loss_dict["reg"].item() * len(labels)
            epoch_aux += loss_dict["aux"].item() * len(labels)
            pred = outputs["final_probs"].argmax(1)
            correct += (pred == labels).sum().item()
            total += len(labels)

        scheduler.step()
        n = max(total, 1)
        acc = correct / n
        print(
            f"  [Phase 2-3] Epoch {epoch:3d}/{n_epochs}  "
            f"loss={epoch_loss/n:.4f}  cls={epoch_cls/n:.4f}  "
            f"reg={epoch_reg/n:.4f}  aux={epoch_aux/n:.4f}  acc={acc:.3f}"
        )

        if epoch % 10 == 0:
            tw = loss_dict["task_weights"]
            print(
                f"    Task weights: cls={tw['cls']:.3f}  reg={tw['reg']:.3f}  "
                f"aux={tw['aux']:.3f}  agree={tw['agree']:.3f}"
            )

    os.makedirs(Path(output_path).parent, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"[Phase 2-3] Saved model checkpoint → {output_path}")
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point: run full training pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="CognitiveTwin training pipeline")
    parser.add_argument("--seediv-dir", default="./datasets/SEED-IV", help="SEED-IV data directory")
    parser.add_argument("--deap-dir", default="./datasets/DEAP/data_preprocessed_python", help="DEAP data directory")
    parser.add_argument("--epochs-pretrain", type=int, default=30)
    parser.add_argument("--epochs-full", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrain", action="store_true", help="Skip EEG pre-training")
    args = parser.parse_args()

    _set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Phase 1: Pre-train EEG branch
    pretrained_eeg = None
    if not args.no_pretrain:
        pretrained_eeg = pretrain_eeg_branch_on_seediv(
            args.seediv_dir,
            n_epochs=args.epochs_pretrain,
            batch_size=args.batch_size,
            device=device,
        )

    # Phase 2-3: Full training on DEAP
    model = train_full_model(
        args.deap_dir,
        pretrained_eeg=pretrained_eeg,
        n_epochs=args.epochs_full,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
    )

    # Inference demo with declare_state
    print("\n[Demo] Running declare_state() on a random batch …")
    model.eval()
    with torch.no_grad():
        dummy_eeg = torch.randn(2, 32, 64, 256, device=device)
        dummy_eye = torch.randn(2, 128, 7, device=device)
        dummy_hrv = torch.randn(2, 8, 10, device=device)
        outputs = model(dummy_eeg, dummy_eye, dummy_hrv)
        states = model.declare_state(outputs)

    for i, s in enumerate(states):
        print(f"\n  Sample {i}:")
        print(json.dumps(s, indent=4))


if __name__ == "__main__":
    main()
