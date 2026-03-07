"""LOSO evaluation and ablation study for CognitiveTwin."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset_loaders import DEAPDataset
from models.multimodal_fusion import CognitiveTwinFusionModel
from training.train_multimodal import DEAPMultimodalDataset, train_full_model


TARGET_ACCURACY = 0.85
CLASS_NAMES = ["Underload", "Optimal", "Overload", "Fatigue"]


# ---------------------------------------------------------------------------
# LOSO evaluation
# ---------------------------------------------------------------------------

def loso_evaluation(
    data_dir: str,
    n_subjects: int = 32,
    n_epochs: int = 20,
    batch_size: int = 32,
    device: Optional[torch.device] = None,
) -> Dict[str, object]:
    """Leave-One-Subject-Out (LOSO) cross-validation evaluation.

    Args:
        data_dir: DEAP data directory.
        n_subjects: Number of subjects (default 32).
        n_epochs: Training epochs per fold.
        batch_size: Batch size.
        device: Torch device.

    Returns:
        Dictionary with per-subject accuracies and overall statistics.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_subject_ids = list(range(1, n_subjects + 1))
    per_subject_acc: List[float] = []
    all_preds, all_true = [], []

    for test_sid in all_subject_ids:
        train_sids = [s for s in all_subject_ids if s != test_sid]

        print(f"\n[LOSO] Fold: test subject={test_sid:02d}")

        # Train
        model = train_full_model(
            data_dir=data_dir,
            subject_ids=train_sids,
            n_epochs=n_epochs,
            batch_size=batch_size,
            device=device,
        )
        model.eval()

        # Test
        test_dataset = DEAPMultimodalDataset(data_dir, subject_ids=[test_sid])
        if len(test_dataset) == 0:
            print(f"  [LOSO] No data for subject {test_sid}, skipping.")
            continue
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        fold_preds, fold_true = [], []
        with torch.no_grad():
            for batch in test_loader:
                scalograms, eye_seqs, hrv_seqs, labels, _, _ = [
                    b.to(device) for b in batch
                ]
                outputs = model(scalograms, eye_seqs, hrv_seqs)
                states = model.declare_state(outputs)
                preds = [s["predicted_class"] for s in states]
                fold_preds.extend(preds)
                fold_true.extend(labels.cpu().tolist())

        if fold_true:
            acc = accuracy_score(fold_true, fold_preds)
            per_subject_acc.append(acc)
            all_preds.extend(fold_preds)
            all_true.extend(fold_true)
            print(f"  [LOSO] Subject {test_sid:02d} accuracy: {acc:.3f}")

    if not per_subject_acc:
        print("[LOSO] No evaluation data available.")
        return {}

    overall_acc = np.mean(per_subject_acc)
    print(f"\n[LOSO] Overall accuracy: {overall_acc:.3f} (target: {TARGET_ACCURACY})")

    if all_true:
        print("\n[LOSO] Classification Report:")
        print(classification_report(all_true, all_preds, target_names=CLASS_NAMES))
        print("[LOSO] Confusion Matrix:")
        print(confusion_matrix(all_true, all_preds))

    return {
        "per_subject_accuracies": per_subject_acc,
        "overall_accuracy": overall_acc,
        "meets_target": overall_acc >= TARGET_ACCURACY,
    }


# ---------------------------------------------------------------------------
# Ablation study
# ---------------------------------------------------------------------------

def ablation_study(
    data_dir: str,
    test_subject_id: int = 1,
    n_epochs: int = 20,
    batch_size: int = 32,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Compare EEG-only, Eye-only, HRV-only, and full fusion accuracy.

    Args:
        data_dir: DEAP data directory.
        test_subject_id: Subject to hold out for testing.
        n_epochs: Training epochs per condition.
        batch_size: Batch size.
        device: Torch device.

    Returns:
        Dictionary mapping condition name to test accuracy.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_subjects = 32
    train_sids = [s for s in range(1, n_subjects + 1) if s != test_subject_id]
    test_dataset = DEAPMultimodalDataset(data_dir, subject_ids=[test_subject_id])
    if len(test_dataset) == 0:
        print("[Ablation] No test data available.")
        return {}

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    results: Dict[str, float] = {}

    conditions = ["full", "eeg_only", "eye_only", "hrv_only"]
    for condition in conditions:
        print(f"\n[Ablation] Condition: {condition}")
        model = train_full_model(
            data_dir=data_dir,
            subject_ids=train_sids,
            n_epochs=n_epochs,
            batch_size=batch_size,
            device=device,
        )
        model.eval()

        preds, true = [], []
        with torch.no_grad():
            for batch in test_loader:
                scalograms, eye_seqs, hrv_seqs, labels, _, _ = [
                    b.to(device) for b in batch
                ]
                # For ablation: zero out unused modalities
                if condition == "eeg_only":
                    eye_seqs = torch.zeros_like(eye_seqs)
                    hrv_seqs = torch.zeros_like(hrv_seqs)
                elif condition == "eye_only":
                    scalograms = torch.zeros_like(scalograms)
                    hrv_seqs = torch.zeros_like(hrv_seqs)
                elif condition == "hrv_only":
                    scalograms = torch.zeros_like(scalograms)
                    eye_seqs = torch.zeros_like(eye_seqs)

                outputs = model(scalograms, eye_seqs, hrv_seqs)
                states = model.declare_state(outputs)
                preds.extend([s["predicted_class"] for s in states])
                true.extend(labels.cpu().tolist())

        acc = accuracy_score(true, preds) if true else 0.0
        results[condition] = acc
        print(f"  [Ablation] {condition}: accuracy={acc:.3f}")

    print("\n[Ablation] Summary:")
    for k, v in results.items():
        print(f"  {k:12s}: {v:.3f}")
    fusion_benefit = results.get("full", 0.0) - max(
        results.get("eeg_only", 0.0),
        results.get("eye_only", 0.0),
        results.get("hrv_only", 0.0),
    )
    print(f"  Fusion benefit: +{fusion_benefit:.3f}")
    results["fusion_benefit"] = fusion_benefit
    return results
