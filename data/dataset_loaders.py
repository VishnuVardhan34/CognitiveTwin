"""
Dataset loaders for DEAP, SEED-IV, and DROZY datasets.

Supports loading, label mapping, and windowing for multimodal cognitive load estimation.
"""

from __future__ import annotations

import os
import pickle
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.io as sio


# ---------------------------------------------------------------------------
# Label mapping utilities
# ---------------------------------------------------------------------------

def _deap_av_to_cognitive(arousal: float, valence: float) -> int:
    """Map DEAP arousal/valence scores to 4-class cognitive state.

    Classes:
        0 – Underload  (low arousal, low valence)
        1 – Optimal    (high arousal, high valence)
        2 – Overload   (high arousal, low valence)
        3 – Fatigue    (very low arousal, any valence)

    Args:
        arousal: Arousal score on 1–9 scale.
        valence: Valence score on 1–9 scale.

    Returns:
        Integer class label in {0, 1, 2, 3}.
    """
    if arousal < 3.5:
        return 3  # Fatigue
    high_arousal = arousal >= 5.0
    high_valence = valence >= 5.0
    if high_arousal and high_valence:
        return 1  # Optimal
    if high_arousal and not high_valence:
        return 2  # Overload
    return 0  # Underload


def _seediv_to_cognitive(seed_label: int) -> int:
    """Map SEED-IV emotion label to cognitive state.

    SEED-IV: 0=Neutral, 1=Sad, 2=Fear, 3=Happy
    Cognitive: 0=Underload, 1=Optimal, 2=Overload, 3=Fatigue

    Args:
        seed_label: Original SEED-IV label.

    Returns:
        Integer cognitive class label.
    """
    mapping = {0: 0, 1: 3, 2: 2, 3: 1}
    return mapping.get(int(seed_label), 0)


def _kss_to_cognitive(kss: int) -> int:
    """Map Karolinska Sleepiness Scale (KSS) score to cognitive state.

    Args:
        kss: KSS score on 1–9 scale.

    Returns:
        Integer cognitive class label.
    """
    if kss <= 5:
        return 1  # Optimal
    return 3  # Fatigue


# ---------------------------------------------------------------------------
# DEAP Dataset
# ---------------------------------------------------------------------------

class DEAPDataset:
    """Loader for the DEAP dataset.

    The DEAP dataset consists of EEG and peripheral physiological signals
    recorded from 32 participants while watching 40 one-minute music videos.
    Data are stored as pickled .dat files (one per participant).

    Data format per subject:
        data:   (40, 40, 8064) – trials × channels × samples @ 128 Hz
        labels: (40, 4)        – [valence, arousal, dominance, liking] on 1–9

    Channels:
        0–31 : EEG (32 channels, 10-20 system)
        32   : EOG horizontal
        33   : EOG vertical
        34   : EMG zygomaticus
        35   : EMG trapezius
        36   : GSR
        37   : Respiration
        38   : Plethysmograph (PPG)
        39   : Temperature

    Args:
        data_dir: Path to the DEAP data_preprocessed_python directory.
    """

    N_TRIALS: int = 40
    N_CHANNELS: int = 40
    N_SAMPLES: int = 8064
    SFREQ: int = 128
    EEG_CHANNELS: int = 32
    PPG_IDX: int = 38  # plethysmograph channel (0-indexed)

    def __init__(self, data_dir: str) -> None:
        self.data_dir = Path(data_dir)

    def load_subject(self, sid: int) -> Dict[str, np.ndarray]:
        """Load a single subject's data.

        Args:
            sid: Subject ID (1–32).

        Returns:
            Dictionary with keys:
                - ``eeg``           : (40, 32, 8064) float32
                - ``periph``        : (40, 8, 8064)  float32 (channels 32-39)
                - ``ppg``           : (40, 8064)     float32
                - ``cognitive_label``: (40,) int64
                - ``arousal_valence``: (40, 2) float32 – [arousal, valence] normalised to [0, 1]
        """
        fname = self.data_dir / f"s{sid:02d}.dat"
        if not fname.exists():
            raise FileNotFoundError(f"DEAP subject file not found: {fname}")

        with open(fname, "rb") as fh:
            subject = pickle.load(fh, encoding="latin1")

        data: np.ndarray = subject["data"].astype(np.float32)     # (40, 40, 8064)
        labels: np.ndarray = subject["labels"].astype(np.float32)  # (40, 4)

        eeg = data[:, : self.EEG_CHANNELS, :]          # (40, 32, 8064)
        periph = data[:, self.EEG_CHANNELS :, :]       # (40, 8, 8064)
        ppg = data[:, self.PPG_IDX, :]                 # (40, 8064)

        valence = labels[:, 0]   # 1–9
        arousal = labels[:, 1]   # 1–9

        cognitive_labels = np.array(
            [_deap_av_to_cognitive(a, v) for a, v in zip(arousal, valence)],
            dtype=np.int64,
        )

        # Normalise arousal / valence to [0, 1]
        arousal_valence = np.stack(
            [(arousal - 1.0) / 8.0, (valence - 1.0) / 8.0], axis=1
        ).astype(np.float32)

        return {
            "eeg": eeg,
            "periph": periph,
            "ppg": ppg,
            "cognitive_label": cognitive_labels,
            "arousal_valence": arousal_valence,
        }

    def load_all(
        self, subject_ids: Optional[List[int]] = None
    ) -> Dict[str, np.ndarray]:
        """Load and concatenate data for multiple subjects.

        Args:
            subject_ids: List of subject IDs to load. Defaults to all 32.

        Returns:
            Concatenated dictionary (same keys as ``load_subject``), plus
            ``subject_id`` array of shape (N,).
        """
        if subject_ids is None:
            subject_ids = list(range(1, 33))

        all_eeg, all_periph, all_ppg, all_labels, all_av, all_sids = (
            [],
            [],
            [],
            [],
            [],
            [],
        )

        for sid in subject_ids:
            try:
                d = self.load_subject(sid)
            except FileNotFoundError:
                continue
            n = d["eeg"].shape[0]
            all_eeg.append(d["eeg"])
            all_periph.append(d["periph"])
            all_ppg.append(d["ppg"])
            all_labels.append(d["cognitive_label"])
            all_av.append(d["arousal_valence"])
            all_sids.append(np.full(n, sid, dtype=np.int64))

        if not all_eeg:
            raise RuntimeError("No DEAP data could be loaded.")

        return {
            "eeg": np.concatenate(all_eeg, axis=0),
            "periph": np.concatenate(all_periph, axis=0),
            "ppg": np.concatenate(all_ppg, axis=0),
            "cognitive_label": np.concatenate(all_labels, axis=0),
            "arousal_valence": np.concatenate(all_av, axis=0),
            "subject_id": np.concatenate(all_sids, axis=0),
        }


# ---------------------------------------------------------------------------
# SEED-IV Dataset
# ---------------------------------------------------------------------------

class SEEDIVDataset:
    """Loader for the SEED-IV dataset.

    Directory structure assumed::

        seediv_dir/
        └── eeg_raw_data/
            ├── 1/          ← session
            │   ├── 1_20160518.mat
            │   └── ...
            ├── 2/
            └── 3/

    Each .mat file contains multiple trial variables (e.g., ``cz_eeg1`` …
    ``cz_eeg24``).  The companion label array ``label_seediv.mat`` or the
    hardcoded sequence 0,1,2,3,… (see SEED-IV paper) is used.

    Args:
        data_dir: Root directory of SEED-IV dataset.
        sfreq: EEG sampling frequency (default 200 Hz).
        window_samples: Samples per segment (default 400 = 2 s @ 200 Hz).
    """

    N_SUBJECTS: int = 15
    N_SESSIONS: int = 3
    SFREQ: int = 200
    N_CHANNELS: int = 62

    # Trial-order labels per session (from SEED-IV paper)
    SESSION_LABELS: Dict[int, List[int]] = {
        1: [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
        2: [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
        3: [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0],
    }

    def __init__(
        self,
        data_dir: str,
        sfreq: int = 200,
        window_samples: int = 400,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.sfreq = sfreq
        self.window_samples = window_samples

    def _load_mat_trial(
        self, mat_path: Path, session: int
    ) -> Tuple[List[np.ndarray], List[int]]:
        """Load all EEG trials from a single .mat file.

        Args:
            mat_path: Path to the .mat file.
            session: Session number (1–3) for label lookup.

        Returns:
            Tuple of (list of EEG arrays, list of int labels).
        """
        mat = sio.loadmat(str(mat_path))
        trial_keys = sorted(
            [k for k in mat.keys() if not k.startswith("_")],
            key=lambda x: int("".join(filter(str.isdigit, x)) or "0"),
        )

        labels = self.SESSION_LABELS.get(session, [])
        trials, trial_labels = [], []

        for i, key in enumerate(trial_keys):
            arr = mat[key]
            if arr.ndim != 2:
                continue
            # Ensure shape is (n_channels, n_samples)
            if arr.shape[0] > arr.shape[1]:
                arr = arr.T
            if arr.shape[0] != self.N_CHANNELS:
                arr = arr[: self.N_CHANNELS, :]
            trials.append(arr.astype(np.float32))
            label = labels[i] if i < len(labels) else 0
            trial_labels.append(label)

        return trials, trial_labels

    def _segment_trial(self, eeg: np.ndarray) -> np.ndarray:
        """Segment a trial into fixed-length windows.

        Args:
            eeg: EEG array of shape (n_channels, n_samples).

        Returns:
            Windowed array of shape (n_windows, n_channels, window_samples).
        """
        n_samples = eeg.shape[1]
        n_windows = n_samples // self.window_samples
        segments = []
        for i in range(n_windows):
            start = i * self.window_samples
            end = start + self.window_samples
            segments.append(eeg[:, start:end])
        return np.stack(segments, axis=0) if segments else np.empty((0, self.N_CHANNELS, self.window_samples), dtype=np.float32)

    def load_all_as_cognitive(
        self,
        subject_ids: Optional[List[int]] = None,
    ) -> Dict[str, np.ndarray]:
        """Load all SEED-IV data mapped to cognitive labels.

        Args:
            subject_ids: List of subject IDs (1-based). Defaults to all 15.

        Returns:
            Dictionary with keys:
                - ``eeg``            : (N, 62, window_samples) float32
                - ``cognitive_label``: (N,) int64
                - ``subject_id``     : (N,) int64
        """
        if subject_ids is None:
            subject_ids = list(range(1, self.N_SUBJECTS + 1))

        all_eeg, all_labels, all_sids = [], [], []

        eeg_dir = self.data_dir / "eeg_raw_data"
        if not eeg_dir.exists():
            eeg_dir = self.data_dir  # fallback

        for session in range(1, self.N_SESSIONS + 1):
            session_dir = eeg_dir / str(session)
            if not session_dir.exists():
                continue
            mat_files = sorted(session_dir.glob("*.mat"))

            for mat_file in mat_files:
                # Infer subject ID from file stem digits
                stem_digits = "".join(filter(str.isdigit, mat_file.stem.split("_")[0]))
                sid = int(stem_digits) if stem_digits else 0
                if sid not in subject_ids:
                    continue

                trials, trial_labels = self._load_mat_trial(mat_file, session)
                for trial, label in zip(trials, trial_labels):
                    segments = self._segment_trial(trial)
                    if segments.shape[0] == 0:
                        continue
                    cog_label = _seediv_to_cognitive(label)
                    all_eeg.append(segments)
                    all_labels.append(
                        np.full(segments.shape[0], cog_label, dtype=np.int64)
                    )
                    all_sids.append(np.full(segments.shape[0], sid, dtype=np.int64))

        if not all_eeg:
            # Return empty arrays with correct dtypes if no data found
            return {
                "eeg": np.empty((0, self.N_CHANNELS, self.window_samples), dtype=np.float32),
                "cognitive_label": np.empty((0,), dtype=np.int64),
                "subject_id": np.empty((0,), dtype=np.int64),
            }

        return {
            "eeg": np.concatenate(all_eeg, axis=0),
            "cognitive_label": np.concatenate(all_labels, axis=0),
            "subject_id": np.concatenate(all_sids, axis=0),
        }


# ---------------------------------------------------------------------------
# DROZY Dataset
# ---------------------------------------------------------------------------

class DROZYDataset:
    """Loader for the DROZY dataset.

    DROZY contains EEG + facial video data with Karolinska Sleepiness Scale
    (KSS) annotations for 14 participants.

    Expected directory structure::

        drozy_dir/
        ├── PSG/
        │   ├── participant_01.mat
        │   └── ...
        └── KSS/
            ├── participant_01.txt
            └── ...

    Args:
        data_dir: Root directory of DROZY dataset.
        sfreq: EEG sampling frequency (default 256 Hz).
    """

    N_PARTICIPANTS: int = 14
    SFREQ: int = 256
    EAR_THRESHOLD: float = 0.2

    def __init__(self, data_dir: str, sfreq: int = 256) -> None:
        self.data_dir = Path(data_dir)
        self.sfreq = sfreq

    def _load_kss(self, pid: int) -> int:
        """Load KSS rating for a participant.

        Args:
            pid: Participant ID (1-based).

        Returns:
            KSS score as integer (1–9).
        """
        kss_path = self.data_dir / "KSS" / f"participant_{pid:02d}.txt"
        if not kss_path.exists():
            return 5  # default to optimal
        with open(kss_path, "r") as fh:
            line = fh.readline().strip()
            return int(float(line))

    def _load_psg_eeg(self, pid: int) -> Optional[np.ndarray]:
        """Load PSG EEG data for a participant.

        Args:
            pid: Participant ID (1-based).

        Returns:
            EEG array of shape (n_channels, n_samples) or None if not found.
        """
        psg_path = self.data_dir / "PSG" / f"participant_{pid:02d}.mat"
        if not psg_path.exists():
            return None
        mat = sio.loadmat(str(psg_path))
        eeg_key = next(
            (k for k in mat.keys() if not k.startswith("_") and "eeg" in k.lower()),
            None,
        )
        if eeg_key is None:
            eeg_key = next((k for k in mat.keys() if not k.startswith("_")), None)
        if eeg_key is None:
            return None
        arr = mat[eeg_key]
        if arr.ndim != 2:
            return None
        if arr.shape[0] > arr.shape[1]:
            arr = arr.T
        return arr.astype(np.float32)

    def _extract_ear_features(self, landmarks: np.ndarray) -> np.ndarray:
        """Compute eye aspect ratio (EAR) from 68-point facial landmarks.

        Args:
            landmarks: Array of shape (N_frames, 68, 2) with (x, y) coordinates.

        Returns:
            Feature array of shape (N_frames, 3): [EAR, PERCLOS_rolling, blink_rate].
        """
        # Landmark indices for left eye: 36-41, right eye: 42-47
        def _ear(pts: np.ndarray) -> np.ndarray:
            # pts: (N, 6, 2)
            A = np.linalg.norm(pts[:, 1] - pts[:, 5], axis=-1)
            B = np.linalg.norm(pts[:, 2] - pts[:, 4], axis=-1)
            C = np.linalg.norm(pts[:, 0] - pts[:, 3], axis=-1)
            return (A + B) / (2.0 * C + 1e-6)

        left_pts = landmarks[:, 36:42, :]
        right_pts = landmarks[:, 42:48, :]
        ear = (_ear(left_pts) + _ear(right_pts)) / 2.0  # (N_frames,)

        # PERCLOS: percentage of eye closure over rolling 60-frame window
        closed = (ear < self.EAR_THRESHOLD).astype(np.float32)
        win = 60
        perclos = np.convolve(closed, np.ones(win) / win, mode="same")

        # Blink rate: count transitions below threshold per second (30 fps assumed)
        blinks = np.diff((ear < self.EAR_THRESHOLD).astype(np.int32), prepend=0)
        blinks = np.maximum(blinks, 0).astype(np.float32)
        fps = 30
        blink_rate = np.convolve(blinks, np.ones(fps) / fps, mode="same")

        return np.stack([ear, perclos, blink_rate], axis=1).astype(np.float32)

    def load_participant(self, pid: int) -> Dict[str, np.ndarray]:
        """Load a single participant's data.

        Args:
            pid: Participant ID (1–14).

        Returns:
            Dictionary with keys:
                - ``eeg``            : (n_channels, n_samples) float32 or None
                - ``cognitive_label``: int
                - ``kss``            : int
        """
        kss = self._load_kss(pid)
        eeg = self._load_psg_eeg(pid)
        cognitive_label = _kss_to_cognitive(kss)
        return {
            "eeg": eeg,
            "cognitive_label": cognitive_label,
            "kss": kss,
        }

    def load_all(
        self,
        participant_ids: Optional[List[int]] = None,
    ) -> Dict[str, object]:
        """Load all participants.

        Args:
            participant_ids: IDs to load. Defaults to all 14.

        Returns:
            Dictionary with ``participants`` list and ``cognitive_labels`` array.
        """
        if participant_ids is None:
            participant_ids = list(range(1, self.N_PARTICIPANTS + 1))

        participants = []
        labels = []
        for pid in participant_ids:
            d = self.load_participant(pid)
            participants.append(d)
            labels.append(d["cognitive_label"])

        return {
            "participants": participants,
            "cognitive_labels": np.array(labels, dtype=np.int64),
        }
