#!/usr/bin/env python3
"""
Dataset download helper for CognitiveTwin.

This script provides alternative download methods for the DEAP, SEED-IV, and
DROZY datasets, which are particularly useful for users in regions where the
primary dataset websites are inaccessible (e.g. India).

Supported methods:
  - Kaggle API  (recommended – works globally including India)
  - Direct URL  (fallback, may be blocked in some regions)
  - Manual      (instructions to request the dataset by e-mail)

Usage::

    python scripts/download_datasets.py --dataset deap   --method kaggle --out ./datasets
    python scripts/download_datasets.py --dataset seediv --method kaggle --out ./datasets
    python scripts/download_datasets.py --dataset drozy  --method manual

    # Download all datasets via Kaggle in one go:
    python scripts/download_datasets.py --dataset all --method kaggle --out ./datasets

Kaggle API setup (one-time):
    1. Log in at https://www.kaggle.com and go to Account → API → Create New Token.
    2. Save the downloaded ``kaggle.json`` to ``~/.kaggle/kaggle.json``
       (Linux/macOS) or ``C:\\Users\\<user>\\.kaggle\\kaggle.json`` (Windows).
    3. ``chmod 600 ~/.kaggle/kaggle.json``  (Linux/macOS only)
    4. ``pip install kaggle``
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import textwrap
import zipfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Dataset metadata
# ---------------------------------------------------------------------------

DATASETS: dict[str, dict] = {
    "deap": {
        "name": "DEAP",
        "description": (
            "Database for Emotion Analysis using Physiological Signals\n"
            "32 participants, 40 EEG channels + peripheral signals @ 128 Hz"
        ),
        "primary_url": "http://www.eecs.qmul.ac.uk/mmv/datasets/deap/download.html",
        "kaggle": {
            # Two Kaggle mirrors – try in order
            "datasets": [
                "laevitasimpl/deap-dataset-for-emotion-analysis",
                "birdy654/eeg-brainwave-dataset-feeling-emotions",
            ],
            "note": (
                "After download, look for 'data_preprocessed_python.zip' (or the "
                "directory) and move its contents to datasets/DEAP/data_preprocessed_python/."
            ),
        },
        "manual": {
            "steps": [
                "1. Open https://www.kaggle.com/datasets/laevitasimpl/deap-dataset-for-emotion-analysis\n"
                "   in your browser and click 'Download' (requires a free Kaggle account).",
                "2. Alternatively, e-mail the DEAP authors at deap@eecs.qmul.ac.uk with your name "
                "   and institutional affiliation.  They typically respond within a few days.",
                "3. Extract the archive to  datasets/DEAP/data_preprocessed_python/",
            ],
        },
        "target_subdir": "DEAP",
    },
    "seediv": {
        "name": "SEED-IV",
        "description": (
            "SJTU Emotion EEG Dataset – IV\n"
            "15 participants, 62 EEG channels @ 200 Hz, 4 emotion classes"
        ),
        "primary_url": "https://bcmi.sjtu.edu.cn/~seed/seed-iv.html",
        "kaggle": {
            "datasets": [
                "qiriro/seed-iv-eeg-emotion-recognition",
            ],
            "note": (
                "Extract the downloaded archive so that the directory layout is "
                "datasets/SEED-IV/eeg_raw_data/{1,2,3}/<subject>.mat"
            ),
        },
        "manual": {
            "steps": [
                "1. Open https://www.kaggle.com/datasets/qiriro/seed-iv-eeg-emotion-recognition "
                "   and click 'Download' (requires a free Kaggle account).",
                "2. Alternatively, fill in the request form on the SEED website "
                "   (https://bcmi.sjtu.edu.cn/~seed/seed-iv.html#download-link) using an "
                "   institutional/university e-mail address.  Access is usually granted within "
                "   1–3 business days.",
                "3. Extract the archive to  datasets/SEED-IV/",
            ],
        },
        "target_subdir": "SEED-IV",
    },
    "drozy": {
        "name": "DROZY",
        "description": (
            "Drowsiness dataset with PSG (EEG) + KSS annotations\n"
            "14 participants, 256 Hz"
        ),
        "primary_url": "http://drozy.ulg.ac.be/",
        "kaggle": {
            # DROZY does not currently have an official Kaggle mirror;
            # the script will warn the user and fall back to manual instructions.
            "datasets": [],
            "note": "DROZY is not yet on Kaggle – see manual instructions below.",
        },
        "manual": {
            "steps": [
                "1. The DROZY dataset is hosted via the University of Liège.  "
                "   Try the direct link: http://drozy.ulg.ac.be/  "
                "   (some ISPs in India may need a VPN to reach .ulg.ac.be).",
                "2. Alternatively, e-mail the dataset authors at drozy@ulg.ac.be "
                "   (or the corresponding author listed in the DROZY paper) with your "
                "   name and affiliation to request a Google Drive / Dropbox link.",
                "3. The Zenodo record https://doi.org/10.5281/zenodo.1230005 may also "
                "   contain a mirror; check for availability.",
                "4. Extract to  datasets/DROZY/",
            ],
        },
        "target_subdir": "DROZY",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_kaggle() -> bool:
    """Return True if the kaggle Python package and credentials are available."""
    try:
        import kaggle  # noqa: F401  # type: ignore
        return True
    except ImportError:
        return False


def _install_kaggle() -> bool:
    """Attempt to install the kaggle package via pip."""
    print("[INFO] Installing the 'kaggle' package…")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "kaggle", "--quiet"],
    )
    return result.returncode == 0


def _kaggle_credentials_exist() -> bool:
    """Return True if kaggle.json credentials file exists."""
    default_path = Path.home() / ".kaggle" / "kaggle.json"
    env_path = os.environ.get("KAGGLE_CONFIG_DIR")
    if env_path:
        return (Path(env_path) / "kaggle.json").exists()
    return default_path.exists()


def _download_kaggle_dataset(dataset_slug: str, dest_dir: Path) -> bool:
    """Download a Kaggle dataset using the kaggle CLI.

    Args:
        dataset_slug: Kaggle dataset identifier (``owner/dataset-name``).
        dest_dir: Local directory to download and unzip into.

    Returns:
        True on success, False otherwise.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "kaggle",
        "datasets", "download",
        "-d", dataset_slug,
        "-p", str(dest_dir),
        "--unzip",
    ]
    print(f"[INFO] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode == 0


def _print_manual_instructions(meta: dict) -> None:
    """Print step-by-step manual download instructions."""
    print(f"\n{'=' * 60}")
    print(f"  Manual download instructions for {meta['name']}")
    print(f"{'=' * 60}")
    for step in meta["manual"]["steps"]:
        print(textwrap.fill(step, width=72, subsequent_indent="   "))
    print()


# ---------------------------------------------------------------------------
# Per-dataset download logic
# ---------------------------------------------------------------------------

def download_via_kaggle(key: str, out_dir: Path) -> None:
    """Download a dataset via the Kaggle API.

    Args:
        key: Dataset key (``deap``, ``seediv``, or ``drozy``).
        out_dir: Root output directory (datasets will land in subdirectories).
    """
    meta = DATASETS[key]
    slugs: list[str] = meta["kaggle"]["datasets"]

    if not slugs:
        print(f"[WARN] No Kaggle mirror is available for {meta['name']}.")
        print(meta["kaggle"]["note"])
        _print_manual_instructions(meta)
        return

    # Ensure kaggle package is installed
    if not _check_kaggle():
        if not _install_kaggle():
            print("[ERROR] Could not install the 'kaggle' package automatically.")
            print("        Please run:  pip install kaggle")
            sys.exit(1)

    # Ensure credentials are present
    if not _kaggle_credentials_exist():
        print("\n[ERROR] Kaggle API credentials not found.")
        print(textwrap.dedent("""
            To set up the Kaggle API (one-time):
              1. Log in at https://www.kaggle.com
              2. Go to Account -> API -> Create New Token
              3. Save kaggle.json to  ~/.kaggle/kaggle.json
              4. chmod 600 ~/.kaggle/kaggle.json   (Linux/macOS only)
        """))
        sys.exit(1)

    dest = out_dir / meta["target_subdir"]
    print(f"\n[INFO] Downloading {meta['name']} → {dest}")

    success = False
    for slug in slugs:
        print(f"[INFO] Trying Kaggle dataset: {slug}")
        if _download_kaggle_dataset(slug, dest):
            success = True
            break
        print(f"[WARN] Failed to download {slug}, trying next mirror…")

    if success:
        print(f"[OK]  {meta['name']} downloaded to {dest}")
        if meta["kaggle"]["note"]:
            print(f"[NOTE] {meta['kaggle']['note']}")
    else:
        print(f"[ERROR] All Kaggle mirrors failed for {meta['name']}.")
        _print_manual_instructions(meta)


def show_manual(key: str) -> None:
    """Print manual download instructions for a dataset.

    Args:
        key: Dataset key (``deap``, ``seediv``, or ``drozy``).
    """
    _print_manual_instructions(DATASETS[key])


def show_info(key: str) -> None:
    """Print summary information about a dataset.

    Args:
        key: Dataset key.
    """
    meta = DATASETS[key]
    print(f"\n{'=' * 60}")
    print(f"  {meta['name']}")
    print(f"{'=' * 60}")
    print(f"  {meta['description']}")
    print(f"\n  Primary URL : {meta['primary_url']}")
    if meta["kaggle"]["datasets"]:
        for slug in meta["kaggle"]["datasets"]:
            print(f"  Kaggle      : https://www.kaggle.com/datasets/{slug}")
    else:
        print(f"  Kaggle      : Not available ({meta['kaggle']['note']})")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        choices=["deap", "seediv", "drozy", "all"],
        required=True,
        help="Which dataset to download (or 'all' for all three).",
    )
    parser.add_argument(
        "--method",
        choices=["kaggle", "manual", "info"],
        default="kaggle",
        help=(
            "Download method: 'kaggle' uses the Kaggle API (recommended), "
            "'manual' prints step-by-step instructions, "
            "'info' shows dataset details and all download links."
        ),
    )
    parser.add_argument(
        "--out",
        default="./datasets",
        metavar="DIR",
        help="Root directory to store downloaded datasets (default: ./datasets).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    keys = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    out_dir = Path(args.out).expanduser().resolve()

    for key in keys:
        if args.method == "kaggle":
            download_via_kaggle(key, out_dir)
        elif args.method == "manual":
            show_manual(key)
        elif args.method == "info":
            show_info(key)


if __name__ == "__main__":
    main()
