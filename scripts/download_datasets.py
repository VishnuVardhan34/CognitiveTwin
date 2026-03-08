#!/usr/bin/env python3
"""
Dataset download helper for CognitiveTwin.

This script provides alternative download methods for the DEAP, SEED-IV, and
DROZY datasets, which are particularly useful for users in regions where the
primary dataset websites are inaccessible (e.g. India).

Supported methods:
  - kaggle      (Kaggle API – requires free account + API token)
  - zenodo      (Zenodo open repository – works globally including India)
  - huggingface (Hugging Face Hub – works globally including India)
  - gdrive      (Google Drive via gdown – works globally including India)
  - manual      (prints step-by-step instructions and e-mail contacts)
  - info        (shows all known links and mirrors)

Usage::

    # Recommended for India / regions where Kaggle pages show "page not found":
    python scripts/download_datasets.py --dataset drozy  --method zenodo  --out ./datasets
    python scripts/download_datasets.py --dataset deap   --method huggingface --out ./datasets
    python scripts/download_datasets.py --dataset seediv --method huggingface --out ./datasets

    # Kaggle (if the specific dataset page loads for you):
    python scripts/download_datasets.py --dataset deap   --method kaggle --out ./datasets
    python scripts/download_datasets.py --dataset seediv --method kaggle --out ./datasets

    # Download all datasets at once (tries kaggle first, falls back to manual):
    python scripts/download_datasets.py --dataset all --method kaggle --out ./datasets

    # Print manual/email instructions for any dataset:
    python scripts/download_datasets.py --dataset deap --method manual

    # Show all known links and mirrors:
    python scripts/download_datasets.py --dataset all --method info

Kaggle API setup (one-time):
    1. Log in at https://www.kaggle.com and go to Account → API → Create New Token.
    2. Save the downloaded ``kaggle.json`` to ``~/.kaggle/kaggle.json``
       (Linux/macOS) or ``C:\\Users\\<user>\\.kaggle\\kaggle.json`` (Windows).
    3. ``chmod 600 ~/.kaggle/kaggle.json``  (Linux/macOS only)
    4. ``pip install kaggle``

Hugging Face setup (one-time):
    1. Create a free account at https://huggingface.co
    2. ``pip install huggingface_hub``
    3. ``huggingface-cli login``  (or set HF_TOKEN environment variable)

Google Drive / gdown setup (one-time):
    1. ``pip install gdown``
    2. Obtain the shareable Google Drive link or file-ID from the dataset authors.
    3. Run with  --gdrive-id <file-or-folder-id>

Note — "We can't find that page" on Kaggle:
    Specific Kaggle dataset pages are sometimes removed or renamed by their
    uploaders. If you see a 404 on a Kaggle dataset URL, use the Zenodo,
    Hugging Face, or Google Drive methods instead, or request the data
    directly by e-mail (see --method manual).
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import textwrap
import zipfile
from pathlib import Path
from urllib.parse import urlparse

# Message shown when a Kaggle dataset page cannot be found (404 / removed).
_KAGGLE_404_MSG = "We can't find that page"

# Allowed domain for Zenodo file downloads (security: reject redirects off zenodo.org).
_ZENODO_ALLOWED_DOMAIN = "zenodo.org"

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
            # Multiple Kaggle mirrors – tried in order; some may be removed.
            # If you see "We can't find that page" on Kaggle, use the
            # --method huggingface or --method zenodo alternatives instead.
            "datasets": [
                "laevitasimpl/deap-dataset-for-emotion-analysis",
                "birdy654/eeg-brainwave-dataset-feeling-emotions",
                "pranavagneecm/deap",
            ],
            "note": (
                "After download, look for 'data_preprocessed_python.zip' (or the "
                "directory) and move its contents to datasets/DEAP/data_preprocessed_python/. "
                f"If you see '{_KAGGLE_404_MSG}', the Kaggle dataset was removed – "
                "use --method huggingface or --method manual instead."
            ),
        },
        "huggingface": {
            # HuggingFace Hub datasets – accessible from India
            "datasets": [
                "SzLeaves/DEAP",
                "s3prl/DEAP",
            ],
            "note": (
                "Requires a free Hugging Face account and `pip install huggingface_hub`. "
                "Run `huggingface-cli login` once before downloading. "
                "Extract to datasets/DEAP/data_preprocessed_python/"
            ),
        },
        "zenodo": {
            # Zenodo records – open-access repository accessible from India
            "records": [],
            "note": (
                "DEAP is a restricted-access dataset and is not hosted on Zenodo. "
                "Please request access via e-mail or use the Hugging Face mirror."
            ),
        },
        "gdrive": {
            "ids": [],
            "note": (
                "The DEAP authors sometimes share a Google Drive link on request. "
                "E-mail deap@eecs.qmul.ac.uk and ask for a Drive link. "
                "Then re-run with: python scripts/download_datasets.py --dataset deap "
                "--method gdrive --gdrive-id <ID> --out ./datasets"
            ),
        },
        "manual": {
            "steps": [
                "Option A – Hugging Face (recommended for India):\n"
                "   pip install huggingface_hub\n"
                "   huggingface-cli login          # one-time setup\n"
                "   python scripts/download_datasets.py --dataset deap --method huggingface --out ./datasets",
                "Option B – Kaggle (if the dataset page is accessible):\n"
                "   pip install kaggle             # one-time setup\n"
                "   # Place ~/.kaggle/kaggle.json first, then:\n"
                "   python scripts/download_datasets.py --dataset deap --method kaggle --out ./datasets",
                "Option C – E-mail request (always works):\n"
                "   E-mail deap@eecs.qmul.ac.uk with your name and institutional affiliation.\n"
                "   The authors typically respond within a few days with a direct link.\n"
                "   Ask them specifically for a Google Drive link if you are in India.",
                "Option D – Official website (may be slow from India):\n"
                "   Register at http://www.eecs.qmul.ac.uk/mmv/datasets/deap/download.html\n"
                "   and download data_preprocessed_python.zip.",
                "After downloading, extract the archive to  datasets/DEAP/data_preprocessed_python/",
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
            # Multiple Kaggle mirrors – tried in order; some may be removed.
            "datasets": [
                "qiriro/seed-iv-eeg-emotion-recognition",
                "shayanfazeli/seed-iv",
            ],
            "note": (
                "Extract the downloaded archive so that the directory layout is "
                "datasets/SEED-IV/eeg_raw_data/{1,2,3}/<subject>.mat. "
                f"If you see '{_KAGGLE_404_MSG}', use --method huggingface instead."
            ),
        },
        "huggingface": {
            "datasets": [
                "SzLeaves/SEED-IV",
                "bcmi-sjtu/SEED-IV",
            ],
            "note": (
                "Requires a free Hugging Face account and `pip install huggingface_hub`. "
                "Run `huggingface-cli login` once before downloading. "
                "Extract to datasets/SEED-IV/eeg_raw_data/{1,2,3}/"
            ),
        },
        "zenodo": {
            "records": [],
            "note": (
                "SEED-IV requires institutional registration and is not on Zenodo. "
                "Please request access via the BCMI lab form or use the Hugging Face mirror."
            ),
        },
        "gdrive": {
            "ids": [],
            "note": (
                "The BCMI lab sometimes provides a Google Drive link after registration. "
                "Fill in the form at https://bcmi.sjtu.edu.cn/~seed/seed-iv.html#download-link "
                "and request a Drive link specifically. Then re-run with: "
                "python scripts/download_datasets.py --dataset seediv "
                "--method gdrive --gdrive-id <ID> --out ./datasets"
            ),
        },
        "manual": {
            "steps": [
                "Option A – Hugging Face (recommended for India):\n"
                "   pip install huggingface_hub\n"
                "   huggingface-cli login          # one-time setup\n"
                "   python scripts/download_datasets.py --dataset seediv --method huggingface --out ./datasets",
                "Option B – Kaggle (if the dataset page is accessible):\n"
                "   pip install kaggle             # one-time setup\n"
                "   python scripts/download_datasets.py --dataset seediv --method kaggle --out ./datasets",
                "Option C – BCMI lab registration (always works):\n"
                "   Fill in the form at https://bcmi.sjtu.edu.cn/~seed/seed-iv.html#download-link\n"
                "   using an institutional/university e-mail. Access is usually granted within\n"
                "   1–3 business days. Ask for a Google Drive link if you are in India.",
                "Option D – E-mail bcmi@sjtu.edu.cn with your affiliation.\n"
                "   Mention that you need an accessible mirror (e.g. Google Drive)\n"
                "   because the primary link is slow from India.",
                "After downloading, extract to  datasets/SEED-IV/",
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
            # DROZY does not currently have an official Kaggle mirror.
            "datasets": [],
            "note": "DROZY is not on Kaggle – use --method zenodo or --method manual instead.",
        },
        "huggingface": {
            "datasets": [],
            "note": "DROZY is not on Hugging Face Hub – use --method zenodo or --method manual instead.",
        },
        "zenodo": {
            # Zenodo record for the DROZY dataset (open access, accessible from India)
            "records": [
                {
                    "record_id": "1230005",
                    "doi": "10.5281/zenodo.1230005",
                    "url": "https://zenodo.org/record/1230005",
                    "description": "DROZY dataset – PSG EEG + KSS ratings (14 participants)",
                },
            ],
            "note": (
                "Zenodo is open-access and accessible worldwide including India. "
                "The download will land in datasets/DROZY/. "
                "Files will be downloaded directly from https://zenodo.org."
            ),
        },
        "gdrive": {
            "ids": [],
            "note": (
                "E-mail drozy@ulg.ac.be with your name and affiliation and ask for "
                "a Google Drive link. Then re-run with: "
                "python scripts/download_datasets.py --dataset drozy "
                "--method gdrive --gdrive-id <ID> --out ./datasets"
            ),
        },
        "manual": {
            "steps": [
                "Option A – Zenodo (recommended for India – open access, no login required):\n"
                "   python scripts/download_datasets.py --dataset drozy --method zenodo --out ./datasets",
                "Option B – Official University of Liège page:\n"
                "   Try http://drozy.ulg.ac.be/  (a VPN set to a European server may help\n"
                "   if your ISP blocks .ulg.ac.be).",
                "Option C – E-mail request:\n"
                "   E-mail drozy@ulg.ac.be with your name and affiliation.\n"
                "   Ask for a Zenodo, Google Drive, or Dropbox link.\n"
                "   The authors are usually responsive within a week.",
                "After downloading, extract to  datasets/DROZY/",
            ],
        },
        "target_subdir": "DROZY",
    },
}



# ---------------------------------------------------------------------------
# Helpers – Kaggle
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


# ---------------------------------------------------------------------------
# Helpers – Hugging Face Hub
# ---------------------------------------------------------------------------

def _check_huggingface() -> bool:
    """Return True if the huggingface_hub package is available."""
    try:
        import huggingface_hub  # noqa: F401  # type: ignore
        return True
    except ImportError:
        return False


def _install_huggingface() -> bool:
    """Attempt to install the huggingface_hub package via pip."""
    print("[INFO] Installing the 'huggingface_hub' package…")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "huggingface_hub", "--quiet"],
    )
    return result.returncode == 0


def _download_huggingface_dataset(repo_id: str, dest_dir: Path) -> bool:
    """Download a dataset from Hugging Face Hub.

    Args:
        repo_id: Hugging Face dataset identifier (``owner/dataset-name``).
        dest_dir: Local directory to download into.

    Returns:
        True on success, False otherwise.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download  # type: ignore
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(dest_dir),
        )
        return True
    except Exception as exc:
        print(f"[WARN] huggingface_hub download failed for {repo_id}: {exc}")
        return False


# ---------------------------------------------------------------------------
# Helpers – Zenodo
# ---------------------------------------------------------------------------

def _download_zenodo_record(record: dict, dest_dir: Path) -> bool:
    """Download all files from a Zenodo record.

    Args:
        record: Zenodo record metadata dict with keys ``record_id``, ``url``,
                and ``description``.
        dest_dir: Local directory to download files into.

    Returns:
        True on success, False otherwise.
    """
    import json
    import urllib.request

    record_id = record["record_id"]

    # Validate record_id to contain only digits (Zenodo IDs are numeric).
    if not re.fullmatch(r"\d+", record_id):
        print(f"[ERROR] Invalid Zenodo record ID: {record_id!r} – must be numeric.")
        return False

    api_url = f"https://zenodo.org/api/records/{record_id}"
    print(f"[INFO] Fetching Zenodo record metadata from {api_url}")

    try:
        with urllib.request.urlopen(api_url, timeout=30) as resp:  # noqa: S310
            data = json.loads(resp.read().decode())
    except Exception as exc:
        print(f"[ERROR] Could not fetch Zenodo record {record_id}: {exc}")
        return False

    files = data.get("files", [])
    if not files:
        print(f"[ERROR] No files found in Zenodo record {record_id}.")
        return False

    dest_dir.mkdir(parents=True, exist_ok=True)
    success = True
    for file_info in files:
        filename = file_info.get("key") or file_info.get("filename", "unknown")
        download_url = (
            file_info.get("links", {}).get("self")
            or file_info.get("links", {}).get("download")
        )
        if not download_url:
            print(f"[WARN] No download URL for file {filename} – skipping.")
            continue

        # Validate that the download URL is from the expected zenodo.org domain.
        parsed = urlparse(download_url)
        if parsed.netloc != _ZENODO_ALLOWED_DOMAIN and not parsed.netloc.endswith(f".{_ZENODO_ALLOWED_DOMAIN}"):
            print(
                f"[WARN] Unexpected download domain for {filename}: "
                f"{parsed.netloc!r} – skipping for security."
            )
            success = False
            continue

        dest_file = dest_dir / filename
        print(f"[INFO] Downloading {filename} from Zenodo → {dest_file}")
        try:
            urllib.request.urlretrieve(download_url, str(dest_file))  # noqa: S310
            print(f"[OK]   Saved {dest_file}")
        except Exception as exc:
            print(f"[ERROR] Failed to download {filename}: {exc}")
            success = False

    return success


# ---------------------------------------------------------------------------
# Helpers – Google Drive (gdown)
# ---------------------------------------------------------------------------

def _check_gdown() -> bool:
    """Return True if the gdown package is available."""
    try:
        import gdown  # noqa: F401  # type: ignore
        return True
    except ImportError:
        return False


def _install_gdown() -> bool:
    """Attempt to install the gdown package via pip."""
    print("[INFO] Installing the 'gdown' package…")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "gdown", "--quiet"],
    )
    return result.returncode == 0


def _download_gdrive(file_or_folder_id: str, dest_dir: Path, is_folder: bool = False) -> bool:
    """Download a file or folder from Google Drive using gdown.

    Args:
        file_or_folder_id: Google Drive file or folder ID.
        dest_dir: Local directory to download into.
        is_folder: If True, download an entire folder recursively.

    Returns:
        True on success, False otherwise.
    """
    if not _check_gdown():
        if not _install_gdown():
            print("[ERROR] Could not install the 'gdown' package automatically.")
            print("        Please run:  pip install gdown")
            return False

    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        import gdown  # type: ignore
        if is_folder:
            url = f"https://drive.google.com/drive/folders/{file_or_folder_id}"
            gdown.download_folder(url=url, output=str(dest_dir), quiet=False, use_cookies=False)
        else:
            url = f"https://drive.google.com/uc?id={file_or_folder_id}"
            gdown.download(url=url, output=str(dest_dir / ""), quiet=False, fuzzy=True)
        return True
    except Exception as exc:
        print(f"[ERROR] gdown download failed: {exc}")
        return False


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

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

            If you are in India and Kaggle dataset pages show '{_KAGGLE_404_MSG}',
            the dataset may have been removed from Kaggle. Try instead:
              python scripts/download_datasets.py --dataset {key} --method huggingface --out {out}
              python scripts/download_datasets.py --dataset {key} --method zenodo      --out {out}
              python scripts/download_datasets.py --dataset {key} --method manual
        """).format(key=key, out=out_dir))
        sys.exit(1)

    dest = out_dir / meta["target_subdir"]
    print(f"\n[INFO] Downloading {meta['name']} → {dest}")

    success = False
    for slug in slugs:
        print(f"[INFO] Trying Kaggle dataset: {slug}")
        if _download_kaggle_dataset(slug, dest):
            success = True
            break
        print(f"[WARN] Failed to download {slug} (may have been removed or renamed), trying next mirror…")

    if success:
        print(f"[OK]  {meta['name']} downloaded to {dest}")
        if meta["kaggle"]["note"]:
            print(f"[NOTE] {meta['kaggle']['note']}")
    else:
        print(f"[ERROR] All Kaggle mirrors failed for {meta['name']}.")
        print(
            f"\n[TIP]  If you see '{_KAGGLE_404_MSG}' on Kaggle, the dataset\n"
            "       was removed. Try the Hugging Face or Zenodo alternative:\n"
            f"         python scripts/download_datasets.py --dataset {key} --method huggingface\n"
            f"         python scripts/download_datasets.py --dataset {key} --method zenodo\n"
        )
        _print_manual_instructions(meta)


def download_via_huggingface(key: str, out_dir: Path) -> None:
    """Download a dataset via Hugging Face Hub.

    Hugging Face (huggingface.co) is accessible from India and most regions.

    Args:
        key: Dataset key (``deap``, ``seediv``, or ``drozy``).
        out_dir: Root output directory.
    """
    meta = DATASETS[key]
    hf_meta = meta.get("huggingface", {})
    repo_ids: list[str] = hf_meta.get("datasets", [])

    if not repo_ids:
        print(f"[WARN] No Hugging Face mirror is available for {meta['name']}.")
        print(hf_meta.get("note", ""))
        _print_manual_instructions(meta)
        return

    if not _check_huggingface():
        if not _install_huggingface():
            print("[ERROR] Could not install the 'huggingface_hub' package automatically.")
            print("        Please run:  pip install huggingface_hub")
            sys.exit(1)

    dest = out_dir / meta["target_subdir"]
    print(f"\n[INFO] Downloading {meta['name']} from Hugging Face → {dest}")

    success = False
    for repo_id in repo_ids:
        print(f"[INFO] Trying Hugging Face dataset: {repo_id}")
        if _download_huggingface_dataset(repo_id, dest):
            success = True
            break
        print(f"[WARN] Failed to download {repo_id}, trying next mirror…")

    if success:
        print(f"[OK]  {meta['name']} downloaded to {dest}")
        if hf_meta.get("note"):
            print(f"[NOTE] {hf_meta['note']}")
    else:
        print(f"[ERROR] All Hugging Face mirrors failed for {meta['name']}.")
        _print_manual_instructions(meta)


def download_via_zenodo(key: str, out_dir: Path) -> None:
    """Download a dataset from Zenodo (open-access, accessible from India).

    Args:
        key: Dataset key (``deap``, ``seediv``, or ``drozy``).
        out_dir: Root output directory.
    """
    meta = DATASETS[key]
    zenodo_meta = meta.get("zenodo", {})
    records: list[dict] = zenodo_meta.get("records", [])

    if not records:
        print(f"[WARN] No Zenodo record is available for {meta['name']}.")
        print(zenodo_meta.get("note", ""))
        _print_manual_instructions(meta)
        return

    dest = out_dir / meta["target_subdir"]
    print(f"\n[INFO] Downloading {meta['name']} from Zenodo → {dest}")
    print("[NOTE] Zenodo is open-access and works worldwide including India.")

    success = False
    for record in records:
        print(
            f"[INFO] Fetching Zenodo record {record['record_id']} "
            f"({record.get('description', '')}) – DOI: {record.get('doi', 'N/A')}"
        )
        if _download_zenodo_record(record, dest):
            success = True
            break
        print(f"[WARN] Zenodo record {record['record_id']} download failed, trying next…")

    if success:
        print(f"[OK]  {meta['name']} downloaded to {dest}")
        if zenodo_meta.get("note"):
            print(f"[NOTE] {zenodo_meta['note']}")
    else:
        print(f"[ERROR] Zenodo download failed for {meta['name']}.")
        _print_manual_instructions(meta)


def download_via_gdrive(key: str, out_dir: Path, gdrive_id: str, is_folder: bool = False) -> None:
    """Download a dataset from Google Drive using gdown.

    Google Drive is accessible from India. The authors of DEAP, SEED-IV, and
    DROZY will often share a Drive link on request (see --method manual).

    Args:
        key: Dataset key (``deap``, ``seediv``, or ``drozy``).
        out_dir: Root output directory.
        gdrive_id: Google Drive file or folder ID.
        is_folder: If True, download the entire folder recursively.
    """
    meta = DATASETS[key]
    gdrive_meta = meta.get("gdrive", {})

    if not gdrive_id:
        print(f"[WARN] No Google Drive ID provided for {meta['name']}.")
        print(gdrive_meta.get("note", ""))
        _print_manual_instructions(meta)
        return

    dest = out_dir / meta["target_subdir"]
    print(f"\n[INFO] Downloading {meta['name']} from Google Drive → {dest}")

    if _download_gdrive(gdrive_id, dest, is_folder=is_folder):
        print(f"[OK]  {meta['name']} downloaded to {dest}")
    else:
        print(f"[ERROR] Google Drive download failed for {meta['name']}.")
        _print_manual_instructions(meta)


def show_manual(key: str) -> None:
    """Print manual download instructions for a dataset.

    Args:
        key: Dataset key (``deap``, ``seediv``, or ``drozy``).
    """
    _print_manual_instructions(DATASETS[key])


def show_info(key: str) -> None:
    """Print summary information about a dataset including all mirrors.

    Args:
        key: Dataset key.
    """
    meta = DATASETS[key]
    print(f"\n{'=' * 60}")
    print(f"  {meta['name']}")
    print(f"{'=' * 60}")
    print(f"  {meta['description']}")
    print(f"\n  Primary URL : {meta['primary_url']}")

    # Kaggle
    if meta["kaggle"]["datasets"]:
        for slug in meta["kaggle"]["datasets"]:
            print(f"  Kaggle      : https://www.kaggle.com/datasets/{slug}")
    else:
        print(f"  Kaggle      : Not available ({meta['kaggle']['note']})")

    # Hugging Face
    hf = meta.get("huggingface", {})
    if hf.get("datasets"):
        for repo_id in hf["datasets"]:
            print(f"  HuggingFace : https://huggingface.co/datasets/{repo_id}")
    else:
        print(f"  HuggingFace : Not available ({hf.get('note', 'N/A')})")

    # Zenodo
    zenodo = meta.get("zenodo", {})
    if zenodo.get("records"):
        for record in zenodo["records"]:
            print(f"  Zenodo      : {record['url']}  (DOI: {record['doi']})")
            print(f"                {record.get('description', '')}")
    else:
        print(f"  Zenodo      : Not available ({zenodo.get('note', 'N/A')})")

    # Google Drive
    gdrive = meta.get("gdrive", {})
    if gdrive.get("ids"):
        for gid in gdrive["ids"]:
            print(f"  Google Drive: https://drive.google.com/drive/folders/{gid}")
    else:
        print(f"  Google Drive: {gdrive.get('note', 'Not available')}")

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
        choices=["kaggle", "huggingface", "zenodo", "gdrive", "manual", "info"],
        default="kaggle",
        help=(
            "Download method:\n"
            "  kaggle       – Kaggle API (requires free account + API token)\n"
            "  huggingface  – Hugging Face Hub (accessible from India, requires free account)\n"
            "  zenodo       – Zenodo open repository (no login required, accessible from India)\n"
            "  gdrive       – Google Drive via gdown (use --gdrive-id, accessible from India)\n"
            "  manual       – Print step-by-step instructions and e-mail contacts\n"
            "  info         – Show dataset details and all download links\n"
            "\n"
            f"India / 'page not found' tip: if Kaggle returns '{_KAGGLE_404_MSG}',\n"
            "use --method huggingface or --method zenodo instead."
        ),
    )
    parser.add_argument(
        "--out",
        default="./datasets",
        metavar="DIR",
        help="Root directory to store downloaded datasets (default: ./datasets).",
    )
    parser.add_argument(
        "--gdrive-id",
        default="",
        metavar="ID",
        help=(
            "Google Drive file or folder ID to download (used with --method gdrive). "
            "Obtain this ID from the dataset authors."
        ),
    )
    parser.add_argument(
        "--gdrive-folder",
        action="store_true",
        default=False,
        help="When using --method gdrive, treat the ID as a folder and download recursively.",
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
        elif args.method == "huggingface":
            download_via_huggingface(key, out_dir)
        elif args.method == "zenodo":
            download_via_zenodo(key, out_dir)
        elif args.method == "gdrive":
            download_via_gdrive(key, out_dir, args.gdrive_id, args.gdrive_folder)
        elif args.method == "manual":
            show_manual(key)
        elif args.method == "info":
            show_info(key)


if __name__ == "__main__":
    main()
