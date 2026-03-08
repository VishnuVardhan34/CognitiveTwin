# 🧠 CognitiveTwin

**Real-Time Digital Twin of Human Cognitive Load via EEG + Eye Tracking + HRV Fusion**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org)
[![React](https://img.shields.io/badge/React-18-61dafb.svg)](https://reactjs.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

CognitiveTwin is a multimodal biosignal processing system that estimates real-time
cognitive load using EEG scalograms, eye-tracking features, and HRV (PPG-derived)
data fused through a cross-modal transformer architecture with confidence-gated
Unscented Kalman Filter smoothing.

---

## 👥 Team

| Name | Student ID |
|------|------------|
| Sumanth Kotikalapudi | BT23CSH003 |
| Sai Charna Kukkala | BT23CSH002 |
| Sumeeth Kumar | BT23CSH030 |
| Vishnu Nutalapati | BT23CSH031 |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   CognitiveTwin Architecture                │
│                                                             │
│  EEG (32ch)        Eye Tracking (7 feat)   HRV/PPG (10 feat)│
│      │                     │                    │           │
│  CWT Scalogram    EOG/Gaze Features     RR Intervals        │
│ (32×64×T)          (T×7)                  (W×10)            │
│      │                     │                    │           │
│  EEGWaveletCNN    EyeTrackingNet           HRVNet           │
│  (EEGNet 2D-CNN)  (1D-CNN+BiLSTM+Attn)  (1D-CNN+GRU)      │
│      │                     │                    │           │
│     64d                   64d                  32d          │
│      └─────────────────────┴────────────────────┘          │
│                            │                                │
│              CrossModalAttention (MHA, 4 heads)             │
│                     (3×64 tokens)                           │
│                            │                                │
│              ModalityConfidenceGate (softmax)               │
│                    [w_eeg, w_eye, w_hrv]                    │
│                            │                                │
│               Weighted Fusion → 64-dim embedding            │
│                            │                                │
│          ┌─────────────────┴─────────────────┐             │
│     FusedClassifier              AV Regressor               │
│     (MLP 64→128→64→4)           (MLP 64→2, Tanh)           │
│          │                           │                      │
│     4-class logits           [arousal, valence]             │
│          │                                                   │
│   Weighted Decision Fusion (learnable [0.5, 0.2, 0.2, 0.1])│
│   final_probs = Σ dw_i · P(model_i)                        │
│          │                                                   │
│  ┌───────────────────────────────────────────────┐         │
│  │  Unscented Kalman Filter (4-D state vector)   │         │
│  │  [cognitive_load, arousal, valence, fatigue]  │         │
│  └───────────────────────────────────────────────┘         │
│          │                                                   │
│   WebSocket Server (2 Hz broadcast)                         │
│          │                                                   │
│   React Dashboard (D3 gauge, A-V plot, trajectory)          │
└─────────────────────────────────────────────────────────────┘
```

### Cognitive State Classes

| Class | Label | Description |
|-------|-------|-------------|
| Underload | 0 | Low arousal, low valence — insufficient mental engagement |
| Optimal | 1 | High arousal, high valence — peak performance zone |
| Overload | 2 | High arousal, low valence — mental fatigue from excess load |
| Fatigue | 3 | Very low arousal — drowsiness/exhaustion |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Node.js 18+ (for the React frontend)
- CUDA-capable GPU (optional but recommended)

### 1. Clone & Install

```bash
git clone https://github.com/VishnuVardhan34/CognitiveTwin.git
cd CognitiveTwin

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Or install as a package
pip install -e .
```

### 2. Download Datasets

> **⚠️ Users in India (or anywhere the primary links are inaccessible):**
> The official dataset websites (and sometimes specific Kaggle dataset pages) can be
> hard to reach from India, and Kaggle sometimes shows **"We can't find that page"**
> when a dataset has been removed or renamed.
>
> **Recommended India-friendly alternatives (no VPN needed):**
>
> | Dataset | Best Option | Command |
> |---------|-------------|---------|
> | DEAP    | Hugging Face | `python scripts/download_datasets.py --dataset deap --method huggingface --out ./datasets` |
> | SEED-IV | Hugging Face | `python scripts/download_datasets.py --dataset seediv --method huggingface --out ./datasets` |
> | DROZY   | Zenodo (open access) | `python scripts/download_datasets.py --dataset drozy --method zenodo --out ./datasets` |
>
> **One-time setup for Hugging Face:**
> ```bash
> pip install huggingface_hub
> huggingface-cli login    # free account at huggingface.co
> ```
>
> **If a dataset author sends you a Google Drive link**, use:
> ```bash
> pip install gdown
> python scripts/download_datasets.py --dataset deap --method gdrive \
>     --gdrive-id <FILE_OR_FOLDER_ID> --out ./datasets
> ```
>
> See **[Section: Alternative Dataset Download Methods](#-alternative-dataset-download-methods)**
> below for the full list of options.

#### DEAP Dataset

**Option A – Hugging Face (recommended for India – accessible worldwide):**
```bash
pip install huggingface_hub        # one-time setup
huggingface-cli login              # free account required
python scripts/download_datasets.py --dataset deap --method huggingface --out ./datasets
```

**Option B – Kaggle (if the dataset page loads in your region):**
```bash
# Install Kaggle API (once)
pip install kaggle

# Download via Kaggle (requires a free account + API token)
python scripts/download_datasets.py --dataset deap --method kaggle --out ./datasets
# or directly:
kaggle datasets download -d laevitasimpl/deap-dataset-for-emotion-analysis \
    -p datasets/DEAP --unzip
```

> **Kaggle "page not found" fix:** If `laevitasimpl/deap-dataset-for-emotion-analysis`
> returns a 404, the dataset was removed. The script automatically tries alternate slugs
> (`birdy654/eeg-brainwave-dataset-feeling-emotions`, `pranavagneecm/deap`).
> If all fail, switch to the Hugging Face method above.

**Option C – Official website (may require VPN from India):**
1. Register at [eecs.qmul.ac.uk/mmv/datasets/deap](http://www.eecs.qmul.ac.uk/mmv/datasets/deap/download.html)
2. Download `data_preprocessed_python.zip`
3. Extract to `datasets/DEAP/data_preprocessed_python/`

**Option D – Request by e-mail:**
Send your name and institutional affiliation to **deap@eecs.qmul.ac.uk** —
ask specifically for a **Google Drive or Hugging Face link** for India-compatible access.

```bash
mkdir -p datasets/DEAP
unzip data_preprocessed_python.zip -d datasets/DEAP/
```

#### SEED-IV Dataset

**Option A – Hugging Face (recommended for India):**
```bash
pip install huggingface_hub        # one-time setup
huggingface-cli login
python scripts/download_datasets.py --dataset seediv --method huggingface --out ./datasets
```

**Option B – Kaggle (if the dataset page loads):**
```bash
python scripts/download_datasets.py --dataset seediv --method kaggle --out ./datasets
# or directly:
kaggle datasets download -d qiriro/seed-iv-eeg-emotion-recognition \
    -p datasets/SEED-IV --unzip
```

> **Kaggle "page not found" fix:** The script tries `qiriro/seed-iv-eeg-emotion-recognition`
> and `shayanfazeli/seed-iv` as fallbacks. Use the Hugging Face method if all fail.

**Option C – Request by e-mail:**
Contact the BCMI lab at **bcmi@sjtu.edu.cn** with your institutional e-mail.
Ask for a **Google Drive link** for India-compatible access.

#### DROZY Dataset

**Option A – Zenodo (recommended for India – open access, no login required):**
```bash
python scripts/download_datasets.py --dataset drozy --method zenodo --out ./datasets
```
Direct Zenodo record: <https://zenodo.org/record/1230005> (DOI: 10.5281/zenodo.1230005)

**Option B – Google Drive (request from authors):**
1. E-mail **drozy@ulg.ac.be** asking for a Google Drive link.
2. Then run:
   ```bash
   pip install gdown
   python scripts/download_datasets.py --dataset drozy --method gdrive \
       --gdrive-id <ID_FROM_AUTHORS> --out ./datasets
   ```

**Option C – Official page (may need VPN from India):**
Available at [drozy.ulg.ac.be](http://drozy.ulg.ac.be/) *(try a VPN if this is blocked)*

> **See [`scripts/download_datasets.py`](scripts/download_datasets.py) for a
> fully-automated download helper that supports Kaggle, Hugging Face, Zenodo,
> Google Drive, and prints step-by-step manual instructions as a fallback.**

### 3. Configure

Edit `configs/default_config.yaml` to update dataset paths:

```yaml
data:
  deap_dir: "./datasets/DEAP/data_preprocessed_python"
  seediv_dir: "./datasets/SEED-IV"
  drozy_dir: "./datasets/DROZY"
```

### 4. Train

```bash
# Full training pipeline (Phase 1: EEG pretrain on SEED-IV, Phase 2-3: full fusion on DEAP)
python -m training.train_multimodal \
  --seediv-dir ./datasets/SEED-IV \
  --deap-dir ./datasets/DEAP/data_preprocessed_python \
  --epochs-pretrain 30 \
  --epochs-full 50 \
  --batch-size 32

# Or using the console entry point
cognitivetwin-train --help
```

### 5. Evaluate

```bash
# LOSO cross-validation
python -c "
from training.evaluate import loso_evaluation
results = loso_evaluation('./datasets/DEAP/data_preprocessed_python')
print(results)
"

# Ablation study
python -c "
from training.evaluate import ablation_study
ablation_study('./datasets/DEAP/data_preprocessed_python')
"
```

### 6. Export to ONNX

```bash
python training/export_onnx.py \
  --checkpoint checkpoints/cognitivetwin_model.pth \
  --output checkpoints/cognitivetwin.onnx
```

### 7. Start the Backend Server

```bash
python -m backend.websocket_server
# or
cognitivetwin-server
```

### 8. Start the Frontend

```bash
cd frontend
npm install
npm start
# Open http://localhost:3000
```

---

## 🌐 Alternative Dataset Download Methods

> **This section is for users in India or other regions where the official dataset
> websites are inaccessible, or where Kaggle shows "We can't find that page".**

### Why Kaggle sometimes shows "We can't find that page"

Kaggle dataset pages can disappear when:
- The uploader removes or renames the dataset.
- The dataset is flagged for copyright and taken down.
- Regional routing issues cause the page not to load.

If this happens, use the **Hugging Face Hub** (for DEAP / SEED-IV) or
**Zenodo** (for DROZY) methods – both are open, globally accessible repositories
that work reliably from India.

### Quick Reference – India-Accessible Methods

| Dataset | Method | Command |
|---------|--------|---------|
| DEAP    | Hugging Face | `python scripts/download_datasets.py --dataset deap --method huggingface --out ./datasets` |
| SEED-IV | Hugging Face | `python scripts/download_datasets.py --dataset seediv --method huggingface --out ./datasets` |
| DROZY   | Zenodo | `python scripts/download_datasets.py --dataset drozy --method zenodo --out ./datasets` |
| Any     | Google Drive | `python scripts/download_datasets.py --dataset <X> --method gdrive --gdrive-id <ID>` |
| Any     | Manual / e-mail | `python scripts/download_datasets.py --dataset <X> --method manual` |

### Hugging Face Hub (recommended for DEAP and SEED-IV)

Hugging Face (`huggingface.co`) is accessible from India and most regions globally.

**One-time setup:**
1. Create a free account at [huggingface.co](https://huggingface.co).
2. `pip install huggingface_hub`
3. `huggingface-cli login`  (or set the `HF_TOKEN` environment variable)

**Download:**
```bash
# DEAP
python scripts/download_datasets.py --dataset deap --method huggingface --out ./datasets

# SEED-IV
python scripts/download_datasets.py --dataset seediv --method huggingface --out ./datasets
```

### Zenodo (recommended for DROZY)

Zenodo (`zenodo.org`) is an open-access repository hosted by CERN.
It is accessible from India with no login required.

```bash
# DROZY – downloads directly from https://zenodo.org/record/1230005
python scripts/download_datasets.py --dataset drozy --method zenodo --out ./datasets
```

You can also download the DROZY archive manually from the browser:
<https://zenodo.org/record/1230005> → click **Download** on the `.zip` file.

### Google Drive via gdown (when dataset authors share a Drive link)

If the dataset authors respond to your e-mail with a Google Drive link,
use `gdown` (works reliably from India):

**One-time setup:**
```bash
pip install gdown
```

**Download:**
```bash
# Replace <ID> with the file/folder ID from the Drive share link
python scripts/download_datasets.py --dataset deap --method gdrive \
    --gdrive-id <ID> --out ./datasets

# For a shared folder, add --gdrive-folder
python scripts/download_datasets.py --dataset seediv --method gdrive \
    --gdrive-id <FOLDER_ID> --gdrive-folder --out ./datasets
```

### Kaggle API Setup (one-time)

1. Create a free account at [kaggle.com](https://www.kaggle.com).
2. Go to **Account → Settings → API → Create New Token** – this downloads `kaggle.json`.
3. Place it at `~/.kaggle/kaggle.json` (Linux/macOS) or `C:\Users\<user>\.kaggle\kaggle.json` (Windows).
4. On Linux/macOS run `chmod 600 ~/.kaggle/kaggle.json`.
5. `pip install kaggle`

### Using the Download Helper Script

```bash
# Download a single dataset (recommended India methods)
python scripts/download_datasets.py --dataset deap   --method huggingface --out ./datasets
python scripts/download_datasets.py --dataset seediv --method huggingface --out ./datasets
python scripts/download_datasets.py --dataset drozy  --method zenodo      --out ./datasets

# Kaggle (if pages are accessible)
python scripts/download_datasets.py --dataset deap   --method kaggle --out ./datasets
python scripts/download_datasets.py --dataset seediv --method kaggle --out ./datasets

# Google Drive (if the author provided an ID)
python scripts/download_datasets.py --dataset drozy --method gdrive \
    --gdrive-id <ID> --out ./datasets

# Print manual/email instructions for any dataset
python scripts/download_datasets.py --dataset deap --method manual

# Show all known links and mirrors
python scripts/download_datasets.py --dataset all --method info
```

### Direct Download Links (for browser download)

| Dataset | Source | Link |
|---------|--------|------|
| DEAP | Hugging Face | [huggingface.co/datasets/SzLeaves/DEAP](https://huggingface.co/datasets/SzLeaves/DEAP) |
| DEAP (alternate) | Kaggle | [kaggle.com/datasets/laevitasimpl/deap-dataset-for-emotion-analysis](https://www.kaggle.com/datasets/laevitasimpl/deap-dataset-for-emotion-analysis) |
| DEAP (alternate) | Kaggle | [kaggle.com/datasets/birdy654/eeg-brainwave-dataset-feeling-emotions](https://www.kaggle.com/datasets/birdy654/eeg-brainwave-dataset-feeling-emotions) |
| SEED-IV | Hugging Face | [huggingface.co/datasets/SzLeaves/SEED-IV](https://huggingface.co/datasets/SzLeaves/SEED-IV) |
| SEED-IV (alternate) | Kaggle | [kaggle.com/datasets/qiriro/seed-iv-eeg-emotion-recognition](https://www.kaggle.com/datasets/qiriro/seed-iv-eeg-emotion-recognition) |
| DROZY | Zenodo | [zenodo.org/record/1230005](https://zenodo.org/record/1230005) (DOI: 10.5281/zenodo.1230005) |

### E-mail Requests (always works)

If all automated methods fail, you can request datasets directly from the authors.
Ask specifically for a **Google Drive or Hugging Face link** for India-compatible access.

| Dataset | Contact |
|---------|---------|
| DEAP | deap@eecs.qmul.ac.uk — include your name and institution |
| SEED-IV | bcmi@sjtu.edu.cn — use an institutional/university e-mail |
| DROZY | drozy@ulg.ac.be — ask for a Google Drive or alternative link |

### Using a VPN

If you prefer downloading from the official sites, a free VPN (e.g.
[Windscribe](https://windscribe.com/), [ProtonVPN](https://protonvpn.com/)) can
bypass regional restrictions.  Connect to a European or US server, then use the
original links listed in **Section 2** above.

---

## 📁 Directory Structure

```
CognitiveTwin/
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
├── configs/
│   └── default_config.yaml        # All hyperparameters and paths
├── scripts/
│   └── download_datasets.py       # Alternative dataset downloader (Kaggle API)
├── data/
│   ├── __init__.py
│   └── dataset_loaders.py         # DEAP, SEED-IV, DROZY loaders
├── preprocessing/
│   ├── __init__.py
│   ├── eeg_preprocessor.py        # FIR bandpass, ICA, z-score
│   ├── eye_preprocessor.py        # Fixation/saccade, pupil detrending
│   ├── hrv_preprocessor.py        # PPG peak detection, HRV features
│   └── wavelet_transform.py       # CWT Morlet scalograms
├── models/
│   ├── __init__.py
│   ├── eeg_branch.py              # EEGWaveletCNN (spectral path)
│   ├── eye_branch.py              # 1D-CNN + BiLSTM + Attention
│   ├── hrv_branch.py              # 1D-CNN + GRU (HRV path)
│   ├── cross_attention.py         # Cross-modal transformer attention
│   ├── confidence_gate.py         # Modality confidence gating
│   └── multimodal_fusion.py       # Full fusion model + declare_state
├── state_estimation/
│   ├── __init__.py
│   └── ukf.py                     # UKF with neural transition model
├── training/
│   ├── __init__.py
│   ├── losses.py                  # Uncertainty-weighted multi-task loss
│   ├── train_multimodal.py        # Full training pipeline (phases 1–4)
│   ├── evaluate.py                # LOSO evaluation + ablation study
│   └── export_onnx.py             # Export to ONNX
├── backend/
│   ├── __init__.py
│   └── websocket_server.py        # WebSocket server + adaptive UI policy
├── frontend/
│   ├── package.json
│   ├── public/index.html
│   └── src/
│       ├── App.jsx                # Main dashboard
│       ├── App.css                # Dark-theme adaptive styling
│       ├── index.js
│       └── components/
│           ├── CognitiveGauge.jsx      # D3 radial gauge
│           ├── ArousalValencePlot.jsx  # 2D A-V scatter
│           ├── TrajectoryChart.jsx     # 30s trajectory
│           └── AlertOverlay.jsx        # Fatigue/overload alerts
├── pipeline/
│   ├── __init__.py
│   └── orchestrator.py            # End-to-end real-time pipeline
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py
│   ├── test_models.py
│   ├── test_fusion.py
│   └── test_ukf.py
└── notebooks/
    ├── 01_data_exploration.ipynb
    └── 02_model_training_demo.ipynb
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test module
pytest tests/test_models.py -v
pytest tests/test_fusion.py -v
pytest tests/test_preprocessing.py -v
pytest tests/test_ukf.py -v
```

---

## 📊 Performance Targets

| Metric | Target |
|--------|--------|
| LOSO Accuracy | ≥ 85% |
| End-to-end latency | ≤ 200 ms |
| Broadcast rate | 2 Hz |

---

## 🔬 Technical Details

### Signal Processing Pipeline

1. **EEG**: FIR bandpass (0.5–45 Hz) → FastICA artefact removal → Z-score normalisation → CWT Morlet scalogram (64 freq × T time)
2. **Eye Tracking**: Gaze position + pupil diameter → velocity-based fixation/saccade detection → 7-dim feature vector per timestep
3. **HRV**: PPG peak detection → RR intervals → 10-dim feature vector (5 time-domain + 5 frequency-domain)

### Model Architecture

- **EEG Branch**: EEGNet-inspired 2D CNN operating on scalogram images → 64-dim
- **Eye Branch**: 1D-CNN + Bidirectional LSTM + attention pooling → 64-dim
- **HRV Branch**: 1D-CNN + GRU (last hidden state) → 32-dim
- **Cross-Modal Attention**: MHA (4 heads) over 3-token sequence → 3×64
- **Confidence Gate**: Per-modality MLP scorers → softmax weights [w_eeg, w_eye, w_hrv]
- **Decision Fusion**: Learnable weights [fused, eeg, eye, hrv] → final class probabilities

### Loss Function

Multi-task uncertainty-weighted loss (Kendall et al., 2018):

```
L = (1/σ₁²)·L_cls + (1/σ₂²)·L_reg + (1/σ₃²)·L_aux + (1/σ₄²)·L_agree
    + log σ₁ + log σ₂ + log σ₃ + log σ₄
```

Where σᵢ are learnable task-specific uncertainty parameters.

### State Estimation

Unscented Kalman Filter with 4-dimensional state vector:
- `[cognitive_load, arousal, valence, fatigue_index]`
- Observations: 64-dim fused neural embedding
- Transition: Simple mean-reverting model or learned Transformer

---

## 🖥️ Frontend Dashboard

The React dashboard includes:

| Component | Description |
|-----------|-------------|
| `CognitiveGauge` | D3 arc gauge with needle indicator and class probability bars |
| `ArousalValencePlot` | 2D scatter in arousal-valence space with history trail |
| `TrajectoryChart` | 30-second step-line chart of cognitive state evolution |
| `AlertOverlay` | Modal alert for overload/fatigue detection with auto-dismiss |

### Adaptive UI Modes

- **Normal**: Full dashboard layout
- **Simplified** (Overload): Collapse secondary panels, enlarge critical elements
- **Engagement** (Underload): Show engagement cues, increase information density
- **Alert** (Fatigue): Break reminder overlay with auditory/visual cues

---

## 🤝 Contributing

### Branching Strategy

```
main            ← stable releases only
develop         ← integration branch
feature/<name>  ← feature development
fix/<name>      ← bug fixes
```

### Code Style

- Python: PEP 8, type hints on all public functions, NumPy-style docstrings
- JavaScript/JSX: ES2022+, functional components with hooks
- Commits: Conventional Commits (`feat:`, `fix:`, `docs:`, `test:`)

### Pull Request Process

1. Branch from `develop`
2. Write/update tests for your changes
3. Ensure `pytest tests/ -v` passes
4. Open PR against `develop` with a clear description

---

## 📚 References

1. Lawhern, V. J., et al. (2018). EEGNet: A compact convolutional neural network for EEG-based brain–computer interfaces. *Journal of Neural Engineering*.
2. Kendall, A., Gal, Y., & Cipolla, R. (2018). Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics. *CVPR 2018*.
3. Koelstra, S., et al. (2012). DEAP: A Database for Emotion Analysis Using Physiological Signals. *IEEE Transactions on Affective Computing*.
4. Liu, W., et al. (2018). SEED-IV: Emotion Recognition from Multi-Channel EEG Data. *IEEE Transactions on Cybernetics*.
5. Julier, S. J., & Uhlmann, J. K. (1997). A New Extension of the Kalman Filter to Nonlinear Systems. *SPIE Defense, Security, and Sensing*.
6. Vaswani, A., et al. (2017). Attention Is All You Need. *NeurIPS 2017*.

---

## 📄 License

This project is licensed under the MIT License.

```
MIT License

Copyright (c) 2024 Sumanth Kotikalapudi, Sai Charna Kukkala,
                   Sumeeth Kumar, Vishnu Nutalapati

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
