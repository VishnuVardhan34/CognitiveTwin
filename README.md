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
> The official dataset websites can be hard to reach from India.  
> Use the **Kaggle / alternative methods** described below, or run:
> ```bash
> pip install kaggle
> # Set up ~/.kaggle/kaggle.json (see below), then:
> python scripts/download_datasets.py --dataset all --method kaggle --out ./datasets
> ```

#### DEAP Dataset

**Option A – Official (requires registration, may be blocked in some regions):**
1. Register at [eecs.qmul.ac.uk/mmv/datasets/deap](http://www.eecs.qmul.ac.uk/mmv/datasets/deap/download.html)
2. Download `data_preprocessed_python.zip`
3. Extract to `datasets/DEAP/data_preprocessed_python/`

**Option B – Kaggle (recommended for India / blocked regions):**
```bash
# Install Kaggle API (once)
pip install kaggle

# Download via Kaggle (requires a free account + API token)
kaggle datasets download -d laevitasimpl/deap-dataset-for-emotion-analysis \
    -p datasets/DEAP --unzip
```

**Option C – Request by e-mail:**
Send your name and institutional affiliation to **deap@eecs.qmul.ac.uk** —
the authors usually respond within a few days with a direct download link.

```bash
mkdir -p datasets/DEAP
unzip data_preprocessed_python.zip -d datasets/DEAP/
```

#### SEED-IV Dataset

**Option A – Official (requires registration, may be slow from India):**
1. Fill in the request form at [bcmi.sjtu.edu.cn/~seed/seed-iv.html](https://bcmi.sjtu.edu.cn/~seed/seed-iv.html#download-link) using an institutional e-mail.
2. Extract to `datasets/SEED-IV/`

**Option B – Kaggle:**
```bash
kaggle datasets download -d qiriro/seed-iv-eeg-emotion-recognition \
    -p datasets/SEED-IV --unzip
```

**Option C – Request by e-mail:**
Contact the BCMI lab at **bcmi@sjtu.edu.cn** with your institutional e-mail
for a Google Drive / direct link.

#### DROZY Dataset

**Option A – Official:**
1. Available at [drozy.ulg.ac.be](http://drozy.ulg.ac.be/) *(try a VPN if this is blocked)*
2. Extract to `datasets/DROZY/`

**Option B – Zenodo mirror:**
Check [https://zenodo.org/search?q=DROZY](https://zenodo.org/search?q=DROZY) for a
community-uploaded copy.

**Option C – Request by e-mail:**
Contact the DROZY dataset authors (University of Liège) at **drozy@ulg.ac.be** for
a Google Drive or alternative link.

> **See [`scripts/download_datasets.py`](scripts/download_datasets.py) for a
> fully-automated download helper that uses the Kaggle API and prints step-by-step
> manual instructions as a fallback.**

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
> websites are inaccessible or very slow.**

### Kaggle API Setup (one-time)

1. Create a free account at [kaggle.com](https://www.kaggle.com).
2. Go to **Account → Settings → API → Create New Token** – this downloads `kaggle.json`.
3. Place it at `~/.kaggle/kaggle.json` (Linux/macOS) or `C:\Users\<user>\.kaggle\kaggle.json` (Windows).
4. On Linux/macOS run `chmod 600 ~/.kaggle/kaggle.json`.
5. `pip install kaggle`

### Using the Download Helper Script

```bash
# Download a single dataset
python scripts/download_datasets.py --dataset deap   --method kaggle --out ./datasets
python scripts/download_datasets.py --dataset seediv --method kaggle --out ./datasets
python scripts/download_datasets.py --dataset drozy  --method manual  # no Kaggle mirror yet

# Download all three datasets at once
python scripts/download_datasets.py --dataset all --method kaggle --out ./datasets

# Print manual/email instructions for any dataset
python scripts/download_datasets.py --dataset deap --method manual

# Show all known links and mirrors
python scripts/download_datasets.py --dataset all --method info
```

### Direct Kaggle Links (browser download)

| Dataset | Kaggle Link |
|---------|-------------|
| DEAP | [kaggle.com/datasets/laevitasimpl/deap-dataset-for-emotion-analysis](https://www.kaggle.com/datasets/laevitasimpl/deap-dataset-for-emotion-analysis) |
| DEAP (alternate) | [kaggle.com/datasets/birdy654/eeg-brainwave-dataset-feeling-emotions](https://www.kaggle.com/datasets/birdy654/eeg-brainwave-dataset-feeling-emotions) |
| SEED-IV | [kaggle.com/datasets/qiriro/seed-iv-eeg-emotion-recognition](https://www.kaggle.com/datasets/qiriro/seed-iv-eeg-emotion-recognition) |
| DROZY | Not yet on Kaggle – use e-mail request (see below) |

### E-mail Requests (always works)

If Kaggle mirrors are unavailable, you can request datasets directly:

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
