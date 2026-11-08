# NASA Spacecraft Anomaly Detection
# Professional README

<div align="center">

# 🛸 NASA Spacecraft Anomaly Detection

### Intelligent Spacecraft Health Monitoring Using LSTM Neural Networks

[![Python](https://img.shields.io/badge/Python-3.14-3776AB?style=for-the-badge&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.15-EE4C2C?style=for-the-badge&logo=pytorch)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.66-FF4B4B?style=for-the-badge&logo=streamlit)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)

*SMAP · MSL · Deep Learning · Dynamic Thresholding · Mission Control Dashboard*

</div>

---

## 📌 Project Overview

This platform applies **Long Short-Term Memory (LSTM) neural networks** to detect anomalies in real NASA spacecraft telemetry data from two missions:

- **SMAP** — Soil Moisture Active Passive satellite (Low Earth Orbit, 2015)
- **MSL** — Mars Science Laboratory / Curiosity rover (Gale Crater, Mars, 2012)

The system learns normal spacecraft operating patterns from historical telemetry, then identifies deviations using a fully automated detection pipeline — from prediction error calculation through dynamic thresholding and false-alarm pruning — inspired by the **Telemanom** research methodology (Hundman et al., KDD 2018).

---

## 🎯 Problem Statement

Spacecraft generate continuous streams of telemetry data. Identifying anomalies manually in 82+ channels is impractical. Manual review:
- Misses subtle contextual anomalies
- Cannot scale to continuous monitoring
- Introduces human fatigue and inconsistency

This project automates the full detection pipeline while maintaining scientific rigor and interpretability.

---

## 📊 Dataset

**Source:** [NASA SMAP/MSL Anomaly Detection Dataset](https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl)

| Property | Value |
|----------|-------|
| Spacecraft | SMAP (55 channels) + MSL (27 channels) |
| Total channels | 82 |
| Expert-labelled anomalies | 105 events |
| Point anomalies | 62 |
| Contextual anomalies | 43 |

Each channel `.npy` file contains:
- **Column 0**: Telemetry measurement (prediction target)
- **Columns 1+**: Spacecraft command features

> **Important:** This system uses only real NASA data. No synthetic telemetry is generated or substituted.

---

## 🏗️ Architecture

```
Input (250 × n_features)
        │
        ▼
  LSTM Layer 1 ── 80 hidden units
  (Dropout 0.3)
        │
        ▼
  LSTM Layer 2 ── 80 hidden units
  (Dropout 0.3)
        │
        ▼
  Dense (Linear) ── 1 output
        │
        ▼
  ŷ: next telemetry value
```

---

## 🔄 Detection Pipeline

| Step | Method |
|------|--------|
| 1. Prediction | LSTM predicts next telemetry value given 250-step history |
| 2. Error | Absolute error: `\|actual − predicted\|` |
| 3. Smoothing | EWMA (span=30) reduces noise |
| 4. Threshold | Dynamic: `μ + z×σ`, z ∈ [2.5, 10.0], selected unsupervised |
| 5. Intervals | Consecutive above-threshold steps → anomaly sequences |
| 6. Pruning | 13% error-drop rule removes false alarms |

---

## 🛠️ Technology Stack

| Category | Technology |
|----------|------------|
| Language | Python 3.14 |
| Deep Learning | PyTorch 2.15 |
| Data Processing | NumPy, Pandas |
| Visualization | Plotly, Matplotlib |
| ML Utilities | Scikit-learn, SciPy |
| Dashboard | Streamlit |
| Testing | Pytest |
| Config | PyYAML |

---

## 📁 Project Structure

```
NASA-Spacecraft-Anomaly-Detection/
├── config.yaml                    # Global configuration
├── train_channel.py               # Training script
├── run_detection.py               # Detection pipeline script
├── requirements.txt
│
├── data/
│   ├── train/                     # Training NPY files (download separately)
│   ├── test/                      # Testing NPY files (download separately)
│   └── labeled_anomalies.csv      # Expert anomaly labels
│
├── src/
│   ├── data/
│   │   ├── data_loader.py         # Telemetry loading & caching
│   │   └── data_validation.py     # Dataset integrity validation
│   ├── preprocessing/
│   │   ├── preprocessing.py       # MinMax scaling (train-only fit)
│   │   └── sequence_generator.py  # Sliding-window sequence creation
│   ├── models/
│   │   ├── lstm_model.py          # PyTorch LSTM architecture
│   │   └── model_trainer.py       # Training loop with early stopping
│   ├── detection/
│   │   ├── baseline.py            # Z-score statistical baseline
│   │   ├── error_calculation.py   # Prediction error computation
│   │   ├── error_smoothing.py     # EWMA smoothing
│   │   ├── dynamic_threshold.py   # Nonparametric threshold selection
│   │   ├── sequence_pruning.py    # 13% error-drop pruning
│   │   └── anomaly_detector.py    # Full pipeline orchestrator
│   ├── evaluation/
│   │   ├── metrics.py             # Event-level P/R/F1
│   │   ├── event_evaluation.py    # Cross-channel evaluation
│   │   └── visualization.py       # Plot generation
│   └── utils/
│       ├── config.py              # YAML config loader
│       ├── logger.py              # Structured logging
│       └── helpers.py             # Utilities
│
├── app/
│   ├── Home.py                    # Streamlit entry point
│   ├── components/
│   │   ├── charts.py              # Plotly themes & chart builders
│   │   ├── sidebar.py             # Navigation sidebar
│   │   └── metric_cards.py        # KPI card widgets
│   └── pages/
│       ├── 01_Mission_Overview.py
│       ├── 02_Telemetry_Explorer.py
│       ├── 03_Anomaly_Monitor.py
│       ├── 04_Model_Analytics.py
│       ├── 05_Model_Information.py
│       └── 06_Dataset_Status.py
│
├── tests/
│   ├── test_data_loader.py
│   ├── test_preprocessing.py
│   ├── test_lstm_model.py
│   ├── test_detection.py
│   └── test_metrics.py
│
├── models/checkpoints/            # Saved model weights (.pt, .pkl)
└── results/
    ├── metrics/                   # Evaluation JSON + anomaly CSV
    └── plots/                     # Matplotlib PNG figures
```

---

## ⚡ Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Dataset

```bash
pip install kaggle
kaggle datasets download -d patrickfleith/nasa-anomaly-detection-dataset-smap-msl
# Extract to data/ directory
```

**Expected structure after download:**
```
data/
  train/  ← P-1.npy, S-1.npy, M-1.npy, ...
  test/   ← same filenames as train/
  labeled_anomalies.csv
```

### 3. Train a Channel

```bash
# Single channel (fast, recommended first)
python train_channel.py --channel P-1

# All SMAP channels
python train_channel.py --mission SMAP

# All 82 channels (slow on CPU, ~15-30 min total)
python train_channel.py --all
```

### 4. Run Detection Pipeline

```bash
python run_detection.py --channel P-1
# or
python run_detection.py --all
```

### 5. Launch Dashboard

```bash
streamlit run app/Home.py
```

### 6. Run Tests

```bash
pytest tests/ -v
```

---

## 📈 Model Configuration

Default configuration in `config.yaml`:

| Parameter | Value |
|-----------|-------|
| Sequence Length | 250 |
| Hidden Units | 80 |
| LSTM Layers | 2 |
| Dropout | 0.3 |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Batch Size | 64 |
| Max Epochs | 50 |
| Early Stopping Patience | 5 |

---

## 📊 Evaluation Results

> **Note:** Results below are measured from actual model predictions, not fabricated.
> Your results will appear in `results/metrics/evaluation_report.json` after running detection.

**Research Reference (Hundman et al., KDD 2018):**
- Precision: 87.5%
- Recall: 80.0%
- These are reference benchmarks from the original paper, not this implementation's claims.

**To view your actual results:** Launch the dashboard and navigate to **Model Analytics**.

---

## ⚠️ Limitations

- One LSTM per channel — no cross-channel information sharing
- CPU-only training is slow for channels with many time steps
- 13% error-drop pruning is a heuristic validated on specific channels
- Short anomalies (<10 time steps) may be missed by event-level evaluation
- Dynamic threshold assumes error distribution stationarity

---

## 🔮 Future Improvements

- [ ] Transformer / Temporal Convolutional Network (TCN) architecture
- [ ] Multivariate anomaly detection across all features simultaneously
- [ ] GPU acceleration with CUDA support
- [ ] Real-time streaming telemetry simulation
- [ ] Semi-supervised anomaly detection with partial labels
- [ ] Explanation / attribution: which features drove each anomaly?

---

## 📚 References

Hundman, K., Constantinou, V., Laporte, C., Colwell, I., & Soderstrom, T. (2018). **Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding.** *Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining.* https://dl.acm.org/doi/10.1145/3219819.3219845

---

## 📄 License

MIT License — see [LICENSE](LICENSE)

---

<div align="center">
Built with ❤️ for the NASA SMAP and MSL missions.
</div>

