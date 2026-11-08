# NASA Spacecraft Anomaly Detection — Project Progress

## Project Identity
- **Name:** NASA Spacecraft Anomaly Detection
- **Subtitle:** Intelligent Spacecraft Health Monitoring Using LSTM Neural Networks
- **Core Tech:** Python 3.14, PyTorch 2.15, Streamlit, Plotly, Scikit-learn
- **Missions:** SMAP (Soil Moisture Active Passive) & MSL (Curiosity Rover)

---

## Phase Checklist & Status

| Phase | Description | Status | Verification / Artifacts |
|---|---|:---:|---|
| **Phase 1: Environment & Setup** | Python, PyTorch, Plotly, Streamlit, directory hierarchy, validation utilities | ✅ Completed | 44/44 unit tests passing, `download_dataset.py`, `data_validation.py` |
| **Phase 2: Exploratory Data Analysis** | Analyze dimensions, channels, distributions, normal vs anomalous telemetry | ✅ Completed | `notebooks/01_EDA.ipynb`, `src/evaluation/visualization.py` |
| **Phase 3: Data Preprocessing** | Train-only MinMax scaling, leakage prevention, 250-step sliding windows | ✅ Completed | `src/preprocessing/preprocessing.py`, `sequence_generator.py` |
| **Phase 4: Statistical Baseline** | Z-score anomaly detector (`Z > 3.0`), event-level metrics | ✅ Completed | `src/detection/baseline.py`, `run_baseline.py` |
| **Phase 5: LSTM Model Development** | 2-layer LSTM (80 hidden units, dropout 0.3), Adam, MSE loss, early stopping | ✅ Completed | `src/models/lstm_model.py`, `model_trainer.py`, `train_channel.py` |
| **Phase 6: Anomaly Detection Engine** | Prediction error, EWMA smoothing (span 30), dynamic thresholding, 13% pruning | ✅ Completed | `src/detection/` (all modules), `run_detection.py` |
| **Phase 7: Model Evaluation** | Event-level overlap evaluation, SMAP vs MSL breakdown, comparison with baseline | ✅ Completed | `src/evaluation/metrics.py`, `event_evaluation.py` |
| **Phase 8: Premium UI/UX Dashboard** | Dark aerospace mission control UI, 6 dedicated pages, Plotly interactivity | ✅ Completed | `app/Home.py`, `app/pages/*.py`, `app/components/*.py` |
| **Phase 9: Performance Optimization** | Cached checkpoint loading, vectorized numpy, streaming charts | ✅ Completed | Streamlit caching, fast loading |
| **Phase 10: Testing & QA** | Automated pytest suite covering loader, preprocessing, model, detection, metrics | ✅ Completed | 44 passing unit tests in `tests/` |
| **Phase 11: Documentation** | Professional README, architecture diagrams, mathematical formulation | ✅ Completed | `README.md`, `LICENSE`, `PROJECT_PROGRESS.md` |
| **Phase 12: Deployment** | Config management, requirements.txt, Streamlit Cloud readiness | ✅ Completed | `requirements.txt`, `config.yaml`, ready for cloud launch |

---

## Pipeline Architecture Details

1. **Preprocessing:**
   - MinMax scaler fit strictly on training split (column 0: telemetry value).
   - Pass-through for command columns.
   - Sliding window of length $L = 250$ time steps.
   - Chronological validation split ($80\%$ train, $20\%$ validation).

2. **Model:**
   - 2-layer LSTM with 80 hidden units per layer and $0.3$ dropout.
   - Linear head mapping to scalar next telemetry prediction $\hat{y}_t$.
   - Loss: Mean Squared Error (MSE), optimized via Adam ($lr = 0.001$).
   - Early stopping on validation loss with patience of 5 epochs.

3. **Detection Engine:**
   - Raw Error: $e_t = |y_t - \hat{y}_t|$
   - EWMA Smoothing: $s_t = \alpha e_t + (1 - \alpha) s_{t-1}$ with span $= 30$.
   - Dynamic Threshold: $\epsilon = \mu(s) + z \cdot \sigma(s)$ for $z \in [2.5, 10.0]$ step $0.5$, scored by sequence separation objective without ground truth labels.
   - Sequence generation: grouping contiguous violations into $[t_{start}, t_{end}]$.
   - Pruning: False alarm pruning requiring $\ge 13\%$ error drop upon sequence omission.

4. **Event-Level Evaluation:**
   - An anomaly is marked detected (True Positive) if a predicted sequence overlaps a ground truth interval.
   - Metrics computed: Precision, Recall, $F_1$-Score, TP, FP, FN.

