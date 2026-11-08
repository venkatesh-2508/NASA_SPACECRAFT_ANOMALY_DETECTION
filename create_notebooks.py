"""
Create the 4 Jupyter notebooks with rich exploratory code and markdown.
"""
import json
from pathlib import Path

def create_notebook(cells, filename):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.14.5"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    out_dir = Path("notebooks")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / filename, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

def md(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    }

def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.strip().split("\n")]
    }

# 1. 01_EDA.ipynb
eda_cells = [
    md("# 01 - Exploratory Data Analysis (EDA)\n## NASA Spacecraft Anomaly Detection (SMAP & MSL)\n\nThis notebook analyzes the NASA telemetry dataset, inspecting channel dimensions, operating characteristics, and expert anomaly intervals."),
    code("""import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to sys.path
PROJECT_ROOT = Path("..").resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_loader import DataLoader
from src.data.data_validation import DataValidator

loader = DataLoader(str(PROJECT_ROOT / "data"))
validator = DataValidator(loader)
report = validator.validate_all()
validator.print_report(report)"""),
    md("### 1. Anomaly Labels Breakdown\nInspect distribution across spacecraft (SMAP vs MSL) and anomaly classes (point vs contextual)."),
    code("""df_labels = loader.load_labels()
print(f"Total labeled channels: {len(df_labels)}")
df_labels.head(10)"""),
    md("### 2. Telemetry Channel Inspection (SMAP & MSL)"),
    code("""smap_chs = loader.get_smap_channels()
msl_chs = loader.get_msl_channels()
print(f"SMAP channels: {len(smap_chs)} | MSL channels: {len(msl_chs)}")

# Load sample channel
sample_ch = "P-1"
train_data = loader.load_train(sample_ch)
test_data = loader.load_test(sample_ch)
print(f"Channel {sample_ch} - Train: {train_data.shape}, Test: {test_data.shape}")"""),
    md("### 3. Visualizing Telemetry with Overlaid Anomaly Windows"),
    code("""fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(test_data[:, 0], label="Telemetry Value", color="#38BDF8", lw=1)

anomalies = loader.get_channel_labels(sample_ch)
for i, (start, end) in enumerate(anomalies):
    ax.axvspan(start, end, color="#EF4444", alpha=0.3, label="Anomaly Window" if i == 0 else "")

ax.set_title(f"Test Telemetry: Channel {sample_ch} with Ground-Truth Anomalies")
ax.set_xlabel("Time Step")
ax.set_ylabel("Telemetry Value")
ax.legend()
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.show()""")
]
create_notebook(eda_cells, "01_EDA.ipynb")

# 2. 02_Data_Preprocessing.ipynb
prep_cells = [
    md("# 02 - Data Preprocessing & Sequence Generation\n\nImplements leakage-free feature scaling and 250-timestep sliding window generation."),
    code("""import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path("..").resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_loader import DataLoader
from src.preprocessing.preprocessing import TelemetryPreprocessor
from src.preprocessing.sequence_generator import generate_sequences, train_val_split

loader = DataLoader(str(PROJECT_ROOT / "data"))
train_raw = loader.load_train("P-1")
test_raw = loader.load_test("P-1")

print("Raw Train Shape:", train_raw.shape)
print("Raw Test Shape:", test_raw.shape)"""),
    md("### Fit MinMax Scaler on Train Only"),
    code("""preprocessor = TelemetryPreprocessor(scale_telemetry=True)
train_scaled = preprocessor.fit_transform(train_raw)
test_scaled = preprocessor.transform(test_raw)

print(f"Train Telemetry Range: [{train_scaled[:, 0].min():.4f}, {train_scaled[:, 0].max():.4f}]")
print(f"Test Telemetry Range: [{test_scaled[:, 0].min():.4f}, {test_scaled[:, 0].max():.4f}]")"""),
    md("### Sliding Window Sequence Generation (L = 250)"),
    code("""X_train, y_train = generate_sequences(train_scaled, sequence_length=250)
X_val_tr, y_val_tr, X_val, y_val = train_val_split(X_train, y_train, val_fraction=0.2)

print("X_train sequences:", X_val_tr.shape)
print("y_train targets:", y_val_tr.shape)
print("X_val sequences:", X_val.shape)
print("y_val targets:", y_val.shape)""")
]
create_notebook(prep_cells, "02_Data_Preprocessing.ipynb")

# 3. 03_Model_Experiments.ipynb
exp_cells = [
    md("# 03 - Model Experiments: 2-Layer LSTM\n\nBuilds the PyTorch LSTM architecture (80 hidden units, 0.3 dropout) and inspects its parameter profile."),
    code("""import sys
from pathlib import Path
import torch

PROJECT_ROOT = Path("..").resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.lstm_model import LSTMAnomalyDetector
from src.utils.config import load_config

cfg = load_config(str(PROJECT_ROOT / "config.yaml"))
n_features = 25  # Channel P-1 has 25 features

model = LSTMAnomalyDetector(
    input_size=n_features,
    hidden_size=80,
    num_layers=2,
    dropout=0.3
)
print(model)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params:,}")"""),
    md("### Forward Pass Verification"),
    code("""dummy_input = torch.randn(16, 250, n_features)
output = model(dummy_input)
print("Output shape:", output.shape)
assert output.shape == (16,)
print("Forward pass successful!")""")
]
create_notebook(exp_cells, "03_Model_Experiments.ipynb")

# 4. 04_Model_Evaluation.ipynb
eval_cells = [
    md("# 04 - Model Evaluation & Performance Benchmark\n\nCompares event-level metrics between Z-score statistical baseline and LSTM with dynamic thresholding."),
    code("""import sys
from pathlib import Path
import json

PROJECT_ROOT = Path("..").resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import event_level_metrics
from src.utils.helpers import load_json

metrics_dir = PROJECT_ROOT / "results" / "metrics"
baseline_path = metrics_dir / "baseline_report.json"
lstm_path = metrics_dir / "evaluation_report.json"

if baseline_path.exists():
    baseline_rep = load_json(str(baseline_path))
    print("Baseline Overall:", baseline_rep.get("overall", {}))

if lstm_path.exists():
    lstm_rep = load_json(str(lstm_path))
    print("LSTM Overall:", lstm_rep.get("overall", {}))""")
]
create_notebook(eval_cells, "04_Model_Evaluation.ipynb")

print("Created all 4 notebooks in notebooks/ directory.")

