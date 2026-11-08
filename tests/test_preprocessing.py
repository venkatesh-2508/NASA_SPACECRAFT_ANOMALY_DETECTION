"""
Tests: Preprocessing Pipeline
"""
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.preprocessing import TelemetryPreprocessor
from src.preprocessing.sequence_generator import generate_sequences, train_val_split


# ============================================================
# TelemetryPreprocessor
# ============================================================

def test_fit_transform_shape():
    data = np.random.randn(500, 3).astype(np.float32)
    pp = TelemetryPreprocessor()
    scaled = pp.fit_transform(data)
    assert scaled.shape == data.shape


def test_scaled_range():
    data = np.random.randn(500, 1).astype(np.float32)
    pp = TelemetryPreprocessor()
    scaled = pp.fit_transform(data)
    # Telemetry column should be in [0, 1]
    assert scaled[:, 0].min() >= -1e-6
    assert scaled[:, 0].max() <= 1.0 + 1e-6


def test_no_leakage():
    """Fit only on train, transform test — verify different value ranges don't crash."""
    train = np.random.randn(500, 2).astype(np.float32)
    test = np.random.randn(200, 2).astype(np.float32) * 10  # different scale
    pp = TelemetryPreprocessor()
    pp.fit(train)
    test_scaled = pp.transform(test)
    assert test_scaled.shape == test.shape


def test_inverse_transform():
    data = np.random.randn(300, 1).astype(np.float32)
    pp = TelemetryPreprocessor()
    scaled = pp.fit_transform(data)
    recovered = pp.inverse_transform_telemetry(scaled[:, 0])
    np.testing.assert_allclose(recovered, data[:, 0], atol=1e-5)


def test_transform_before_fit_raises():
    pp = TelemetryPreprocessor()
    with pytest.raises(RuntimeError):
        pp.transform(np.ones((10, 2)))


def test_save_load(tmp_path):
    data = np.random.randn(200, 2).astype(np.float32)
    pp = TelemetryPreprocessor()
    pp.fit(data)
    path = str(tmp_path / "scaler.pkl")
    pp.save(path)
    loaded = TelemetryPreprocessor.load(path)
    assert loaded.is_fitted
    np.testing.assert_allclose(loaded._tele_min, pp._tele_min)


# ============================================================
# Sequence Generator
# ============================================================

def test_generate_sequences_shapes():
    data = np.random.randn(1000, 3).astype(np.float32)
    X, y = generate_sequences(data, sequence_length=250)
    assert X.shape == (750, 250, 3)
    assert y.shape == (750,)


def test_generate_sequences_target():
    """The target y[i] should equal data[i + seq_len, 0]."""
    data = np.arange(600).reshape(600, 1).astype(np.float32)
    X, y = generate_sequences(data, sequence_length=100)
    for i in range(5):
        assert y[i] == pytest.approx(data[i + 100, 0])


def test_insufficient_data_raises():
    data = np.random.randn(100, 2).astype(np.float32)
    with pytest.raises(ValueError):
        generate_sequences(data, sequence_length=200)


def test_train_val_split_sizes():
    X = np.random.randn(1000, 250, 3)
    y = np.random.randn(1000)
    X_tr, y_tr, X_v, y_v = train_val_split(X, y, val_fraction=0.2)
    assert len(X_tr) == 800
    assert len(X_v) == 200
    assert len(y_tr) == 800
    assert len(y_v) == 200

