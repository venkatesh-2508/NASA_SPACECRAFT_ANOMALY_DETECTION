"""
Tests: Detection Pipeline
"""
import sys
from pathlib import Path
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.error_calculation import compute_prediction_error
from src.detection.error_smoothing import ewma_smooth
from src.detection.dynamic_threshold import select_dynamic_threshold
from src.detection.sequence_pruning import prune_sequences
from src.detection.baseline import ZScoreBaseline


# ============================================================
# Error Calculation
# ============================================================

def test_error_calculation_shape():
    actual = np.random.randn(500).astype(np.float32)
    pred = np.random.randn(500).astype(np.float32)
    err = compute_prediction_error(actual, pred)
    assert err.shape == (500,)
    assert (err >= 0).all()


def test_error_shape_mismatch():
    with pytest.raises(ValueError):
        compute_prediction_error(np.ones(100), np.ones(200))


# ============================================================
# EWMA Smoothing
# ============================================================

def test_ewma_shape():
    err = np.random.rand(300).astype(np.float32)
    smoothed = ewma_smooth(err, span=30)
    assert smoothed.shape == err.shape


def test_ewma_non_negative():
    err = np.abs(np.random.randn(200).astype(np.float32))
    smoothed = ewma_smooth(err, span=10)
    assert (smoothed >= 0).all()


def test_ewma_2d_raises():
    with pytest.raises(ValueError):
        ewma_smooth(np.ones((10, 5)))


# ============================================================
# Dynamic Threshold
# ============================================================

def test_dynamic_threshold_range():
    errors = np.abs(np.random.randn(500).astype(np.float32))
    threshold, z = select_dynamic_threshold(errors)
    mu = np.mean(errors)
    sigma = np.std(errors)
    # threshold must be at least mu + 2.5*sigma
    assert threshold >= mu + 2.4 * sigma


def test_dynamic_threshold_zero_std():
    errors = np.ones(100, dtype=np.float32)
    threshold, z = select_dynamic_threshold(errors)
    assert threshold > 0


# ============================================================
# Sequence Pruning
# ============================================================

def test_prune_keeps_high_error():
    errors = np.ones(500) * 0.1
    errors[200:250] = 10.0  # large spike — should be kept
    smoothed = errors.astype(np.float32)
    seqs = [(200, 249)]
    pruned = prune_sequences(smoothed, seqs, min_percent_change=0.13)
    assert len(pruned) == 1


def test_prune_removes_low_error():
    errors = np.ones(500) * 1.0
    errors[200:205] = 1.01  # tiny spike — should be pruned
    smoothed = errors.astype(np.float32)
    seqs = [(200, 204)]
    pruned = prune_sequences(smoothed, seqs, min_percent_change=0.13)
    assert len(pruned) == 0


def test_prune_empty_sequences():
    errors = np.ones(100)
    pruned = prune_sequences(errors, [], 0.13)
    assert pruned == []


# ============================================================
# Z-Score Baseline
# ============================================================

def test_zscore_fit_score():
    train = np.random.randn(1000, 2)
    test = np.random.randn(300, 2)
    test[100:110, 0] = 10.0  # inject anomaly
    bl = ZScoreBaseline(threshold=3.0)
    bl.fit(train)
    z = bl.score(test)
    assert z.shape == (300,)
    assert z[105] > 3.0


def test_zscore_detect_intervals():
    train = np.random.randn(1000)
    test = np.random.randn(300)
    test[100:110] = 15.0
    bl = ZScoreBaseline(threshold=3.0)
    bl.fit(train.reshape(-1, 1))
    intervals, z_scores = bl.detect(test.reshape(-1, 1))
    assert len(intervals) > 0
    # At least one interval overlaps [100,109]
    any_overlap = any(s <= 109 and e >= 100 for s, e in intervals)
    assert any_overlap

