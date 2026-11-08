"""
Tests: Evaluation Metrics
"""
import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import event_level_metrics, aggregate_metrics


def test_perfect_detection():
    pred = [(100, 150)]
    gt = [(100, 150)]
    m = event_level_metrics(pred, gt)
    assert m["TP"] == 1
    assert m["FP"] == 0
    assert m["FN"] == 0
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0


def test_missed_anomaly():
    pred = []
    gt = [(100, 150)]
    m = event_level_metrics(pred, gt)
    assert m["TP"] == 0
    assert m["FN"] == 1
    assert m["recall"] == 0.0


def test_false_alarm():
    pred = [(300, 350)]
    gt = [(100, 150)]
    m = event_level_metrics(pred, gt)
    assert m["TP"] == 0
    assert m["FP"] == 1
    assert m["precision"] == 0.0


def test_partial_overlap():
    pred = [(120, 200)]
    gt = [(100, 150)]
    m = event_level_metrics(pred, gt)
    # Overlap exists (120 <= 150 and 100 <= 200)
    assert m["TP"] == 1
    assert m["FN"] == 0


def test_empty_both():
    m = event_level_metrics([], [])
    assert m["TP"] == 0
    assert m["f1"] == 1.0


def test_aggregate():
    results = [
        {"channel_id": "P-1", "mission": "SMAP",
         "pruned_sequences": [(100, 150)], "ground_truth": [(100, 150)]},
        {"channel_id": "M-1", "mission": "MSL",
         "pruned_sequences": [], "ground_truth": [(200, 250)]},
    ]
    agg = aggregate_metrics(results)
    assert agg["TP"] == 1
    assert agg["FN"] == 1
    assert agg["precision"] > 0
    assert agg["recall"] == pytest.approx(0.5)
    assert len(agg["per_channel"]) == 2

