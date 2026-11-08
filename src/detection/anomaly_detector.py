"""
NASA Spacecraft Anomaly Detection
Module: Anomaly Detector (Full Pipeline Orchestrator)

Runs the complete detection pipeline for a single telemetry channel:
  1. Load telemetry + preprocess
  2. Generate sequences
  3. LSTM predictions
  4. Prediction error
  5. EWMA smoothing
  6. Dynamic threshold selection
  7. Sequence generation with error buffer
  8. False-alarm pruning (13% error-drop rule)
  9. Result packaging with aligned test-set time indices
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.data.data_loader import DataLoader
from src.preprocessing.preprocessing import TelemetryPreprocessor
from src.preprocessing.sequence_generator import generate_sequences
from src.detection.error_calculation import compute_prediction_error
from src.detection.error_smoothing import ewma_smooth
from src.detection.dynamic_threshold import select_dynamic_threshold
from src.detection.sequence_pruning import prune_sequences
from src.utils.logger import setup_logger
from src.utils.helpers import infer_mission, save_json, ensure_dir

logger = setup_logger(__name__)


def _errors_to_sequences(is_above: np.ndarray, error_buffer: int = 50) -> List[Tuple[int, int]]:
    """Convert a boolean flag array to contiguous buffered (start, end) intervals."""
    if not np.any(is_above):
        return []

    indices = np.where(is_above)[0]
    n = len(is_above)
    buffered = set()
    for idx in indices:
        s = max(0, idx - error_buffer)
        e = min(n - 1, idx + error_buffer)
        buffered.update(range(s, e + 1))

    sorted_idx = sorted(buffered)
    seqs = []
    start = sorted_idx[0]
    prev = start
    for idx in sorted_idx[1:]:
        if idx == prev + 1:
            prev = idx
        else:
            seqs.append((start, prev))
            start = idx
            prev = idx
    seqs.append((start, prev))
    return seqs


class AnomalyDetector:
    """
    End-to-end anomaly detection for one telemetry channel.
    Requires a trained LSTM model checkpoint.
    """

    def __init__(self, cfg: dict, data_loader: DataLoader):
        self.cfg = cfg
        self.loader = data_loader
        self.seq_len = cfg.get("preprocessing", {}).get("sequence_length", 250)
        self.smooth_span = cfg.get("detection", {}).get("smoothing_span", 30)
        self.z_min = cfg.get("detection", {}).get("z_score_min", 2.5)
        self.z_max = cfg.get("detection", {}).get("z_score_max", 10.0)
        self.z_step = cfg.get("detection", {}).get("z_score_step", 0.5)
        self.error_buffer = cfg.get("detection", {}).get("error_buffer", 50)
        self.min_pct_change = cfg.get("detection", {}).get("min_percent_change", 0.13)

    def run_channel(
        self,
        channel_id: str,
        model,  # LSTMAnomalyDetector (already loaded)
        preprocessor: TelemetryPreprocessor,
        trainer,  # ModelTrainer instance for predict()
    ) -> Dict:
        """
        Run the full detection pipeline for one channel.
        Returns a result dict with all intermediate and final outputs.
        """
        logger.info(f"Running detection: {channel_id}")

        # --- Load data ---
        train_raw = self.loader.load_train(channel_id)
        test_raw = self.loader.load_test(channel_id)

        # --- Preprocess (transform only on test — scaler already fitted) ---
        train_scaled = preprocessor.transform(train_raw)
        test_scaled = preprocessor.transform(test_raw)

        # --- Generate sequences on test data ---
        X_test, y_test_seq = generate_sequences(test_scaled, self.seq_len)

        # Actual values (scaled for error calc)
        actual_scaled = test_scaled[self.seq_len:, 0]

        # --- LSTM predictions ---
        predictions_scaled = trainer.predict(model, X_test)

        # --- Prediction error ---
        raw_error = compute_prediction_error(actual_scaled, predictions_scaled)

        # --- EWMA smoothing ---
        smoothed_error = ewma_smooth(raw_error, span=self.smooth_span)

        # --- Dynamic threshold ---
        threshold, z_selected = select_dynamic_threshold(
            smoothed_error,
            z_min=self.z_min,
            z_max=self.z_max,
            z_step=self.z_step,
            error_buffer=self.error_buffer,
        )

        # --- Convert to sequences with error buffer ---
        is_above = smoothed_error > threshold
        raw_sequences_local = _errors_to_sequences(is_above, error_buffer=self.error_buffer)

        # --- Prune false alarms ---
        pruned_sequences_local = prune_sequences(
            smoothed_error,
            raw_sequences_local,
            min_percent_change=self.min_pct_change,
            threshold=threshold,
        )

        # Align sequences with original test sequence positions
        raw_sequences_aligned = [(s + self.seq_len, e + self.seq_len) for s, e in raw_sequences_local]
        pruned_sequences_aligned = [(s + self.seq_len, e + self.seq_len) for s, e in pruned_sequences_local]

        # --- Inverse transform predictions for display ---
        actual_original = preprocessor.inverse_transform_telemetry(actual_scaled)
        predictions_original = preprocessor.inverse_transform_telemetry(predictions_scaled)

        result = {
            "channel_id": channel_id,
            "mission": infer_mission(channel_id),
            "test_length": len(test_raw),
            "sequence_offset": self.seq_len,
            "actual": actual_original.tolist(),
            "predicted": predictions_original.tolist(),
            "raw_error": raw_error.tolist(),
            "smoothed_error": smoothed_error.tolist(),
            "threshold": float(threshold),
            "z_selected": float(z_selected),
            "raw_sequences": raw_sequences_aligned,
            "pruned_sequences": pruned_sequences_aligned,
            "raw_sequences_local": raw_sequences_local,
            "pruned_sequences_local": pruned_sequences_local,
            "num_raw_sequences": len(raw_sequences_aligned),
            "num_pruned_sequences": len(pruned_sequences_aligned),
            "ground_truth": self.loader.get_channel_labels(channel_id),
        }
        logger.info(
            f"  [{channel_id}] threshold={threshold:.4f} | "
            f"raw_seqs={len(raw_sequences_aligned)} | pruned_seqs={len(pruned_sequences_aligned)}"
        )
        return result
