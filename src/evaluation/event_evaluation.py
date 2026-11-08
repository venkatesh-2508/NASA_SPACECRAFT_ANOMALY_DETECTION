"""
NASA Spacecraft Anomaly Detection
Module: Event Evaluation

Higher-level evaluation: compares per-channel detection results against
the labeled_anomalies.csv, broken out by mission and anomaly class.
"""
from pathlib import Path
from typing import Dict, List

import pandas as pd
import numpy as np

from src.evaluation.metrics import event_level_metrics, aggregate_metrics
from src.data.data_loader import DataLoader
from src.utils.logger import setup_logger
from src.utils.helpers import save_json

logger = setup_logger(__name__)


def evaluate_all(
    channel_results: List[Dict],
    loader: DataLoader,
    save_dir: str = None,
) -> Dict:
    """
    Full evaluation report broken down by mission and anomaly class.

    Returns
    -------
    report : dict
    """
    try:
        labels_df = loader.load_labels()
    except Exception as e:
        logger.error(f"Could not load labels: {e}")
        labels_df = pd.DataFrame()

    report = {
        "overall": aggregate_metrics(channel_results),
        "SMAP": None,
        "MSL": None,
        "point": None,
        "contextual": None,
    }

    # --- By mission ---
    for mission in ("SMAP", "MSL"):
        mission_results = [r for r in channel_results if r.get("mission") == mission]
        report[mission] = aggregate_metrics(mission_results) if mission_results else {}

    # --- By anomaly class ---
    if not labels_df.empty and "anomaly_class" in labels_df.columns:
        for cls in labels_df["anomaly_class"].dropna().unique():
            channels_in_class = set(
                labels_df[labels_df["anomaly_class"] == cls]["channel"].str.upper()
            )
            class_results = [
                r for r in channel_results
                if r.get("channel_id", "").upper() in channels_in_class
            ]
            report[cls.lower()] = aggregate_metrics(class_results) if class_results else {}

    # --- Save ---
    if save_dir:
        save_json(report, f"{save_dir}/evaluation_report.json")
        logger.info(f"Evaluation report saved to {save_dir}/evaluation_report.json")

    return report


def build_anomaly_table(channel_results: List[Dict]) -> pd.DataFrame:
    """
    Build a flat DataFrame of all detected anomaly sequences.
    One row per predicted anomaly interval.
    """
    rows = []
    for res in channel_results:
        cid = res.get("channel_id", "?")
        mission = res.get("mission", "?")
        threshold = res.get("threshold", np.nan)
        smoothed = np.array(res.get("smoothed_error", []))
        offset = res.get("sequence_offset", 0)

        for start, end in res.get("pruned_sequences", []):
            # pruned_sequences are already test-aligned
            local_start = max(0, int(start) - offset)
            local_end = min(len(smoothed) - 1, int(end) - offset)
            seg = smoothed[local_start:local_end+1] if len(smoothed) > 0 and local_start <= local_end else np.array([])
            rows.append({
                "channel": cid,
                "mission": mission,
                "start": int(start),
                "end": int(end),
                "duration": int(end - start + 1),
                "max_error": float(np.max(seg)) if len(seg) > 0 else np.nan,
                "mean_error": float(np.mean(seg)) if len(seg) > 0 else np.nan,
                "threshold": float(threshold),
                "status": "DETECTED",
            })

    if not rows:
        return pd.DataFrame(columns=["channel", "mission", "start", "end", "duration",
                                      "max_error", "mean_error", "threshold", "status"])
    return pd.DataFrame(rows)

