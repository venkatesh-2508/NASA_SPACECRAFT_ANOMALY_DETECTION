"""
NASA Spacecraft Anomaly Detection
Script: run_baseline.py

Runs the statistical baseline (Z-score thresholding) across telemetry channels
and saves metrics for comparison against the LSTM model.
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.utils.config import load_config, get_path
from src.utils.logger import setup_logger
from src.utils.helpers import save_json, infer_mission
from src.data.data_loader import DataLoader
from src.detection.baseline import ZScoreBaseline
from src.evaluation.metrics import event_level_metrics, aggregate_metrics
from src.evaluation.event_evaluation import evaluate_all

logger = setup_logger("run_baseline", log_file="results/baseline.log")


def evaluate_channel_baseline(channel_id: str, loader: DataLoader, threshold: float = 3.0) -> dict:
    """Run Z-score baseline for a single channel."""
    try:
        train_data = loader.load_train(channel_id)
        test_data = loader.load_test(channel_id)
    except Exception as e:
        logger.error(f"Failed to load data for {channel_id}: {e}")
        return {"channel_id": channel_id, "status": "ERROR", "error": str(e)}

    # Fit baseline on training telemetry only (column 0)
    baseline = ZScoreBaseline(threshold=threshold)
    baseline.fit(train_data)

    # Detect anomalies on test telemetry
    detected_intervals, z_scores = baseline.detect(test_data)
    ground_truth = loader.get_channel_labels(channel_id)

    # Calculate metrics
    metrics = event_level_metrics(detected_intervals, ground_truth)
    metrics["channel_id"] = channel_id
    metrics["mission"] = infer_mission(channel_id)
    metrics["num_detected"] = len(detected_intervals)
    metrics["num_ground_truth"] = len(ground_truth)
    metrics["threshold"] = threshold
    metrics["pruned_sequences"] = detected_intervals
    metrics["ground_truth"] = ground_truth

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run Z-score baseline anomaly detection")
    parser.add_argument("--channel", action="append", default=[], help="Channel ID(s)")
    parser.add_argument("--all", action="store_true", help="Run on all available channels")
    parser.add_argument("--threshold", type=float, default=3.0, help="Z-score threshold (default: 3.0)")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = get_path(cfg, "data_dir")
    metrics_dir = get_path(cfg, "metrics_dir")
    loader = DataLoader(str(data_dir))

    if args.all:
        channels = loader.get_channel_ids()
    elif args.channel:
        channels = args.channel
    else:
        channels = loader.get_channel_ids()

    if not channels:
        logger.error("No channels available. Run download_dataset.py first.")
        sys.exit(1)

    logger.info(f"Running Z-score baseline (threshold={args.threshold}) on {len(channels)} channels...")
    results = []
    for cid in channels:
        res = evaluate_channel_baseline(cid, loader, threshold=args.threshold)
        if "f1" in res:
            results.append(res)
            logger.info(f"  [{cid}] P={res['precision']:.3f} R={res['recall']:.3f} F1={res['f1']:.3f} (TP={res['TP']}, FP={res['FP']}, FN={res['FN']})")

    if results:
        # Save baseline report
        report = evaluate_all(results, loader, save_dir=None)
        save_json(report, str(metrics_dir / "baseline_report.json"))

        overall = report["overall"]
        logger.info(f"\nBaseline Overall: Precision={overall['precision']:.4f}, Recall={overall['recall']:.4f}, F1={overall['f1']:.4f}")
        logger.info(f"Total TP: {overall['TP']}, FP: {overall['FP']}, FN: {overall['FN']}")


if __name__ == "__main__":
    main()

