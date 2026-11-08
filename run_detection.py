"""
NASA Spacecraft Anomaly Detection
Script: run_detection.py

Run the full LSTM-based detection pipeline on trained channels
and save results to results/predictions/.

Usage:
  python run_detection.py --channel P-1
  python run_detection.py --all
  python run_detection.py --mission SMAP
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.utils.config import load_config, get_path
from src.utils.logger import setup_logger
from src.utils.helpers import save_json, infer_mission
from src.data.data_loader import DataLoader
from src.preprocessing.preprocessing import TelemetryPreprocessor
from src.detection.anomaly_detector import AnomalyDetector
from src.evaluation.event_evaluation import evaluate_all, build_anomaly_table

logger = setup_logger("run_detection", log_file="results/detection.log")


def detect_channel(channel_id: str, cfg: dict, loader: DataLoader, detector: AnomalyDetector) -> dict:
    """Run detection pipeline for a single trained channel."""
    try:
        import torch
        from src.models.lstm_model import load_model_from_checkpoint
        from src.models.model_trainer import ModelTrainer
    except ImportError as e:
        return {"channel_id": channel_id, "status": "ERROR", "error": str(e)}

    models_dir = get_path(cfg, "models_dir")
    checkpoint = models_dir / f"{channel_id}_best.pt"
    scaler_path = models_dir / f"{channel_id}_scaler.pkl"

    if not checkpoint.exists():
        logger.warning(f"No checkpoint for {channel_id}. Run train_channel.py first.")
        return {"channel_id": channel_id, "status": "NO_CHECKPOINT"}

    if not scaler_path.exists():
        logger.warning(f"No scaler for {channel_id}.")
        return {"channel_id": channel_id, "status": "NO_SCALER"}

    try:
        model = load_model_from_checkpoint(str(checkpoint))
        preprocessor = TelemetryPreprocessor.load(str(scaler_path))
        trainer = ModelTrainer(cfg)
        result = detector.run_channel(channel_id, model, preprocessor, trainer)
        result["status"] = "OK"
        return result
    except Exception as e:
        logger.error(f"Detection failed for {channel_id}: {e}")
        return {"channel_id": channel_id, "status": "ERROR", "error": str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel", action="append", default=[])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--mission", choices=["SMAP", "MSL"])
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = get_path(cfg, "data_dir")
    pred_dir = get_path(cfg, "predictions_dir")
    metrics_dir = get_path(cfg, "metrics_dir")
    loader = DataLoader(str(data_dir))
    detector = AnomalyDetector(cfg, loader)

    if args.all:
        channels = loader.get_channel_ids()
    elif args.mission == "SMAP":
        channels = loader.get_smap_channels()
    elif args.mission == "MSL":
        channels = loader.get_msl_channels()
    else:
        channels = args.channel

    if not channels:
        print("No channels specified.")
        sys.exit(1)

    all_results = []
    for cid in channels:
        res = detect_channel(cid, cfg, loader, detector)
        if res.get("status") == "OK":
            # Save full arrays to compressed NPZ
            np.savez_compressed(
                str(pred_dir / f"{cid}_arrays.npz"),
                actual=np.array(res["actual"], dtype=np.float32),
                predicted=np.array(res["predicted"], dtype=np.float32),
                raw_error=np.array(res["raw_error"], dtype=np.float32),
                smoothed_error=np.array(res["smoothed_error"], dtype=np.float32),
            )
            # Save per-channel result metadata (without large arrays for JSON)
            compact = {k: v for k, v in res.items() if k not in ("actual", "predicted", "raw_error", "smoothed_error")}
            save_json(compact, str(pred_dir / f"{cid}_detection.json"))
            all_results.append(res)

    # --- Aggregate evaluation ---
    if all_results:
        eval_report = evaluate_all(all_results, loader, str(metrics_dir))
        logger.info(f"\nOverall: P={eval_report['overall']['precision']:.3f} R={eval_report['overall']['recall']:.3f} F1={eval_report['overall']['f1']:.3f}")

        # Save anomaly table
        table = build_anomaly_table(all_results)
        table.to_csv(str(metrics_dir / "anomaly_table.csv"), index=False)
        logger.info(f"Anomaly table saved with {len(table)} rows")

    logger.info("Detection pipeline complete.")


if __name__ == "__main__":
    main()

