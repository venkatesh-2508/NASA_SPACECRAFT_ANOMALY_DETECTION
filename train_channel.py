"""
NASA Spacecraft Anomaly Detection
Script: train_channel.py

Train an LSTM model for one or more telemetry channels.

Usage:
  python train_channel.py --channel P-1
  python train_channel.py --channel P-1 --channel S-1
  python train_channel.py --all
  python train_channel.py --mission SMAP
"""
import argparse
import sys
import time
from pathlib import Path

# --- Ensure project root is on sys.path ---
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.utils.config import load_config, get_path
from src.utils.logger import setup_logger
from src.utils.helpers import set_random_seeds, save_json
from src.data.data_loader import DataLoader
from src.data.data_validation import DataValidator
from src.preprocessing.preprocessing import TelemetryPreprocessor
from src.preprocessing.sequence_generator import generate_sequences, train_val_split

logger = setup_logger("train_channel", log_file="results/training.log")


def train_one_channel(channel_id: str, cfg: dict, loader: DataLoader) -> dict:
    """Full training pipeline for a single channel."""
    try:
        import torch
        from src.models.lstm_model import build_model
        from src.models.model_trainer import ModelTrainer
    except ImportError as e:
        logger.error(f"PyTorch import error: {e}")
        return {"channel_id": channel_id, "status": "ERROR", "error": str(e)}

    logger.info(f"{'='*60}")
    logger.info(f"Training channel: {channel_id}")

    seed = cfg.get("preprocessing", {}).get("random_seed", 42)
    set_random_seeds(seed)

    models_dir = get_path(cfg, "models_dir")
    metrics_dir = get_path(cfg, "metrics_dir")
    pred_dir = get_path(cfg, "predictions_dir")

    # --- Load raw data ---
    try:
        train_raw = loader.load_train(channel_id)
        logger.info(f"  Train shape: {train_raw.shape}")
    except FileNotFoundError as e:
        logger.error(str(e))
        return {"channel_id": channel_id, "status": "ERROR", "error": str(e)}

    n_features = train_raw.shape[1]

    # --- Preprocess ---
    preprocessor = TelemetryPreprocessor(scale_telemetry=True)
    train_scaled = preprocessor.fit_transform(train_raw)
    preprocessor.save(str(models_dir / f"{channel_id}_scaler.pkl"))

    # --- Sequences ---
    seq_len = cfg.get("preprocessing", {}).get("sequence_length", 250)
    val_split = cfg.get("preprocessing", {}).get("validation_split", 0.2)

    if len(train_scaled) <= seq_len:
        logger.warning(f"  Insufficient data ({len(train_scaled)} ≤ {seq_len}). Skipping.")
        return {"channel_id": channel_id, "status": "SKIPPED", "reason": "insufficient data"}

    X, y = generate_sequences(train_scaled, seq_len)
    X_train, y_train, X_val, y_val = train_val_split(X, y, val_split)
    logger.info(f"  Sequences: train={len(X_train)}, val={len(X_val)}")

    # --- Build model ---
    model = build_model(n_features, cfg)
    logger.info(f"  Model: {model}")

    # --- Train ---
    trainer = ModelTrainer(cfg)
    checkpoint_path = str(models_dir / f"{channel_id}_best.pt")
    t0 = time.time()
    history = trainer.train(model, X_train, y_train, X_val, y_val, checkpoint_path)
    elapsed = time.time() - t0

    # --- Save history ---
    history["channel_id"] = channel_id
    history["elapsed_seconds"] = round(elapsed, 1)
    history["n_features"] = n_features
    save_json(history, str(metrics_dir / f"{channel_id}_training_history.json"))

    logger.info(
        f"  Done: best_epoch={history['best_epoch']}, "
        f"best_val_loss={history['best_val_loss']:.6f}, time={elapsed:.1f}s"
    )
    return {"channel_id": channel_id, "status": "OK", "history": history}


def main():
    parser = argparse.ArgumentParser(description="Train LSTM anomaly detector")
    parser.add_argument("--channel", action="append", default=[], help="Channel ID(s) to train")
    parser.add_argument("--all", action="store_true", help="Train all available channels")
    parser.add_argument("--mission", choices=["SMAP", "MSL"], help="Train all channels from a mission")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = get_path(cfg, "data_dir")
    loader = DataLoader(str(data_dir))

    # Validate dataset
    validator = DataValidator(loader)
    report = validator.validate_all()
    if not report["passed"]:
        validator.print_report(report)
        logger.error("Dataset validation failed. Cannot train.")
        sys.exit(1)
    validator.print_report(report)

    # Determine channels to train
    if args.all:
        channels = loader.get_channel_ids()
    elif args.mission == "SMAP":
        channels = loader.get_smap_channels()
    elif args.mission == "MSL":
        channels = loader.get_msl_channels()
    else:
        channels = args.channel

    if not channels:
        print("No channels specified. Use --channel P-1, --all, or --mission SMAP/MSL")
        sys.exit(1)

    logger.info(f"Channels to train: {channels}")
    results = []
    for cid in channels:
        result = train_one_channel(cid, cfg, loader)
        results.append(result)

    ok = sum(1 for r in results if r.get("status") == "OK")
    logger.info(f"\nTraining complete: {ok}/{len(results)} channels successful.")


if __name__ == "__main__":
    main()

