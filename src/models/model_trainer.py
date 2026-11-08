"""
NASA Spacecraft Anomaly Detection
Module: Model Trainer

Trains the LSTM model on a single telemetry channel with:
  - PyTorch Dataset / DataLoader
  - MSE loss
  - Adam optimizer
  - Early stopping (monitors validation loss)
  - Best-checkpoint saving
  - Per-epoch loss tracking
"""
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.models.lstm_model import LSTMAnomalyDetector, save_checkpoint
from src.utils.logger import setup_logger
from src.utils.helpers import save_json

logger = setup_logger(__name__)


class TelemetryDataset(Dataset if TORCH_AVAILABLE else object):
    """PyTorch Dataset wrapping pre-generated sequence arrays."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required.")
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class ModelTrainer:
    """Train an LSTMAnomalyDetector for a single telemetry channel."""

    def __init__(self, cfg: dict):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for ModelTrainer.")
        self.cfg = cfg
        model_cfg = cfg.get("model", {})
        self.batch_size = model_cfg.get("batch_size", 64)
        self.lr = model_cfg.get("learning_rate", 0.001)
        self.max_epochs = model_cfg.get("max_epochs", 50)
        self.patience = model_cfg.get("patience", 5)
        self.min_delta = model_cfg.get("min_delta", 1e-4)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu" and TORCH_AVAILABLE:
            import os
            n_threads = min(8, os.cpu_count() or 4)
            torch.set_num_threads(n_threads)
            logger.info(f"Set PyTorch CPU threads to {n_threads}")
        logger.info(f"Trainer using device: {self.device}")

    def train(
        self,
        model: LSTMAnomalyDetector,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        checkpoint_path: str,
    ) -> Dict:
        """
        Train model and return a history dict with per-epoch losses.

        Parameters
        ----------
        model           : LSTMAnomalyDetector (untrained)
        X_train, y_train: training sequences
        X_val, y_val    : validation sequences
        checkpoint_path : where to save the best checkpoint

        Returns
        -------
        history : dict with 'train_loss', 'val_loss', 'best_epoch', 'best_val_loss'
        """
        model = model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        early_stopper = EarlyStopping(patience=self.patience, min_delta=self.min_delta)

        train_dataset = TelemetryDataset(X_train, y_train)
        val_dataset = TelemetryDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        history = {"train_loss": [], "val_loss": [], "best_epoch": 0, "best_val_loss": float("inf")}
        best_val_loss = float("inf")

        t0 = time.time()
        for epoch in range(1, self.max_epochs + 1):
            # --- Training ---
            model.train()
            train_losses = []
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_losses.append(loss.item())

            # --- Validation ---
            model.eval()
            val_losses = []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    pred = model(X_batch)
                    loss = criterion(pred, y_batch)
                    val_losses.append(loss.item())

            avg_train = np.mean(train_losses)
            avg_val = np.mean(val_losses)
            history["train_loss"].append(float(avg_train))
            history["val_loss"].append(float(avg_val))

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                history["best_epoch"] = epoch
                history["best_val_loss"] = float(best_val_loss)
                save_checkpoint(model, optimizer, epoch, best_val_loss, checkpoint_path)

            elapsed = time.time() - t0
            logger.info(
                f"Epoch {epoch:3d}/{self.max_epochs} | "
                f"Train={avg_train:.6f} | Val={avg_val:.6f} | "
                f"Best={best_val_loss:.6f} | Elapsed={elapsed:.1f}s"
            )

            if early_stopper(avg_val):
                logger.info(f"Early stopping triggered at epoch {epoch}.")
                break

        return history

    def predict(
        self,
        model: LSTMAnomalyDetector,
        X: np.ndarray,
        batch_size: int = 256,
    ) -> np.ndarray:
        """
        Generate predictions for a sequence array.

        Returns
        -------
        predictions : np.ndarray, shape (N,)
        """
        model.eval()
        model = model.to(self.device)
        all_preds = []
        dataset = TelemetryDataset(X, np.zeros(len(X)))  # dummy y
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        with torch.no_grad():
            for X_batch, _ in loader:
                X_batch = X_batch.to(self.device)
                pred = model(X_batch)
                all_preds.append(pred.cpu().numpy())
        return np.concatenate(all_preds, axis=0)

