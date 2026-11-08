"""
NASA Spacecraft Anomaly Detection
Module: LSTM Model (PyTorch)

Architecture:
  Input  : (batch, seq_len, n_features)
  LSTM   : 2 layers, 80 hidden units, dropout=0.3
  Output : (batch, 1) — predicted next telemetry value
"""
from pathlib import Path
from typing import Dict, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class LSTMAnomalyDetector(nn.Module if TORCH_AVAILABLE else object):
    """
    Two-layer LSTM that predicts the next telemetry measurement given
    a window of historical observations.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 80,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for LSTMAnomalyDetector.")
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)          # (batch, seq_len, hidden)
        last_hidden = lstm_out[:, -1, :]    # take output at last time step
        out = self.dropout(last_hidden)
        out = self.fc(out)                  # (batch, 1)
        return out.squeeze(-1)              # (batch,)

    def get_config(self) -> Dict:
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
        }


def build_model(input_size: int, cfg: dict) -> "LSTMAnomalyDetector":
    """Instantiate an LSTMAnomalyDetector from a config dict."""
    model_cfg = cfg.get("model", {})
    model = LSTMAnomalyDetector(
        input_size=input_size,
        hidden_size=model_cfg.get("hidden_size", 80),
        num_layers=model_cfg.get("num_layers", 2),
        dropout=model_cfg.get("dropout", 0.3),
    )
    return model


def save_checkpoint(model: "LSTMAnomalyDetector", optimizer, epoch: int, val_loss: float, filepath: str) -> None:
    """Save model weights + metadata to a checkpoint file."""
    if not TORCH_AVAILABLE:
        return
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "model_config": model.get_config(),
        },
        filepath,
    )
    logger.debug(f"Checkpoint saved: {filepath} (epoch={epoch}, val_loss={val_loss:.6f})")


def load_checkpoint(filepath: str, device: str = "cpu") -> Dict:
    """Load a checkpoint dict from file."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required.")
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    return torch.load(filepath, map_location=device, weights_only=False)


def load_model_from_checkpoint(filepath: str, device: str = "cpu") -> "LSTMAnomalyDetector":
    """Reconstruct and return a model from a saved checkpoint."""
    ckpt = load_checkpoint(filepath, device)
    cfg = ckpt["model_config"]
    model = LSTMAnomalyDetector(**cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model

