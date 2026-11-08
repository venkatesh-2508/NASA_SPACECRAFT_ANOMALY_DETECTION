"""
NASA Spacecraft Anomaly Detection
Module: Sequence Generator

Generates sliding-window sequences for LSTM training/prediction.

Each sample:
  X : shape (sequence_length, n_features)   — input window
  y : scalar                                 — next telemetry value (col 0)
"""
from typing import Tuple

import numpy as np

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def generate_sequences(
    data: np.ndarray,
    sequence_length: int = 250,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Slide a window of `sequence_length` across `data` and produce (X, y) pairs.

    Parameters
    ----------
    data : np.ndarray, shape (T, F)
        Scaled telemetry + command features.
    sequence_length : int
        Number of time steps in each input window.

    Returns
    -------
    X : np.ndarray, shape (N, sequence_length, F)
    y : np.ndarray, shape (N,)   — next telemetry value
    """
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    T, F = data.shape
    if T <= sequence_length:
        raise ValueError(
            f"Data length ({T}) must be greater than sequence_length ({sequence_length})."
        )

    n_samples = T - sequence_length
    X = np.zeros((n_samples, sequence_length, F), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.float32)

    for i in range(n_samples):
        X[i] = data[i : i + sequence_length]
        y[i] = data[i + sequence_length, 0]  # predict telemetry column

    logger.debug(f"Generated {n_samples} sequences: X={X.shape}, y={y.shape}")
    return X, y


def train_val_split(
    X: np.ndarray,
    y: np.ndarray,
    val_fraction: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Chronological (non-shuffled) split of sequences into train and validation.

    Parameters
    ----------
    X, y       : sequence arrays
    val_fraction : fraction reserved for validation

    Returns
    -------
    X_train, y_train, X_val, y_val
    """
    n = len(X)
    split_idx = int(n * (1 - val_fraction))
    return X[:split_idx], y[:split_idx], X[split_idx:], y[split_idx:]

