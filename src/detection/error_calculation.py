"""
NASA Spacecraft Anomaly Detection
Module: Error Calculation

Calculates prediction error between actual and LSTM-predicted telemetry values.
"""
import numpy as np
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def compute_prediction_error(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """
    Compute absolute prediction error element-wise.

    Parameters
    ----------
    actual    : np.ndarray, shape (N,) — ground-truth telemetry values
    predicted : np.ndarray, shape (N,) — LSTM model predictions

    Returns
    -------
    error : np.ndarray, shape (N,) — |actual - predicted|
    """
    if actual.shape != predicted.shape:
        raise ValueError(
            f"Shape mismatch: actual={actual.shape}, predicted={predicted.shape}"
        )
    error = np.abs(actual.astype(np.float64) - predicted.astype(np.float64))
    logger.debug(
        f"Error computed: mean={np.mean(error):.6f}, max={np.max(error):.6f}, shape={error.shape}"
    )
    return error.astype(np.float32)

