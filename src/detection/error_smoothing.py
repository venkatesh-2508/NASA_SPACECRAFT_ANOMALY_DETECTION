"""
NASA Spacecraft Anomaly Detection
Module: Error Smoothing

Applies Exponentially Weighted Moving Average (EWMA) to prediction errors,
reducing noise and improving anomaly detection stability.
"""
import numpy as np
import pandas as pd
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def ewma_smooth(errors: np.ndarray, span: int = 30) -> np.ndarray:
    """
    Apply EWMA smoothing to a 1-D error array.

    Parameters
    ----------
    errors : np.ndarray, shape (N,)
    span   : EWMA span parameter (controls decay rate)

    Returns
    -------
    smoothed : np.ndarray, shape (N,)
    """
    if errors.ndim != 1:
        raise ValueError(f"Expected 1-D array, got shape {errors.shape}")
    series = pd.Series(errors.astype(np.float64))
    smoothed = series.ewm(span=span, adjust=True).mean().values
    logger.debug(f"EWMA smoothed: span={span}, mean={np.mean(smoothed):.6f}")
    return smoothed.astype(np.float32)

