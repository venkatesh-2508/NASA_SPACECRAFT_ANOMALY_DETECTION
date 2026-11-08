"""
NASA Spacecraft Anomaly Detection
Module: Preprocessing Pipeline

Handles feature scaling (MinMaxScaler fit on train only) with strict
data-leakage prevention. Scaler is serialized so it can be reused at
inference time without re-fitting.
"""
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from src.utils.logger import setup_logger
from src.utils.helpers import save_pickle, load_pickle

logger = setup_logger(__name__)


class TelemetryPreprocessor:
    """
    Fits a [0,1] MinMax scaler on the first column (telemetry value)
    of the training data, then transforms both train and test.

    Strategy: scale telemetry column (col 0) independently; pass command
    columns through without scaling (they are typically binary flags).
    """

    def __init__(self, scale_telemetry: bool = True):
        self.scale_telemetry = scale_telemetry
        self._fitted = False
        self._tele_min: Optional[float] = None
        self._tele_max: Optional[float] = None
        self._range: Optional[float] = None

    def fit(self, train_data: np.ndarray) -> "TelemetryPreprocessor":
        """Fit scaler parameters using training data only."""
        if train_data.ndim == 1:
            train_data = train_data.reshape(-1, 1)
        tele = train_data[:, 0]
        self._tele_min = float(np.min(tele))
        self._tele_max = float(np.max(tele))
        self._range = self._tele_max - self._tele_min
        if self._range == 0:
            self._range = 1.0  # prevent division by zero
            logger.warning("Telemetry has zero range; setting range=1 to avoid division by zero.")
        self._fitted = True
        logger.debug(f"Scaler fitted: min={self._tele_min:.4f}, max={self._tele_max:.4f}")
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Scale telemetry column using fitted parameters."""
        if not self._fitted:
            raise RuntimeError("Preprocessor must be fitted before transform.")
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        result = data.copy().astype(np.float32)
        if self.scale_telemetry:
            result[:, 0] = (result[:, 0] - self._tele_min) / self._range
        return result

    def inverse_transform_telemetry(self, scaled: np.ndarray) -> np.ndarray:
        """Inverse-transform the telemetry column back to original scale."""
        if not self._fitted:
            raise RuntimeError("Preprocessor must be fitted before inverse_transform.")
        return scaled * self._range + self._tele_min

    def fit_transform(self, train_data: np.ndarray) -> np.ndarray:
        """Fit and transform training data."""
        self.fit(train_data)
        return self.transform(train_data)

    def save(self, filepath: str) -> None:
        save_pickle(self, filepath)
        logger.debug(f"Preprocessor saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "TelemetryPreprocessor":
        obj = load_pickle(filepath)
        logger.debug(f"Preprocessor loaded from {filepath}")
        return obj

    @property
    def is_fitted(self) -> bool:
        return self._fitted

