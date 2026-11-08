"""
NASA Spacecraft Anomaly Detection
Module: Baseline Anomaly Detector (Z-score)

Implements a simple statistical baseline using Z-score thresholding
on the raw telemetry column of the test data. Training statistics
(mean, std) are computed from train data only to prevent leakage.
"""
from typing import Dict, List, Tuple

import numpy as np

from src.utils.logger import setup_logger
from src.utils.helpers import overlap

logger = setup_logger(__name__)


class ZScoreBaseline:
    """
    Anomaly detection using |Z-score| > threshold.

    Training statistics are fit on the telemetry column (col 0) of
    the TRAINING data only.
    """

    def __init__(self, threshold: float = 3.0):
        self.threshold = threshold
        self._mean: float = 0.0
        self._std: float = 1.0
        self._fitted: bool = False

    def fit(self, train_data: np.ndarray) -> "ZScoreBaseline":
        """Compute mean and std from training telemetry."""
        if train_data.ndim > 1:
            tele = train_data[:, 0]
        else:
            tele = train_data
        self._mean = float(np.mean(tele))
        self._std = float(np.std(tele))
        if self._std == 0:
            self._std = 1.0
        self._fitted = True
        logger.debug(f"Z-score baseline fitted: mean={self._mean:.4f}, std={self._std:.4f}")
        return self

    def score(self, test_data: np.ndarray) -> np.ndarray:
        """Compute absolute Z-scores for test telemetry."""
        if not self._fitted:
            raise RuntimeError("Must call fit() before score().")
        if test_data.ndim > 1:
            tele = test_data[:, 0]
        else:
            tele = test_data
        return np.abs((tele - self._mean) / self._std)

    def detect(self, test_data: np.ndarray) -> Tuple[List[Tuple[int, int]], np.ndarray]:
        """
        Detect anomalous intervals using Z-score threshold.

        Returns
        -------
        intervals : list of (start, end) index pairs
        z_scores  : np.ndarray of per-step Z-scores
        """
        z_scores = self.score(test_data)
        is_anomaly = z_scores > self.threshold

        intervals = []
        in_anomaly = False
        start = 0
        for i, flag in enumerate(is_anomaly):
            if flag and not in_anomaly:
                start = i
                in_anomaly = True
            elif not flag and in_anomaly:
                intervals.append((start, i - 1))
                in_anomaly = False
        if in_anomaly:
            intervals.append((start, len(is_anomaly) - 1))

        return intervals, z_scores

