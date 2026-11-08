"""
NASA Spacecraft Anomaly Detection
Module: Dynamic Thresholding

Implements the nonparametric threshold-selection method from the Telemanom
research methodology (Hundman et al., KDD 2018).

Candidate thresholds: mu + z * sigma  for z in [2.5, 10.0] step 0.5

The threshold that maximises an anomaly-score objective (penalised by
sequence count) is selected WITHOUT using labelled test anomalies.
"""
from typing import List, Optional, Tuple

import numpy as np

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def _score_threshold(
    errors: np.ndarray,
    threshold: float,
    error_buffer: int = 100,
) -> float:
    """
    Score a candidate threshold value based on the fraction of the error
    distribution it captures, penalised by the number of anomaly sequences
    it would generate.  Higher scores are better.

    This is an *unsupervised* score — no labels are consulted.
    """
    is_above = (errors > threshold).astype(int)
    if is_above.sum() == 0:
        return 0.0

    # Count number of distinct sequences
    seqs = 0
    in_seq = False
    for flag in is_above:
        if flag and not in_seq:
            seqs += 1
            in_seq = True
        elif not flag:
            in_seq = False

    # Fraction of errors that exceed threshold (we want this high)
    pct_above = is_above.sum() / len(errors)

    # Penalise excessive fragmentation
    score = pct_above / (1 + 0.1 * seqs)
    return score


def select_dynamic_threshold(
    errors: np.ndarray,
    z_min: float = 2.5,
    z_max: float = 10.0,
    z_step: float = 0.5,
    error_buffer: int = 100,
) -> Tuple[float, float]:
    """
    Select the best anomaly threshold from candidate z multipliers.

    Parameters
    ----------
    errors     : EWMA-smoothed error array (1-D)
    z_min, z_max, z_step : range for z multiplier search
    error_buffer : used in sequence scoring

    Returns
    -------
    (threshold, z_selected) : best threshold value and the z that produced it
    """
    mu = np.mean(errors)
    sigma = np.std(errors)

    if sigma == 0:
        logger.warning("Error std=0; using z=3.0 threshold.")
        return float(mu + 3.0 * 1e-6), 3.0

    z_values = np.arange(z_min, z_max + z_step / 2, z_step)
    best_score = -1.0
    best_threshold = mu + z_min * sigma
    best_z = z_min

    for z in z_values:
        threshold = mu + z * sigma
        score = _score_threshold(errors, threshold, error_buffer)
        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_z = z

    logger.debug(
        f"Dynamic threshold: z={best_z:.1f}, threshold={best_threshold:.6f}, "
        f"mu={mu:.6f}, sigma={sigma:.6f}"
    )
    return float(best_threshold), float(best_z)

