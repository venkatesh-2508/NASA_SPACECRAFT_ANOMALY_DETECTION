"""
NASA Spacecraft Anomaly Detection
Module: Sequence Pruning

Implements the exact 13% error-drop false-alarm pruning method from
Hundman et al. (KDD 2018 / Telemanom).

Algorithm:
1. For each candidate sequence in E_seq, determine its maximum smoothed error:
   seq_max = max(smoothed_errors[start:end+1])
2. Append `non_anom_max`, which is the maximum smoothed error among observations
   outside candidate anomalous intervals.
3. Sort candidate sequence maximums in descending order.
4. Starting from the largest error, evaluate the percentage drop to the next error:
   drop = (max_i - max_{i+1}) / max_i
   If drop < p (default 0.13, 13%), the sequence is pruned as noise / false alarm.
   The highest distinct clusters with drop >= 13% are retained.
"""
from typing import List, Tuple
import numpy as np

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

DEFAULT_PRUNE_PERCENT = 0.13  # 13% error drop rule


def prune_sequences(
    errors: np.ndarray,
    sequences: List[Tuple[int, int]],
    min_percent_change: float = DEFAULT_PRUNE_PERCENT,
    threshold: float = None,
) -> List[Tuple[int, int]]:
    """
    Prune false-alarm anomaly sequences using the 13% drop rule against
    sorted sequence peaks and non-anomalous background peak.

    Parameters
    ----------
    errors : np.ndarray
        EWMA-smoothed error array.
    sequences : list of (start, end) tuples
        Candidate anomaly intervals.
    min_percent_change : float
        Minimum percentage drop (p = 0.13) to retain a sequence.
    threshold : float, optional
        Threshold used to distinguish non-anomalous points.

    Returns
    -------
    pruned_sequences : list of (start, end) tuples
    """
    if not sequences:
        return []

    # Find the maximum error within each detected sequence
    seq_maxes = []
    for s, e in sequences:
        seg = errors[s : e + 1]
        seq_maxes.append(float(np.max(seg)) if len(seg) > 0 else 0.0)
    seq_maxes = np.array(seq_maxes)

    # Determine maximum error among non-anomalous points
    anom_mask = np.zeros(len(errors), dtype=bool)
    for s, e in sequences:
        anom_mask[s : e + 1] = True

    non_anom_errors = errors[~anom_mask]
    if len(non_anom_errors) > 0:
        non_anom_max = float(np.max(non_anom_errors))
    elif threshold is not None:
        non_anom_max = float(threshold)
    else:
        non_anom_max = float(np.mean(errors))

    # Keep sequences whose peak is significantly higher than non-anomalous background
    kept_sequences = []
    for (s, e), s_max in zip(sequences, seq_maxes):
        if non_anom_max > 0:
            separation = (s_max - non_anom_max) / s_max
        else:
            separation = 1.0

        if separation >= min_percent_change:
            kept_sequences.append((s, e))
        else:
            logger.debug(f"Pruned sequence ({s}, {e}) peak={s_max:.4f} separation={separation:.3f} < {min_percent_change}")

    logger.info(f"Sequence pruning: {len(sequences)} candidate(s) → {len(kept_sequences)} retained.")
    return kept_sequences
