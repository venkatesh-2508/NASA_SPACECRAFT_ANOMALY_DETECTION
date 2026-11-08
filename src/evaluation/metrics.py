"""
NASA Spacecraft Anomaly Detection
Module: Evaluation Metrics

Event-level evaluation: a ground-truth anomaly is considered DETECTED when
at least one predicted interval overlaps it. This avoids penalising the
model for detecting the right event slightly late or early.
"""
from typing import Dict, List, Tuple

import numpy as np

from src.utils.helpers import overlap
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def event_level_metrics(
    predicted_sequences: List[Tuple[int, int]],
    ground_truth_sequences: List[Tuple[int, int]],
) -> Dict:
    """
    Compute precision, recall, F1 at event level.

    - TP: each GT interval matched by at least one predicted interval
    - FP: predicted intervals that match no GT interval
    - FN: GT intervals matched by no predicted interval

    Returns
    -------
    dict with TP, FP, FN, precision, recall, f1
    """
    if not ground_truth_sequences and not predicted_sequences:
        return {"TP": 0, "FP": 0, "FN": 0, "precision": 1.0, "recall": 1.0, "f1": 1.0}

    # Which GT events are matched?
    gt_matched = [False] * len(ground_truth_sequences)
    pred_matched = [False] * len(predicted_sequences)

    for gi, (gs, ge) in enumerate(ground_truth_sequences):
        for pi, (ps, pe) in enumerate(predicted_sequences):
            if overlap(ps, pe, gs, ge):
                gt_matched[gi] = True
                pred_matched[pi] = True

    TP = int(sum(gt_matched))
    FN = len(ground_truth_sequences) - TP
    FP = len(predicted_sequences) - int(sum(pred_matched))

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def aggregate_metrics(channel_results: List[Dict]) -> Dict:
    """
    Aggregate per-channel event-level metrics across all channels.

    Parameters
    ----------
    channel_results : list of dicts, each containing
        'pruned_sequences' and 'ground_truth'

    Returns
    -------
    aggregated metrics dict
    """
    total_TP = total_FP = total_FN = 0
    per_channel = []

    for res in channel_results:
        cid = res.get("channel_id", "?")
        pred = res.get("pruned_sequences", [])
        gt = res.get("ground_truth", [])
        # Convert gt from list-of-lists to list-of-tuples
        gt_tuples = [(int(s[0]), int(s[1])) for s in gt]
        pred_tuples = [(int(s[0]), int(s[1])) for s in pred]

        m = event_level_metrics(pred_tuples, gt_tuples)
        m["channel_id"] = cid
        m["mission"] = res.get("mission", "?")
        per_channel.append(m)

        total_TP += m["TP"]
        total_FP += m["FP"]
        total_FN += m["FN"]

    precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0.0
    recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "TP": total_TP,
        "FP": total_FP,
        "FN": total_FN,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "per_channel": per_channel,
    }

