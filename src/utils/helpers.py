"""
NASA Spacecraft Anomaly Detection
Utility: Helper functions
"""
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


def ensure_dir(path: str) -> Path:
    """Create directory and all parents if they don't exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: Any, filepath: str) -> None:
    """Save data to a JSON file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(filepath: str) -> Any:
    """Load data from a JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_pickle(obj: Any, filepath: str) -> None:
    """Pickle an object to file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(filepath: str) -> Any:
    """Load pickled object from file."""
    with open(filepath, "rb") as f:
        return pickle.load(f)


def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def format_duration(seconds: float) -> str:
    """Format seconds as human-readable duration."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def infer_mission(channel_id: str) -> str:
    """Infer spacecraft mission from channel ID prefix."""
    if channel_id.upper().startswith("P-") or channel_id.upper().startswith("S-") or channel_id.upper().startswith("D-") or channel_id.upper().startswith("A-") or channel_id.upper().startswith("G-") or channel_id.upper().startswith("E-") or channel_id.upper().startswith("F-"):
        return "SMAP"
    elif channel_id.upper().startswith("M-") or channel_id.upper().startswith("T-"):
        return "MSL"
    # Fallback: check first letter
    prefix = channel_id.split("-")[0].upper()
    smap_prefixes = {"P", "S", "D", "A", "G", "E", "F"}
    msl_prefixes = {"M", "T"}
    if prefix in smap_prefixes:
        return "SMAP"
    elif prefix in msl_prefixes:
        return "MSL"
    return "UNKNOWN"


def overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    """Check if two intervals [a_start, a_end] and [b_start, b_end] overlap."""
    return a_start <= b_end and b_start <= a_end


def compute_overlap_ratio(pred_start: int, pred_end: int, true_start: int, true_end: int) -> float:
    """Compute the overlap ratio between two intervals."""
    overlap_start = max(pred_start, true_start)
    overlap_end = min(pred_end, true_end)
    if overlap_start > overlap_end:
        return 0.0
    overlap_len = overlap_end - overlap_start + 1
    pred_len = pred_end - pred_start + 1
    return overlap_len / pred_len

