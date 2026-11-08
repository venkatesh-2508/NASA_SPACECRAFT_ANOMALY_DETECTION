"""
NASA Spacecraft Anomaly Detection
Module: Visualization (EDA & Evaluation Plots)

Saves PNG figures to results/plots/ for documentation and reporting.
"""
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless mode
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

STYLE = {
    "bg": "#080D18",
    "card": "#141F30",
    "accent": "#38BDF8",
    "accent2": "#8B5CF6",
    "success": "#22C55E",
    "warning": "#F59E0B",
    "critical": "#EF4444",
    "text": "#F8FAFC",
    "text2": "#94A3B8",
}


def _apply_dark_style(ax, fig):
    fig.patch.set_facecolor(STYLE["bg"])
    ax.set_facecolor(STYLE["card"])
    ax.tick_params(colors=STYLE["text2"])
    ax.xaxis.label.set_color(STYLE["text2"])
    ax.yaxis.label.set_color(STYLE["text2"])
    ax.title.set_color(STYLE["text"])
    for spine in ax.spines.values():
        spine.set_edgecolor(STYLE["text2"])
        spine.set_alpha(0.3)
    ax.grid(True, color=STYLE["text2"], alpha=0.1, linewidth=0.5)


def plot_training_loss(train_loss: List[float], val_loss: List[float], best_epoch: int,
                        channel_id: str, save_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(10, 5))
    _apply_dark_style(ax, fig)
    epochs = range(1, len(train_loss) + 1)
    ax.plot(epochs, train_loss, color=STYLE["accent"], linewidth=2, label="Train Loss")
    ax.plot(epochs, val_loss, color=STYLE["accent2"], linewidth=2, label="Val Loss")
    ax.axvline(best_epoch, color=STYLE["success"], linestyle="--", alpha=0.8, label=f"Best (ep {best_epoch})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(f"Training & Validation Loss — {channel_id}")
    ax.legend(facecolor=STYLE["card"], labelcolor=STYLE["text"])
    path = str(Path(save_dir) / f"{channel_id}_training_loss.png")
    fig.savefig(path, dpi=100, bbox_inches="tight", facecolor=STYLE["bg"])
    plt.close(fig)
    logger.info(f"Saved training loss plot: {path}")
    return path


def plot_telemetry_with_anomalies(
    actual: np.ndarray,
    predicted: np.ndarray,
    smoothed_error: np.ndarray,
    threshold: float,
    gt_intervals: List[Tuple[int, int]],
    pred_intervals: List[Tuple[int, int]],
    channel_id: str,
    save_dir: str,
    offset: int = 0,
) -> str:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    t = np.arange(offset, offset + len(actual))

    _apply_dark_style(ax1, fig)
    _apply_dark_style(ax2, fig)

    # GT shading
    for gs, ge in gt_intervals:
        ax1.axvspan(gs + offset, ge + offset, color=STYLE["critical"], alpha=0.15)
    # Predicted shading
    for ps, pe in pred_intervals:
        ax1.axvspan(ps + offset, pe + offset, color=STYLE["warning"], alpha=0.2)

    ax1.plot(t, actual, color=STYLE["accent"], linewidth=0.8, label="Actual")
    ax1.plot(t, predicted, color=STYLE["accent2"], linewidth=0.8, linestyle="--", label="Predicted")
    ax1.set_ylabel("Telemetry Value")
    ax1.set_title(f"Channel {channel_id} — Telemetry vs Predictions")
    gt_patch = mpatches.Patch(color=STYLE["critical"], alpha=0.3, label="Ground Truth")
    pred_patch = mpatches.Patch(color=STYLE["warning"], alpha=0.4, label="Detected")
    ax1.legend(handles=[*ax1.get_legend_handles_labels()[0], gt_patch, pred_patch],
                facecolor=STYLE["card"], labelcolor=STYLE["text"])

    ax2.plot(t, smoothed_error, color=STYLE["warning"], linewidth=0.8, label="Smoothed Error")
    ax2.axhline(threshold, color=STYLE["critical"], linestyle="--", linewidth=1.2, label=f"Threshold={threshold:.4f}")
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Error")
    ax2.set_title("Prediction Error & Dynamic Threshold")
    ax2.legend(facecolor=STYLE["card"], labelcolor=STYLE["text"])

    fig.tight_layout()
    path = str(Path(save_dir) / f"{channel_id}_detection.png")
    fig.savefig(path, dpi=100, bbox_inches="tight", facecolor=STYLE["bg"])
    plt.close(fig)
    logger.info(f"Saved detection plot: {path}")
    return path

