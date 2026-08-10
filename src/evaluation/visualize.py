"""Visualization Module for GestureFlow Evaluation.

Renders publication-quality $10 \\times 10$ raw and normalized confusion matrix heatmaps
following the dark-slate visual aesthetics defined in docs/design.md.
"""

from pathlib import Path
from typing import List, Union, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def generate_confusion_matrix_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    output_dir: Union[str, Path] = "outputs/evaluation",
) -> Tuple[Path, Path]:
    """Generate raw count and normalized confusion matrix plots.

    Args:
        y_true: Ground truth target label indices.
        y_pred: Predicted class label indices.
        class_names: List of class label strings.
        output_dir: Output directory path.

    Returns:
        Tuple of (path_raw, path_normalized).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cm_raw = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = cm_raw.astype("float") / cm_raw.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)

    # 1. Raw Count Confusion Matrix Plot
    fig, ax = plt.subplots(figsize=(10, 8), facecolor="#080b11")
    ax.set_facecolor("#0f172a")

    sns.heatmap(
        cm_raw,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        ax=ax,
        linewidths=0.5,
        linecolor="#1e293b",
        annot_kws={"size": 10, "weight": "bold", "color": "#ffffff"},
    )

    ax.set_title("GestureCNN Raw Confusion Matrix (Subject 09 Test Set)", fontsize=14, color="#f8fafc", pad=15)
    ax.set_xlabel("Predicted Gesture Class", fontsize=12, color="#94a3b8", labelpad=10)
    ax.set_ylabel("True Gesture Class", fontsize=12, color="#94a3b8", labelpad=10)
    ax.tick_params(colors="#f8fafc", labelsize=10)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    path_raw = output_dir / "confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(path_raw, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)

    # 2. Normalized Confusion Matrix Plot
    fig, ax = plt.subplots(figsize=(10, 8), facecolor="#080b11")
    ax.set_facecolor("#0f172a")

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        ax=ax,
        linewidths=0.5,
        linecolor="#1e293b",
        annot_kws={"size": 10, "weight": "bold", "color": "#ffffff"},
        vmin=0.0,
        vmax=1.0,
    )

    ax.set_title("GestureCNN Normalized Confusion Matrix (Subject 09 Test Set)", fontsize=14, color="#f8fafc", pad=15)
    ax.set_xlabel("Predicted Gesture Class", fontsize=12, color="#94a3b8", labelpad=10)
    ax.set_ylabel("True Gesture Class", fontsize=12, color="#94a3b8", labelpad=10)
    ax.tick_params(colors="#f8fafc", labelsize=10)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    path_norm = output_dir / "confusion_matrix_normalized.png"
    plt.tight_layout()
    plt.savefig(path_norm, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)

    return path_raw, path_norm
