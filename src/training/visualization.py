"""Publication-quality training visualization generator module for GestureFlow.

Generates dark-slate themed training loss, accuracy, precision/recall, and learning rate curves
adhering strictly to docs/design.md visual aesthetics.
"""

from pathlib import Path
from typing import Dict, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from src.training.history import TrainingHistory


def set_dark_theme(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    """Apply dark slate palette visual design styling to matplotlib axes.

    Args:
        ax: Matplotlib axes object.
        title: Plot title string.
        xlabel: X-axis label text.
        ylabel: Y-axis label text.
    """
    ax.set_facecolor("#0f172a")
    ax.set_title(title, color="#f8fafc", fontsize=14, pad=12, fontweight="bold")
    ax.set_xlabel(xlabel, color="#94a3b8", fontsize=11, labelpad=8)
    ax.set_ylabel(ylabel, color="#94a3b8", fontsize=11, labelpad=8)
    ax.tick_params(colors="#94a3b8", labelsize=10)
    ax.grid(True, linestyle="--", alpha=0.3, color="#1e293b")
    for spine in ax.spines.values():
        spine.set_color("#334155")


def generate_training_plots(
    history: Union[TrainingHistory, Dict], output_dir: Path
) -> Dict[str, Path]:
    """Generate and save 4 publication-quality training curve plots.

    Args:
        history: TrainingHistory instance or dict representation.
        output_dir: Destination folder path for plot PNG artifacts.

    Returns:
        Dictionary mapping plot name to absolute output Path.
    """
    if isinstance(history, TrainingHistory):
        h = history.to_dict()
    else:
        h = history

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = h["epochs"]

    saved_plots = {}

    # 1. Loss Curve (outputs/training/loss_curve.png)
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#080b11")
    set_dark_theme(ax, "GestureCNN — Training & Validation Loss", "Epoch", "Cross-Entropy Loss")
    ax.plot(epochs, h["train_loss"], label="Training Loss", color="#00f2fe", linewidth=2.2, marker="o", markersize=4)
    ax.plot(epochs, h["val_loss"], label="Validation Loss", color="#ef4444", linewidth=2.2, marker="s", markersize=4)
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=10)
    fig.tight_layout()
    loss_path = output_dir / "loss_curve.png"
    fig.savefig(loss_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    saved_plots["loss_curve"] = loss_path

    # 2. Accuracy Curve (outputs/training/accuracy_curve.png)
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#080b11")
    set_dark_theme(ax, "GestureCNN — Training & Validation Accuracy", "Epoch", "Accuracy (%)")
    train_acc_pct = [acc * 100 if acc <= 1.0 else acc for acc in h["train_acc"]]
    val_acc_pct = [acc * 100 if acc <= 1.0 else acc for acc in h["val_acc"]]
    ax.plot(epochs, train_acc_pct, label="Training Accuracy (%)", color="#00f2fe", linewidth=2.2, marker="o", markersize=4)
    ax.plot(epochs, val_acc_pct, label="Validation Accuracy (%)", color="#10b981", linewidth=2.2, marker="^", markersize=4)
    ax.set_ylim(0, 105)
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=10)
    fig.tight_layout()
    acc_path = output_dir / "accuracy_curve.png"
    fig.savefig(acc_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    saved_plots["accuracy_curve"] = acc_path

    # 3. Precision & Recall Curve (outputs/training/precision_recall_curve.png)
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#080b11")
    set_dark_theme(ax, "GestureCNN — Macro Precision & Recall", "Epoch", "Score")
    val_prec = h.get("val_precision", h.get("val_prec", [0.0] * len(epochs)))
    val_rec = h.get("val_recall", h.get("val_rec", [0.0] * len(epochs)))
    val_f1 = h.get("val_f1", [0.0] * len(epochs))

    ax.plot(epochs, val_prec, label="Validation Macro Precision", color="#4facfe", linewidth=2.2, marker="d", markersize=4)
    ax.plot(epochs, val_rec, label="Validation Macro Recall", color="#f59e0b", linewidth=2.2, marker="v", markersize=4)
    ax.plot(epochs, val_f1, label="Validation Macro F1", color="#10b981", linewidth=2.2, linestyle="--", marker="o", markersize=4)
    ax.set_ylim(0.0, 1.05)
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=10)
    fig.tight_layout()
    pr_path = output_dir / "precision_recall_curve.png"
    fig.savefig(pr_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    saved_plots["precision_recall_curve"] = pr_path

    # 4. Learning Rate Schedule (outputs/training/learning_rate_schedule.png)
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#080b11")
    set_dark_theme(ax, "GestureCNN — CosineAnnealingLR Schedule", "Epoch", "Learning Rate")
    ax.plot(epochs, h["learning_rates"], label="Learning Rate (AdamW)", color="#00f2fe", linewidth=2.2, marker="o", markersize=4)
    ax.set_yscale("log")
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=10)
    fig.tight_layout()
    lr_path = output_dir / "learning_rate_schedule.png"
    fig.savefig(lr_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    saved_plots["learning_rate_schedule"] = lr_path

    return saved_plots
