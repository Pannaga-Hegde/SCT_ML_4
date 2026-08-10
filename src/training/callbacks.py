"""Training callbacks module for GestureFlow.

Implements EarlyStopping convergence monitor and ModelCheckpoint saver callbacks.
"""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


class EarlyStopping:
    """Early stopping monitor halting training when validation loss fails to improve."""

    def __init__(
        self, patience: int = 5, min_delta: float = 0.0001, mode: str = "min"
    ) -> None:
        """Initialize EarlyStopping.

        Args:
            patience: Number of epochs to wait without improvement before stopping.
            min_delta: Minimum metric change considered as an improvement.
            mode: 'min' for loss metrics, 'max' for accuracy/F1 metrics.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode.lower()
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False

    def __call__(self, current_val: float) -> bool:
        """Evaluate current validation metric and check stop condition.

        Args:
            current_val: Current epoch validation metric value.

        Returns:
            True if early stopping trigger condition is met, False otherwise.
        """
        if self.best_score is None:
            self.best_score = current_val
            return False

        if self.mode == "min":
            improved = current_val < (self.best_score - self.min_delta)
        else:
            improved = current_val > (self.best_score + self.min_delta)

        if improved:
            self.best_score = current_val
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


class ModelCheckpoint:
    """Model checkpoint saver tracking and saving best PyTorch model weights artifact."""

    def __init__(
        self,
        checkpoint_path: Path,
        mode: str = "min",
    ) -> None:
        """Initialize ModelCheckpoint.

        Args:
            checkpoint_path: Path destination to save best model checkpoint (.pth).
            mode: 'min' for validation loss, 'max' for validation accuracy/F1.
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.mode = mode.lower()
        self.best_score: Optional[float] = None
        self.saved_epochs = 0

    def __call__(self, current_val: float, model: nn.Module, epoch: int) -> bool:
        """Check metric improvement and save model state dict if best score achieved.

        Args:
            current_val: Current epoch validation metric value.
            model: PyTorch nn.Module instance.
            epoch: Current epoch integer index.

        Returns:
            True if a new best checkpoint was saved, False otherwise.
        """
        if self.best_score is None:
            is_best = True
        elif self.mode == "min":
            is_best = current_val < self.best_score
        else:
            is_best = current_val > self.best_score

        if is_best:
            self.best_score = current_val
            self.saved_epochs = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_score": self.best_score,
                },
                self.checkpoint_path,
            )
            return True

        return False
