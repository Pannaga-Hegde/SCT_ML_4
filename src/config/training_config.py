"""Training configuration module for GestureFlow.

Centralizes hyperparameters, optimizer settings, scheduler policies, dropout rates,
early stopping parameters, and device selections for neural network training.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

import torch


@dataclass
class TrainingConfig:
    """Central configuration for GestureCNN model training."""

    # Project Directory Paths
    project_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent.parent
    )
    checkpoint_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent.parent
        / "models"
        / "checkpoints"
    )
    output_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent.parent
        / "outputs"
        / "training"
    )

    # Hyperparameters
    num_classes: int = 10
    epochs: int = 25
    batch_size: int = 32
    learning_rate: float = 0.001  # 1e-3
    weight_decay: float = 0.01  # 1e-2 for AdamW
    dropout_rate: float = 0.3

    # Optimizer & Scheduler
    optimizer_name: str = "AdamW"
    scheduler_name: str = "CosineAnnealingLR"
    t_max: int = 25  # Matches total epochs for CosineAnnealingLR
    eta_min: float = 1e-6

    # Callbacks & Patience
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.0001
    best_model_name: str = "best_model.pth"

    # Execution Environment
    seed: int = 42
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )

    def __post_init__(self) -> None:
        """Ensure directories exist upon initialization."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def best_checkpoint_path(self) -> Path:
        """Return absolute path to best model checkpoint artifact."""
        return self.checkpoint_dir / self.best_model_name


# Default global instance
training_config = TrainingConfig()
