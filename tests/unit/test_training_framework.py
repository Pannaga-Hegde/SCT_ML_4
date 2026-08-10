"""Unit tests for training framework, configuration, metrics, callbacks, and history."""

from pathlib import Path
import numpy as np
import torch
import pytest

from src.config.training_config import TrainingConfig
from src.models.cnn import GestureCNN
from src.training.callbacks import EarlyStopping, ModelCheckpoint
from src.training.history import TrainingHistory
from src.training.metrics import TrainingMetricsCalculator
from src.training.trainer import ModelTrainer


def test_training_config_defaults(tmp_path: Path) -> None:
    """Test TrainingConfig default values and directory creation."""
    cfg = TrainingConfig(
        checkpoint_dir=tmp_path / "checkpoints",
        output_dir=tmp_path / "output",
    )
    assert cfg.epochs == 25
    assert cfg.batch_size == 32
    assert cfg.learning_rate == 0.001
    assert cfg.checkpoint_dir.exists()
    assert cfg.output_dir.exists()


def test_training_history_logging(tmp_path: Path) -> None:
    """Test TrainingHistory metrics recording and JSON serialization."""
    history = TrainingHistory()
    history.add_epoch(1, 0.5, 0.4, 0.85, 0.88, 0.84, 0.87, 0.83, 0.86, 0.83, 0.86, 0.001, 12.5)

    assert len(history.epochs) == 1
    assert history.train_loss[0] == 0.5
    assert history.val_loss[0] == 0.4

    json_file = tmp_path / "history.json"
    history.save_json(json_file)
    assert json_file.exists()


def test_metrics_calculator() -> None:
    """Test accuracy, precision, recall, and macro F1 metrics calculation."""
    y_true = np.array([0, 1, 2, 3, 4])
    y_pred = np.array([0, 1, 2, 3, 0])

    metrics = TrainingMetricsCalculator.calculate_metrics(y_true, y_pred)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["accuracy"] == 0.8
    assert "f1_macro" in metrics


def test_early_stopping_callback() -> None:
    """Test EarlyStopping trigger logic."""
    stopper = EarlyStopping(patience=3, min_delta=0.01, mode="min")

    assert not stopper(1.0)  # Initial score
    assert not stopper(0.95) # Improved
    assert not stopper(0.96) # Worse (patience 1)
    assert not stopper(0.96) # Worse (patience 2)
    assert stopper(0.96)     # Worse (patience 3 -> stop triggered)


def test_model_checkpoint_callback(tmp_path: Path) -> None:
    """Test ModelCheckpoint saving state dict artifact."""
    ckpt_path = tmp_path / "best_model.pth"
    saver = ModelCheckpoint(checkpoint_path=ckpt_path, mode="min")

    model = GestureCNN()
    saved = saver(0.5, model, epoch=1)

    assert saved
    assert ckpt_path.exists()
