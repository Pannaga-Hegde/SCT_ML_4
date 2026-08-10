"""Unit tests for Phase 3.2 Experiment Framework, Complexity Analyzer, Summary Generator, and Pre-Training Verification."""

from pathlib import Path
import json
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.config.training_config import TrainingConfig
from src.models.cnn import GestureCNN
from src.models.complexity import ModelComplexityAnalyzer
from src.models.summary import ModelSummaryGenerator
from src.training.callbacks import EarlyStopping, ModelCheckpoint
from src.training.experiment import ExperimentTracker
from src.training.history import TrainingHistory


def test_experiment_tracker(tmp_path: Path) -> None:
    """Test ExperimentTracker initialization and JSON metadata export."""
    cfg = TrainingConfig(output_dir=tmp_path)
    tracker = ExperimentTracker(config=cfg, experiment_name="test_exp")
    
    assert tracker.metadata.experiment_name == "test_exp"
    assert tracker.metadata.parameter_count == 422506
    assert tracker.metadata.num_classes == 10
    assert tracker.metadata.optimizer == "AdamW"

    json_path = tracker.save()
    assert json_path.exists()
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["experiment_name"] == "test_exp"
    assert data["parameter_count"] == 422506


def test_model_complexity_analyzer(tmp_path: Path) -> None:
    """Test ModelComplexityAnalyzer parameter counting, FP32 size, and FLOPs computation."""
    model = GestureCNN(in_channels=1, num_classes=10)
    analyzer = ModelComplexityAnalyzer(model=model, input_shape=(1, 1, 128, 128))
    report = analyzer.analyze()

    assert report.total_params == 422506
    assert report.trainable_params == 422506
    assert report.non_trainable_params == 0
    assert report.model_size_bytes_fp32 == 1690024
    assert round(report.model_size_mb_fp32, 3) == 1.612
    assert report.total_macs > 0
    assert report.total_flops == 2 * report.total_macs

    json_file = tmp_path / "model_complexity.json"
    analyzer.save_json(json_file)
    assert json_file.exists()


def test_model_summary_generator(tmp_path: Path) -> None:
    """Test ModelSummaryGenerator tabular text output generation."""
    model = GestureCNN(in_channels=1, num_classes=10)
    generator = ModelSummaryGenerator(model=model, input_shape=(1, 1, 128, 128))
    summary_text = generator.generate_summary_text()

    assert "Model Summary: GestureCNN" in summary_text
    assert "422,506" in summary_text
    assert "1.612 MB" in summary_text

    txt_file = tmp_path / "model_summary.txt"
    generator.save_summary(txt_file)
    assert txt_file.exists()


def test_training_history_csv_export(tmp_path: Path) -> None:
    """Test TrainingHistory CSV serialization."""
    history = TrainingHistory()
    history.add_epoch(1, 0.5, 0.4, 0.85, 0.88, 0.84, 0.87, 0.83, 0.86, 0.83, 0.86, 0.001, 12.5)

    csv_file = tmp_path / "history.csv"
    history.save_csv(csv_file)
    assert csv_file.exists()

    with open(csv_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == 2
    assert "epoch,train_loss,val_loss,train_acc,val_acc,train_precision,val_precision,train_recall,val_recall,train_f1,val_f1,learning_rate,epoch_duration_seconds" in lines[0]


def test_pre_training_verification_suite(tmp_path: Path) -> None:
    """Execute Phase 3.2 Pre-training Verification checklist with zero optimizer steps."""
    # 1. Model instantiates correctly
    model = GestureCNN(in_channels=1, num_classes=10)
    assert isinstance(model, nn.Module)
    assert model.num_parameters == 422506

    # 2. Forward pass succeeds
    dummy_input = torch.randn(32, 1, 128, 128)
    logits = model(dummy_input)
    assert logits.shape == (32, 10)

    # 3. Loss function initializes
    criterion = nn.CrossEntropyLoss()
    target = torch.randint(0, 10, (32,))
    loss = criterion(logits, target)
    assert loss.item() > 0.0

    # 4. Optimizer initializes
    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    assert len(optimizer.param_groups) == 1

    # 5. Scheduler initializes
    scheduler = CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-6)
    assert scheduler.get_last_lr()[0] == 0.001

    # 6. Callback system initializes
    ckpt_path = tmp_path / "checkpoints" / "best_model.pth"
    stopper = EarlyStopping(patience=5, min_delta=0.0001)
    saver = ModelCheckpoint(checkpoint_path=ckpt_path)
    assert stopper.patience == 5
    assert saver.checkpoint_path == ckpt_path

    # 7. History system initializes
    history = TrainingHistory()
    assert len(history.epochs) == 0

    # 8. Experiment configuration loads
    cfg = TrainingConfig(output_dir=tmp_path / "output", checkpoint_dir=tmp_path / "checkpoints")
    tracker = ExperimentTracker(config=cfg)
    exp_file = tracker.save()
    assert exp_file.exists()

    # 9. Output directories exist
    assert cfg.output_dir.exists()
    assert cfg.checkpoint_dir.exists()

    # CONFIRM ZERO OPTIMIZER STEPS EXECUTED & ZERO CHECKPOINTS CREATED
    assert not ckpt_path.exists()
