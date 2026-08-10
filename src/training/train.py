"""Phase 3.3 Training Execution & Best Model Selection CLI Entrypoint — GestureFlow.

Executes complete CNN model training, validation evaluation, metric logging, checkpoint saving,
plot generation, best model metadata creation, and training report publishing.
"""

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from src.config.training_config import training_config
from src.dataset.loader import create_dataloaders
from src.models.cnn import GestureCNN
from src.training.experiment import ExperimentTracker
from src.training.report import generate_best_model_metadata, generate_training_report
from src.training.trainer import ModelTrainer
from src.training.visualization import generate_training_plots


def set_seed(seed: int = 42) -> None:
    """Enforce strict seed reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed: Fixed integer random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_training_pipeline() -> None:
    """Orchestrate Phase 3.3 CNN Training Execution & Best Model Selection."""
    cfg = training_config
    set_seed(cfg.seed)

    print("=" * 70)
    print("      GestureFlow — Phase 3.3 CNN Training & Model Selection")
    print("=" * 70)
    print(f"Device: {cfg.device.upper()}")
    print(f"Epochs: {cfg.epochs} | Batch Size: {cfg.batch_size} | LR: {cfg.learning_rate}")
    print(f"Optimizer: {cfg.optimizer_name} | Scheduler: {cfg.scheduler_name}")
    print(f"Seed: {cfg.seed} | Early Stopping Patience: {cfg.early_stopping_patience}")
    print("-" * 70)

    # 1. Initialize experiment tracking metadata
    experiment_tracker = ExperimentTracker(config=cfg)
    exp_path = experiment_tracker.save()
    print(f"[1/6] Experiment initialized at: {exp_path}")

    # 2. Build DataLoaders for Train (Subjects 00-06) and Val (Subjects 07-08)
    print("[2/6] Building subject-isolated DataLoaders...")
    manifest_path = Path("outputs/dataset/manifest.csv")
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Dataset manifest missing at {manifest_path}. Run Phase 2 first."
        )

    train_loader, val_loader, _ = create_dataloaders()

    print(f"   - Training Set:   {len(train_loader.dataset):,} images ({len(train_loader)} batches)")
    print(f"   - Validation Set: {len(val_loader.dataset):,} images ({len(val_loader)} batches)")

    # 3. Instantiate GestureCNN model architecture
    print("[3/6] Instantiating PyTorch GestureCNN architecture...")
    model = GestureCNN(num_classes=cfg.num_classes, dropout_rate=cfg.dropout_rate)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   - Trainable Parameters: {param_count:,}")

    # 4. Instantiate ModelTrainer and execute training loop
    print("[4/6] Starting Model Training Loop...")
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
    )

    training_start = time.time()
    history = trainer.fit()
    total_duration = time.time() - training_start
    early_stopped = trainer.early_stopping.early_stop

    # 5. Generate publication-quality visualization plots
    print("[5/6] Generating dark-slate training visualization plots...")
    output_dir = Path("outputs/training")
    plot_paths = generate_training_plots(history, output_dir=output_dir)
    for plot_name, p_path in plot_paths.items():
        print(f"   - Generated: {p_path}")

    # 6. Generate Best Model Metadata and Training Report
    print("[6/6] Publishing Best Model Metadata & Training Report...")
    h_dict = history.to_dict()
    best_f1_idx = int(np.argmax(h_dict["val_f1"]))
    best_epoch = h_dict["epochs"][best_f1_idx]
    best_val_acc = h_dict["val_acc"][best_f1_idx]
    best_val_f1 = h_dict["val_f1"][best_f1_idx]

    best_meta_path = generate_best_model_metadata(
        history=history,
        cfg=cfg,
        best_epoch=best_epoch,
        best_val_acc=best_val_acc,
        best_val_f1=best_val_f1,
    )
    print(f"   - Best Model Metadata: {best_meta_path}")

    report_path = generate_training_report(
        history=history,
        cfg=cfg,
        early_stopped=early_stopped,
    )
    print(f"   - Training Report:      {report_path}")

    # Update experiment.json with final metrics
    if exp_path.exists():
        with open(exp_path, "r", encoding="utf-8") as f:
            exp_data = json.load(f)

        exp_data["end_time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        exp_data["duration_seconds"] = round(total_duration, 2)
        exp_data["training_results"] = {
            "total_epochs": len(h_dict["epochs"]),
            "best_epoch": best_epoch,
            "best_val_loss": min(h_dict["val_loss"]),
            "best_val_accuracy": best_val_acc,
            "best_val_f1": best_val_f1,
            "early_stopped": early_stopped,
            "final_lr": h_dict["learning_rates"][-1],
        }

        with open(exp_path, "w", encoding="utf-8") as f:
            json.dump(exp_data, f, indent=4)

    print("\n" + "=" * 70)
    print("      ✓ PHASE 3.3 TRAINING EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"Best Model Checkpoint: models/checkpoints/best_model.pth")
    print(f"Best Validation Acc:   {best_val_acc * 100:.2f}%")
    print(f"Best Validation F1:    {best_val_f1:.4f}")
    print(f"Total Execution Time:  {total_duration:.2f} seconds")
    print("=" * 70)


if __name__ == "__main__":
    run_training_pipeline()
