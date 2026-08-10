"""Master ModelTrainer orchestrator module for GestureFlow.

Executes training loops, validation evaluation, AdamW optimizer stepping,
CosineAnnealingLR learning rate scheduling, callback monitoring, and metrics logging.
"""

import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.config.training_config import TrainingConfig, training_config
from src.models.cnn import GestureCNN
from src.training.callbacks import EarlyStopping, ModelCheckpoint
from src.training.history import TrainingHistory
from src.training.metrics import TrainingMetricsCalculator


class ModelTrainer:
    """Trainer orchestrator managing model training, validation, callbacks, and history logging."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: TrainingConfig = training_config,
    ) -> None:
        """Initialize ModelTrainer.

        Args:
            model: PyTorch nn.Module instance (e.g. GestureCNN).
            train_loader: DataLoader for training set.
            val_loader: DataLoader for validation set.
            cfg: TrainingConfig instance.
        """
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model = model.to(self.device)

        self.train_loader = train_loader
        self.val_loader = val_loader

        # Loss, Optimizer & Scheduler
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=cfg.t_max,
            eta_min=cfg.eta_min,
        )

        # Callbacks & History Tracker
        self.early_stopping = EarlyStopping(
            patience=cfg.early_stopping_patience,
            min_delta=cfg.early_stopping_min_delta,
            mode="min",
        )
        self.checkpoint_saver = ModelCheckpoint(
            checkpoint_path=cfg.best_checkpoint_path,
            mode="min",
        )
        self.history = TrainingHistory()

    def train_epoch(self) -> Tuple[float, float, float, float, float]:
        """Execute one training epoch over training DataLoader.

        Returns:
            Tuple of (epoch_train_loss, accuracy, precision, recall, f1_macro).
        """
        self.model.train()
        running_loss = 0.0
        all_y_true = []
        all_y_pred = []

        for images, labels, _ in self.train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Check for NaN loss
            if torch.isnan(loss):
                raise ValueError("NaN loss encountered during training iteration.")

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)

            all_y_true.extend(labels.cpu().numpy())
            all_y_pred.extend(preds.cpu().numpy())

        epoch_loss = running_loss / len(self.train_loader.dataset)
        metrics = TrainingMetricsCalculator.calculate_metrics(
            np.array(all_y_true), np.array(all_y_pred)
        )

        return (
            epoch_loss,
            metrics["accuracy"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1_macro"],
        )

    def validate_epoch(self) -> Tuple[float, float, float, float, float]:
        """Execute one evaluation epoch over validation DataLoader.

        Returns:
            Tuple of (epoch_val_loss, accuracy, precision, recall, f1_macro).
        """
        self.model.eval()
        running_loss = 0.0
        all_y_true = []
        all_y_pred = []

        with torch.no_grad():
            for images, labels, _ in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                if torch.isnan(loss):
                    raise ValueError("NaN loss encountered during validation iteration.")

                running_loss += loss.item() * images.size(0)
                preds = torch.argmax(outputs, dim=1)

                all_y_true.extend(labels.cpu().numpy())
                all_y_pred.extend(preds.cpu().numpy())

        epoch_loss = running_loss / len(self.val_loader.dataset)
        metrics = TrainingMetricsCalculator.calculate_metrics(
            np.array(all_y_true), np.array(all_y_pred)
        )

        return (
            epoch_loss,
            metrics["accuracy"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1_macro"],
        )

    def fit(self) -> TrainingHistory:
        """Run full epoch training and validation loop with callbacks and LR scheduling.

        Returns:
            Completed TrainingHistory instance.
        """
        print("=" * 65)
        print(f"Starting GestureCNN Training Pipeline on {self.device.type.upper()}")
        print(f"Model Parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        print("=" * 65)

        start_time = time.time()
        checkpoint_dir = Path("models/checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Monitor Macro F1 score for best model selection (mode="max")
        self.checkpoint_saver = ModelCheckpoint(
            checkpoint_path=self.cfg.best_checkpoint_path,
            mode="max",
        )

        for epoch in range(1, self.cfg.epochs + 1):
            epoch_start = time.time()
            current_lr = self.optimizer.param_groups[0]["lr"]

            t_loss, t_acc, t_prec, t_rec, t_f1 = self.train_epoch()
            v_loss, v_acc, v_prec, v_rec, v_f1 = self.validate_epoch()

            self.scheduler.step()
            epoch_duration = time.time() - epoch_start

            # Record metrics in history
            self.history.add_epoch(
                epoch=epoch,
                t_loss=t_loss,
                v_loss=v_loss,
                t_acc=t_acc,
                v_acc=v_acc,
                t_prec=t_prec,
                v_prec=v_prec,
                t_rec=t_rec,
                v_rec=v_rec,
                t_f1=t_f1,
                v_f1=v_f1,
                lr=current_lr,
                duration=epoch_duration,
            )

            # Checkpoint callbacks: best model (mode="max" on v_f1)
            saved_best = self.checkpoint_saver(v_f1, self.model, epoch)
            best_tag = " [BEST F1 SAVED]" if saved_best else ""

            # Save latest_model.pth after every epoch
            latest_path = checkpoint_dir / "latest_model.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "val_loss": v_loss,
                    "val_acc": v_acc,
                    "val_f1": v_f1,
                },
                latest_path,
            )

            # Save periodic epoch checkpoints (epoch_05.pth, etc.)
            if epoch % 5 == 0:
                periodic_path = checkpoint_dir / f"epoch_{epoch:02d}.pth"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "val_loss": v_loss,
                        "val_acc": v_acc,
                        "val_f1": v_f1,
                    },
                    periodic_path,
                )

            print(
                f"Epoch {epoch:02d}/{self.cfg.epochs:02d} | "
                f"Train Loss: {t_loss:.4f} Acc: {t_acc*100:.2f}% | "
                f"Val Loss: {v_loss:.4f} Acc: {v_acc*100:.2f}% F1: {v_f1:.4f} | "
                f"LR: {current_lr:.6f} | {epoch_duration:.1f}s{best_tag}"
            )

            # Early Stopping on validation loss
            if self.early_stopping(v_loss):
                print(f"\n[Early Stopping Triggered at Epoch {epoch}] Validation loss stopped improving.")
                break

        elapsed = time.time() - start_time
        print("\n" + "=" * 65)
        print(f"Training completed in {elapsed:.2f}s.")
        print(f"Best Validation Loss: {min(self.history.val_loss):.4f}")
        print(f"Best Validation Accuracy: {max(self.history.val_acc)*100:.2f}%")
        print(f"Best Validation Macro F1: {max(self.history.val_f1):.4f}")
        print(f"Best Model Checkpoint saved to: {self.cfg.best_checkpoint_path}")
        print("=" * 65)

        # Export training history artifacts (JSON and CSV)
        history_json_path = self.cfg.output_dir / "training_history.json"
        history_csv_path = self.cfg.output_dir / "training_history.csv"
        self.history.save_json(history_json_path)
        self.history.save_csv(history_csv_path)

        return self.history
