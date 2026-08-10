"""Training history tracker and JSON logger module for GestureFlow."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class TrainingHistory:
    """Dataclass holding historical training and validation performance metrics."""

    epochs: List[int] = field(default_factory=list)
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    train_acc: List[float] = field(default_factory=list)
    val_acc: List[float] = field(default_factory=list)
    train_precision: List[float] = field(default_factory=list)
    val_precision: List[float] = field(default_factory=list)
    train_recall: List[float] = field(default_factory=list)
    val_recall: List[float] = field(default_factory=list)
    train_f1: List[float] = field(default_factory=list)
    val_f1: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    epoch_durations: List[float] = field(default_factory=list)

    def add_epoch(
        self,
        epoch: int,
        t_loss: float,
        v_loss: float,
        t_acc: float,
        v_acc: float,
        t_prec: float,
        v_prec: float,
        t_rec: float,
        v_rec: float,
        t_f1: float,
        v_f1: float,
        lr: float,
        duration: float,
    ) -> None:
        """Append epoch metrics to history lists."""
        self.epochs.append(epoch)
        self.train_loss.append(round(t_loss, 5))
        self.val_loss.append(round(v_loss, 5))
        self.train_acc.append(round(t_acc, 4))
        self.val_acc.append(round(v_acc, 4))
        self.train_precision.append(round(t_prec, 4))
        self.val_precision.append(round(v_prec, 4))
        self.train_recall.append(round(t_rec, 4))
        self.val_recall.append(round(v_rec, 4))
        self.train_f1.append(round(t_f1, 4))
        self.val_f1.append(round(v_f1, 4))
        self.learning_rates.append(lr)
        self.epoch_durations.append(round(duration, 2))

    def to_dict(self) -> Dict:
        """Convert training history to dictionary representation."""
        return {
            "epochs": self.epochs,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "train_acc": self.train_acc,
            "val_acc": self.val_acc,
            "train_precision": self.train_precision,
            "val_precision": self.val_precision,
            "train_recall": self.train_recall,
            "val_recall": self.val_recall,
            "train_f1": self.train_f1,
            "val_f1": self.val_f1,
            "learning_rates": self.learning_rates,
            "epoch_durations": self.epoch_durations,
            "best_val_loss": min(self.val_loss) if self.val_loss else None,
            "best_val_acc": max(self.val_acc) if self.val_acc else None,
            "best_val_f1": max(self.val_f1) if self.val_f1 else None,
            "total_duration_seconds": sum(self.epoch_durations),
        }

    def save_json(self, file_path: Path) -> None:
        """Save history to JSON file artifact.

        Args:
            file_path: Absolute destination file path.
        """
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=4)

    def save_csv(self, file_path: Path) -> None:
        """Save training history to CSV file artifact.

        Args:
            file_path: Absolute destination file path.
        """
        import csv

        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "epoch",
                    "train_loss",
                    "val_loss",
                    "train_acc",
                    "val_acc",
                    "train_precision",
                    "val_precision",
                    "train_recall",
                    "val_recall",
                    "train_f1",
                    "val_f1",
                    "learning_rate",
                    "epoch_duration_seconds",
                ]
            )
            for i in range(len(self.epochs)):
                writer.writerow(
                    [
                        self.epochs[i],
                        self.train_loss[i],
                        self.val_loss[i],
                        self.train_acc[i],
                        self.val_acc[i],
                        self.train_precision[i],
                        self.val_precision[i],
                        self.train_recall[i],
                        self.val_recall[i],
                        self.train_f1[i],
                        self.val_f1[i],
                        self.learning_rates[i],
                        self.epoch_durations[i],
                    ]
                )
