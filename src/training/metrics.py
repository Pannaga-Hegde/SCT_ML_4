"""Training and evaluation metrics calculator module for GestureFlow."""

from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class TrainingMetricsCalculator:
    """Metrics engine computing accuracy, precision, recall, macro F1, and per-class metrics."""

    @staticmethod
    def calculate_metrics(
        y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 10
    ) -> Dict[str, float]:
        """Compute top-1 accuracy, precision, recall, and macro F1-score.

        Args:
            y_true: Ground truth target label array.
            y_pred: Model predicted class label array.
            num_classes: Total number of gesture classes.

        Returns:
            Dictionary of scalar metric values.
        """
        acc = float(accuracy_score(y_true, y_pred))
        prec = float(
            precision_score(
                y_true, y_pred, average="macro", zero_division=0
            )
        )
        rec = float(
            recall_score(
                y_true, y_pred, average="macro", zero_division=0
            )
        )
        f1 = float(
            f1_score(
                y_true, y_pred, average="macro", zero_division=0
            )
        )

        return {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_macro": f1,
        }

    @staticmethod
    def calculate_per_class_accuracy(
        y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 10
    ) -> Dict[int, float]:
        """Compute per-class classification accuracy.

        Args:
            y_true: Ground truth target label array.
            y_pred: Model predicted class label array.
            num_classes: Total number of gesture classes.

        Returns:
            Dictionary mapping integer label ID to accuracy value.
        """
        per_class_acc = {}
        for cls_idx in range(num_classes):
            mask = y_true == cls_idx
            if np.sum(mask) > 0:
                cls_acc = float(np.mean(y_pred[mask] == y_true[mask]))
                per_class_acc[cls_idx] = cls_acc
            else:
                per_class_acc[cls_idx] = 0.0

        return per_class_acc
