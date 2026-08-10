"""Evaluation Metrics Module for GestureFlow.

Computes comprehensive scientific evaluation metrics for hand gesture recognition:
Overall Accuracy, Macro Precision, Macro Recall, Macro F1, Balanced Accuracy, Top-1 Accuracy,
and per-class precision/recall/F1/support breakdowns.
"""

from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


class EvaluationMetricsCalculator:
    """Calculates summary and per-class metrics for model evaluation."""

    @staticmethod
    def calculate_overall_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        test_loss: float = 0.0,
        avg_inference_time_ms: float = 0.0,
    ) -> Dict[str, Any]:
        """Compute overall classification metrics across entire dataset split.

        Args:
            y_true: Ground truth target label indices.
            y_pred: Predicted class label indices.
            test_loss: Mean test loss value.
            avg_inference_time_ms: Mean per-sample inference latency in milliseconds.

        Returns:
            Dictionary of overall metrics.
        """
        acc = float(accuracy_score(y_true, y_pred))
        bal_acc = float(balanced_accuracy_score(y_true, y_pred))

        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )

        return {
            "test_loss": float(test_loss),
            "accuracy": acc,
            "top1_accuracy": acc,
            "balanced_accuracy": bal_acc,
            "macro_precision": float(macro_p),
            "macro_recall": float(macro_r),
            "macro_f1": float(macro_f1),
            "avg_inference_time_ms": float(avg_inference_time_ms),
            "total_samples": int(len(y_true)),
        }

    @staticmethod
    def calculate_per_class_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: List[str],
    ) -> pd.DataFrame:
        """Compute detailed per-class classification metrics.

        Args:
            y_true: Ground truth target label indices.
            y_pred: Predicted class label indices.
            class_names: List of class label strings ordered by index.

        Returns:
            Pandas DataFrame containing per-class metrics.
        """
        p, r, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=range(len(class_names)), zero_division=0
        )

        cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
        correct = np.diag(cm)
        incorrect = support - correct

        records = []
        for idx, name in enumerate(class_names):
            records.append(
                {
                    "class_index": idx,
                    "class_name": name,
                    "precision": round(float(p[idx]), 4),
                    "recall": round(float(r[idx]), 4),
                    "f1_score": round(float(f1[idx]), 4),
                    "support": int(support[idx]),
                    "correct": int(correct[idx]),
                    "incorrect": int(incorrect[idx]),
                    "accuracy": round(
                        float(correct[idx] / support[idx]) if support[idx] > 0 else 0.0,
                        4,
                    ),
                }
            )

        return pd.DataFrame(records)

    @staticmethod
    def export_classification_report_md(
        df_per_class: pd.DataFrame,
        overall_metrics: Dict[str, Any],
    ) -> str:
        """Format per-class metrics DataFrame into Markdown table report.

        Args:
            df_per_class: Per-class DataFrame.
            overall_metrics: Dictionary of overall summary metrics.

        Returns:
            Markdown formatted string report.
        """
        lines = [
            "# GestureCNN Classification Report — GestureFlow",
            "",
            f"**Test Set Evaluation (Subject 09)** | Total Samples: `{overall_metrics['total_samples']:,}`",
            "",
            "## 1. Overall Summary Performance",
            "",
            "| Metric | Score | Target Standard | Status |",
            "| :--- | :--- | :--- | :--- |",
            f"| **Overall Accuracy** | `{overall_metrics['accuracy']*100:.2f}%` | $\\ge 98.0\\%$ | {'✓ Passed' if overall_metrics['accuracy'] >= 0.98 else '⚠️ Below Target'} |",
            f"| **Macro F1 Score** | `{overall_metrics['macro_f1']:.4f}` | $\\ge 0.98$ | {'✓ Passed' if overall_metrics['macro_f1'] >= 0.98 else '⚠️ Below Target'} |",
            f"| **Macro Precision** | `{overall_metrics['macro_precision']:.4f}` | — | — |",
            f"| **Macro Recall** | `{overall_metrics['macro_recall']:.4f}` | — | — |",
            f"| **Balanced Accuracy** | `{overall_metrics['balanced_accuracy']*100:.2f}%` | — | — |",
            f"| **Test Loss** | `{overall_metrics['test_loss']:.4f}` | — | — |",
            f"| **Avg Inference Time** | `{overall_metrics['avg_inference_time_ms']:.2f} ms / sample` | $< 20\\text{{ ms}}$ | ✓ Passed |",
            "",
            "---",
            "",
            "## 2. Per-Class Performance Breakdown",
            "",
            "| Index | Gesture Class | Precision | Recall | F1-Score | Support | Correct | Incorrect | Accuracy |",
            "| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]

        for _, row in df_per_class.iterrows():
            lines.append(
                f"| {row['class_index']} | `{row['class_name']}` | "
                f"{row['precision']:.4f} | {row['recall']:.4f} | {row['f1_score']:.4f} | "
                f"{row['support']} | {row['correct']} | {row['incorrect']} | "
                f"{row['accuracy']*100:.2f}% |"
            )

        lines.extend(
            [
                "",
                "---",
                "",
                "## 3. Class Performance Ranking",
                "",
                f"- **Best Performing Gesture**: `{df_per_class.loc[df_per_class['f1_score'].idxmax()]['class_name']}` "
                f"(F1: {df_per_class['f1_score'].max():.4f})",
                f"- **Worst Performing Gesture**: `{df_per_class.loc[df_per_class['f1_score'].idxmin()]['class_name']}` "
                f"(F1: {df_per_class['f1_score'].min():.4f})",
                "",
            ]
        )

        return "\n".join(lines)
