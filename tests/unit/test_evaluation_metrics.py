"""Unit tests for Phase 4 evaluation metrics, plots, and failure analysis."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import EvaluationMetricsCalculator
from src.evaluation.visualize import generate_confusion_matrix_plots
from src.evaluation.failure_analysis import FailureAnalyzer


@pytest.fixture
def dummy_data():
    """Fixture providing dummy true and predicted labels across 10 classes."""
    y_true = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    # Introduce 1 misclassification (class 2 predicted as class 3)
    y_pred = np.array([0, 1, 3, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    class_names = [f"class_{i:02d}" for i in range(10)]
    return y_true, y_pred, class_names


def test_calculate_overall_metrics(dummy_data):
    """Verify overall metric calculations."""
    y_true, y_pred, _ = dummy_data
    metrics = EvaluationMetricsCalculator.calculate_overall_metrics(
        y_true=y_true, y_pred=y_pred, test_loss=0.05, avg_inference_time_ms=2.5
    )

    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert "balanced_accuracy" in metrics
    assert metrics["accuracy"] == 0.95
    assert metrics["total_samples"] == 20
    assert metrics["avg_inference_time_ms"] == 2.5


def test_calculate_per_class_metrics(dummy_data):
    """Verify per-class metrics DataFrame structure."""
    y_true, y_pred, class_names = dummy_data
    df_per_class = EvaluationMetricsCalculator.calculate_per_class_metrics(
        y_true=y_true, y_pred=y_pred, class_names=class_names
    )

    assert len(df_per_class) == 10
    assert "precision" in df_per_class.columns
    assert "recall" in df_per_class.columns
    assert "f1_score" in df_per_class.columns
    assert "support" in df_per_class.columns
    assert df_per_class["support"].sum() == 20


def test_confusion_matrix_plots(dummy_data, tmp_path):
    """Verify confusion matrix plot creation."""
    y_true, y_pred, class_names = dummy_data
    p_raw, p_norm = generate_confusion_matrix_plots(
        y_true=y_true, y_pred=y_pred, class_names=class_names, output_dir=tmp_path
    )

    assert p_raw.exists()
    assert p_norm.exists()
    assert p_raw.stat().st_size > 0
    assert p_norm.stat().st_size > 0


def test_failure_analyzer(dummy_data, tmp_path):
    """Verify failure analysis engine."""
    y_true, y_pred, class_names = dummy_data

    pred_records = []
    for i in range(len(y_true)):
        pred_records.append(
            {
                "relative_path": f"sample_{i}.png",
                "true_label_idx": int(y_true[i]),
                "true_label": class_names[y_true[i]],
                "pred_label_idx": int(y_pred[i]),
                "pred_label": class_names[y_pred[i]],
                "confidence": 0.95 if y_true[i] == y_pred[i] else 0.88,
                "is_correct": bool(y_true[i] == y_pred[i]),
            }
        )
    df_predictions = pd.DataFrame(pred_records)

    analyzer = FailureAnalyzer(df_predictions=df_predictions, class_names=class_names)
    ranking = analyzer.get_per_class_ranking()
    assert len(ranking) == 10

    report_path = analyzer.generate_report_md(
        overall_metrics={"accuracy": 0.95, "macro_f1": 0.95, "avg_inference_time_ms": 2.5},
        output_path=tmp_path / "failure_analysis.md",
    )
    assert report_path.exists()
    assert "Phase 5 Deployment Readiness Assessment" in report_path.read_text(encoding="utf-8")
