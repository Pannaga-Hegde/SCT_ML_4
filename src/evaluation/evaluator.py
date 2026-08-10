"""ModelEvaluator Orchestrator for GestureFlow.

Loads trained PyTorch GestureCNN checkpoint artifact (best_model.pth),
evaluates performance on the held-out Subject 09 test set, measures latency,
logs individual predictions, and exports structured evaluation artifacts.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config.training_config import TrainingConfig, training_config
from src.dataset.loader import create_dataloaders
from src.models.cnn import GestureCNN
from src.evaluation.metrics import EvaluationMetricsCalculator
from src.evaluation.visualize import generate_confusion_matrix_plots
from src.evaluation.failure_analysis import FailureAnalyzer


class ModelEvaluator:
    """Evaluates trained GestureCNN checkpoint on test set split."""

    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        test_loader: Optional[DataLoader] = None,
        cfg: TrainingConfig = training_config,
    ) -> None:
        """Initialize ModelEvaluator.

        Args:
            checkpoint_path: Path to best_model.pth checkpoint.
            test_loader: DataLoader for test set split.
            cfg: TrainingConfig instance.
        """
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.checkpoint_path = (
            Path(checkpoint_path)
            if checkpoint_path
            else cfg.best_checkpoint_path
        )

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found at: {self.checkpoint_path}"
            )

        # Load class names from classes.json if available
        classes_path = Path("outputs/dataset/classes.json")
        if classes_path.exists():
            with open(classes_path, "r", encoding="utf-8") as f:
                raw_classes = json.load(f)
                # Ensure sorted by integer key
                self.class_names = [raw_classes[str(i)] for i in range(len(raw_classes))]
        else:
            self.class_names = [f"class_{i:02d}" for i in range(cfg.num_classes)]

        # Initialize DataLoader if not provided
        if test_loader is None:
            _, _, self.test_loader = create_dataloaders()
        else:
            self.test_loader = test_loader

        # Load GestureCNN Model Architecture & Checkpoint Weights
        self.model = GestureCNN(
            num_classes=cfg.num_classes, dropout_rate=cfg.dropout_rate
        ).to(self.device)

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.eval()
        self.criterion = nn.CrossEntropyLoss()

    def evaluate(self) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
        """Execute full evaluation loop on held-out test split.

        Returns:
            Tuple of (overall_metrics_dict, df_per_class, df_predictions).
        """
        all_y_true = []
        all_y_pred = []
        all_confidences = []
        all_rel_paths = []
        running_loss = 0.0

        inference_times_ms = []

        with torch.no_grad():
            for images, labels, meta in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                batch_start = time.time()
                outputs = self.model(images)
                batch_time_ms = (time.time() - batch_start) * 1000.0 / images.size(0)

                loss = self.criterion(outputs, labels)
                running_loss += loss.item() * images.size(0)

                probs = torch.softmax(outputs, dim=1)
                confidences, preds = torch.max(probs, dim=1)

                rel_paths = meta["relative_path"] if isinstance(meta, dict) and "relative_path" in meta else meta

                all_y_true.extend(labels.cpu().numpy())
                all_y_pred.extend(preds.cpu().numpy())
                all_confidences.extend(confidences.cpu().numpy())
                all_rel_paths.extend(rel_paths)
                inference_times_ms.extend([batch_time_ms] * images.size(0))

        y_true = np.array(all_y_true)
        y_pred = np.array(all_y_pred)
        confidences = np.array(all_confidences)
        test_loss = running_loss / len(self.test_loader.dataset)
        avg_inference_time = float(np.mean(inference_times_ms))

        # Overall Metrics
        overall_metrics = EvaluationMetricsCalculator.calculate_overall_metrics(
            y_true=y_true,
            y_pred=y_pred,
            test_loss=test_loss,
            avg_inference_time_ms=avg_inference_time,
        )
        overall_metrics["average_confidence"] = float(np.mean(confidences))
        overall_metrics["parameter_count"] = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        overall_metrics["model_size_mb"] = round(
            self.checkpoint_path.stat().st_size / (1024 * 1024), 3
        )

        # Per-Class Metrics
        df_per_class = EvaluationMetricsCalculator.calculate_per_class_metrics(
            y_true=y_true,
            y_pred=y_pred,
            class_names=self.class_names,
        )

        # Detailed Predictions Log
        pred_records = []
        for i in range(len(y_true)):
            pred_records.append(
                {
                    "relative_path": all_rel_paths[i],
                    "true_label_idx": int(y_true[i]),
                    "true_label": self.class_names[y_true[i]],
                    "pred_label_idx": int(y_pred[i]),
                    "pred_label": self.class_names[y_pred[i]],
                    "confidence": round(float(confidences[i]), 4),
                    "is_correct": bool(y_true[i] == y_pred[i]),
                }
            )
        df_predictions = pd.DataFrame(pred_records)

        return overall_metrics, df_per_class, df_predictions

    def save_artifacts(
        self,
        overall_metrics: Dict[str, Any],
        df_per_class: pd.DataFrame,
        df_predictions: pd.DataFrame,
        output_dir: Union[str, Path] = "outputs/evaluation",
    ) -> Dict[str, Path]:
        """Export all evaluation artifacts to outputs/evaluation/.

        Args:
            overall_metrics: Summary metrics dict.
            df_per_class: Per-class metrics DataFrame.
            df_predictions: Individual predictions DataFrame.
            output_dir: Target output folder.

        Returns:
            Dictionary mapping artifact names to output file paths.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        artifact_paths = {}

        # 1. classification_report.csv
        csv_rep_path = output_dir / "classification_report.csv"
        df_per_class.to_csv(csv_rep_path, index=False)
        artifact_paths["classification_report_csv"] = csv_rep_path

        # 2. classification_report.md
        md_rep_path = output_dir / "classification_report.md"
        report_md_str = EvaluationMetricsCalculator.export_classification_report_md(
            df_per_class=df_per_class,
            overall_metrics=overall_metrics,
        )
        with open(md_rep_path, "w", encoding="utf-8") as f:
            f.write(report_md_str)
        artifact_paths["classification_report_md"] = md_rep_path

        # 3. predictions.csv
        pred_csv_path = output_dir / "predictions.csv"
        df_predictions.to_csv(pred_csv_path, index=False)
        artifact_paths["predictions_csv"] = pred_csv_path

        # 4. evaluation_summary.json
        sum_json_path = output_dir / "evaluation_summary.json"
        with open(sum_json_path, "w", encoding="utf-8") as f:
            json.dump(overall_metrics, f, indent=4)
        artifact_paths["evaluation_summary_json"] = sum_json_path

        # 5. Confusion Matrices (confusion_matrix.png, confusion_matrix_normalized.png)
        y_true = df_predictions["true_label_idx"].values
        y_pred = df_predictions["pred_label_idx"].values
        path_raw, path_norm = generate_confusion_matrix_plots(
            y_true=y_true,
            y_pred=y_pred,
            class_names=self.class_names,
            output_dir=output_dir,
        )
        artifact_paths["confusion_matrix_raw"] = path_raw
        artifact_paths["confusion_matrix_normalized"] = path_norm

        # 6. failure_analysis.md
        analyzer = FailureAnalyzer(
            df_predictions=df_predictions, class_names=self.class_names
        )
        fail_path = analyzer.generate_report_md(
            overall_metrics=overall_metrics,
            output_path=output_dir / "failure_analysis.md",
        )
        artifact_paths["failure_analysis_md"] = fail_path

        return artifact_paths
