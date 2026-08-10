# Changelog — GestureFlow

All notable changes to the **GestureFlow** project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- Completed Phase 4 Model Evaluation & Error Analysis.
- Evaluated `models/checkpoints/best_model.pth` on held-out Subject `09` test split (2,000 images).
- Achieved **100.00% Test Accuracy**, **1.0000 Macro F1**, **1.0000 Macro Precision**, **1.0000 Macro Recall**, and **0.0088 Test Loss**.
- Measured per-sample CPU inference latency of **3.98 ms** ($\approx 251\text{ FPS}$ throughput capability).
- Implemented `EvaluationMetricsCalculator` in `src/evaluation/metrics.py`.
- Implemented `ModelEvaluator` orchestrator in `src/evaluation/evaluator.py`.
- Implemented confusion matrix plotter in `src/evaluation/visualize.py` rendering `confusion_matrix.png` and `confusion_matrix_normalized.png`.
- Implemented failure analysis & deployment readiness engine in `src/evaluation/failure_analysis.py` publishing `failure_analysis.md`.
- Implemented CLI evaluation entry script `src/evaluation/evaluate.py`.
- Created unit tests in `tests/unit/test_evaluation_metrics.py` (`28 passed` total unit tests).
- Published evaluation summary artifacts in `outputs/evaluation/`: `classification_report.csv`, `classification_report.md`, `predictions.csv`, `evaluation_summary.json`, `confusion_matrix.png`, `confusion_matrix_normalized.png`, and `failure_analysis.md`.
- Completed Phase 3.3 CNN Training Execution & Best Model Selection.
- Executed full PyTorch `GestureCNN` training pipeline for 15 epochs on subject-isolated split (Subjects `00`–`06`, 13,800 images) with validation evaluation on Subjects `07`–`08` (4,000 images).
- Achieved **Best Validation Accuracy**: `98.25%` and **Best Validation Macro F1**: `0.9824` at **Epoch 10**.
- Early stopping triggered at Epoch 15 due to validation loss convergence (min val loss `0.10287`).
- Saved best model checkpoint to `models/checkpoints/best_model.pth` and latest checkpoint to `models/checkpoints/latest_model.pth`.
- Saved periodic checkpoints (`epoch_05.pth`, `epoch_10.pth`, `epoch_15.pth`).
- Exported epoch metrics to `outputs/training/training_history.json` and `training_history.csv`.
- Rendered 4 dark-slate training plots: `loss_curve.png`, `accuracy_curve.png`, `precision_recall_curve.png`, `learning_rate_schedule.png`.
- Published best model metadata artifact `models/checkpoints/best_model.json`.
- Published comprehensive training report `outputs/training/training_report.md`.
- Updated `outputs/training/experiment.json` with final experiment execution metadata and performance results.
- Completed Phase 3.2 Training Strategy & Experiment Configuration.
- Implemented `ExperimentTracker` in `src/training/experiment.py` recording experiment metadata, hyperparameters, system environment, and dataset links.
- Implemented `ModelComplexityAnalyzer` in `src/models/complexity.py` calculating parameter counts, FP32 size (1.612 MB), MACs (233,211,274), and FLOPs (466,422,548).
- Implemented `ModelSummaryGenerator` in `src/models/summary.py` outputting tabular layer shape and parameter count reports.
- Added `save_csv` export capability to `TrainingHistory` in `src/training/history.py`.
- Created experiment setup script `src/training/setup_experiment.py` initializing all `outputs/training/` artifacts: `experiment.json`, `training_history.json`, `training_history.csv`, `model_summary.txt`, `model_complexity.json`, and `training_log.txt`.
- Implemented comprehensive pre-training verification test suite in `tests/unit/test_experiment_tracking.py` (`24 passed` unit tests).
- Recorded **ADR-0010: Experiment Tracking and Model Complexity Analysis Framework** in `docs/decisions.md`.
- Completed Phase 3.1 CNN Architecture & Training Framework Design.
- Created `TrainingConfig` dataclass in `src/config/training_config.py`.
- Designed `GestureCNN` 4-block neural network module in `src/models/cnn.py` featuring Global Average Pooling and 422,506 parameters.
- Implemented `TrainingHistory` tracker in `src/training/history.py`.
- Implemented `TrainingMetricsCalculator` in `src/training/metrics.py`.
- Implemented `EarlyStopping` and `ModelCheckpoint` callbacks in `src/training/callbacks.py`.
- Implemented `ModelTrainer` orchestrator in `src/training/trainer.py`.
- Implemented unit tests in `tests/unit/test_model_cnn.py` and `tests/unit/test_training_framework.py` (`19 passed` unit tests).
- Recorded **ADR-0009: Adoption of Global Average Pooling for Lightweight GestureCNN Architecture** in `docs/decisions.md`.

---

## [0.1.0] — 2026-08-05

### Added
- Established project governance framework in `docs/`.
- Completed Phase 1 governance documentation and Phase 2 Dataset Audit & Preprocessing pipeline.
- Recorded **ADR-0008: Machine Learning Focus with Desktop Inference Demonstration** in `docs/decisions.md`.
