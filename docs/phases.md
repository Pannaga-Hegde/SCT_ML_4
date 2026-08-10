# Development Phases & Roadmap — GestureFlow

> [!IMPORTANT]
> **Strict Operational Rule**: Exactly **ONE** phase can be marked as `ACTIVE` at any point in time. All other phases must be either `COMPLETED` or `PENDING`. Work must proceed strictly in sequence.

---

## Overview of 6 ML Development Phases

```mermaid
gantt
    title GestureFlow 6-Phase Machine Learning Lifecycle
    dateFormat  YYYY-MM-DD
    section ML Development Lifecycle
    Phase 1: Project Foundation & Documentation   :done, p1, 2026-08-05, 1d
    Phase 2: Dataset Audit & Preprocessing        :done, p2, after p1, 2d
    Phase 3: CNN Development & Training          :done, p3, after p2, 3d
    Phase 4: Model Evaluation & Error Analysis    :done, p4, after p3, 2d
    Phase 5: Real-Time Desktop Inference Demo     :done, p5, after p4, 2d
    Phase 6: Documentation & Repository Finalize  :done, p6, after p5, 1d
```

---

## Phase 1 — Project Foundation & Documentation

- **Status**: `COMPLETED`
- **Goal**: Establish the documentation framework, technical design specifications, engineering rules, dataset governance manual, architecture, and repository directory layout.
- **Deliverables**:
  - `docs/prd.md`
  - `docs/architecture.md`
  - `docs/phases.md`
  - `docs/rules.md`
  - `docs/design.md`
  - `docs/dataset.md`
  - `docs/memory.md`
  - `docs/roadmap.md`
  - `docs/changelog.md`
  - `docs/decisions.md` (ADR-0001 through ADR-0008)
  - `docs/AGENTS.md`
- **Exit Criteria**:
  - [x] All 11 governance documents completed and synchronized in `docs/`.
  - [x] Architecture refactored to focus on ML research lifecycle and desktop webcam demonstration.
  - [x] Directory structure defined (`src/dataset/`, `src/models/`, `src/training/`, `src/evaluation/`, `src/inference/`).

---

## Phase 2 — Dataset Audit & Preprocessing

- **Status**: `COMPLETED`
- **Goal**: Audit raw `archive/leapGestRecog/` dataset, compute image statistics, perform duplicate & corruption scans, enforce subject-aware dataset splits ($70\% / 15\% / 15\%$), apply safe augmentations, implement PyTorch DataLoaders, generate canonical manifest, and pass scientific dataset certification.
- **Deliverables**:
  - `src/dataset/validator.py`: Automated `DatasetValidator` verification engine.
  - `src/dataset/statistics.py`: `DatasetStatistics` engine generating EDA plots.
  - `src/dataset/metadata.py`: `DatasetMetadataGenerator` producing metadata artifacts.
  - `src/dataset/splitter.py`: Enforced `SubjectAwareSplitter` ($70\% / 15\% / 15\%$).
  - `src/dataset/transforms.py`: Preprocessing transforms & safe data augmentation.
  - `src/dataset/loader.py`: `GestureDataset` PyTorch `Dataset` and `DataLoader` factory methods.
  - `src/dataset/audit.py`: `DatasetAuditor` CLI orchestrator engine.
  - `src/dataset/verify_integrity.py`: Manifest generator, duplicate analyzer, and certification engine.
  - Unit tests in `tests/unit/` (`11 passed`).
- **Generated Artifacts (`outputs/dataset/`)**:
  - `outputs/dataset/manifest.csv` (20,000 canonical dataset records)
  - `outputs/dataset/duplicate_analysis.md`
  - `outputs/dataset/dataset_certification.md` (`CERTIFIED FOR PHASE 3`)
  - `outputs/dataset/dataset_report.md`
  - `outputs/dataset/validation_report.md`
  - `outputs/dataset/dataset_statistics.json` & `dataset_statistics.csv`
  - `outputs/dataset/dataset_metadata.json` & `normalization.json`
  - `outputs/dataset/classes.json`
  - `outputs/dataset/dataset_split.json`
  - `outputs/dataset/class_distribution.png`
  - `outputs/dataset/subject_distribution.png`
  - `outputs/dataset/resolution_distribution.png`
  - `outputs/dataset/sample_grid.png`
- **Exit Criteria**:
  - [x] 100% of 20,000 dataset images audited with zero corrupted files.
  - [x] SHA-256 duplicate image detection scan completed (388 clusters categorized; 0 cross-split duplicates, 0 cross-class duplicates).
  - [x] Subject-aware split ($70\% / 15\% / 15\%$) verifies zero subject overlap across splits.
  - [x] PyTorch `GestureDataset` and `DataLoader` verified via passing unit tests.
  - [x] Canonical `manifest.csv` index generated.
  - [x] Dataset certification passed (`CERTIFIED FOR PHASE 3`).

---

## Phase 3 — CNN Development & Training

- **Status**: `COMPLETED`
- **Goal**: Implement PyTorch `GestureCNN` neural network architecture, build training pipeline with loss scheduling, Early Stopping, and save model checkpoint artifacts.
- **Deliverables**:
  - `src/models/cnn.py`: PyTorch `GestureCNN` module.
  - `src/training/callbacks.py`: Early stopping & checkpoint saving callbacks.
  - `src/training/trainer.py`: `ModelTrainer` class executing training loops.
  - `src/training/train.py`: Training CLI entry script.
- **Generated Artifacts**:
  - Checkpoint file in `models/checkpoints/best_model.pth`
  - `models/checkpoints/latest_model.pth` & `models/checkpoints/best_model.json`
  - `outputs/training/training_history.json` & `training_history.csv`
  - `outputs/training/loss_curve.png`, `accuracy_curve.png`, `precision_recall_curve.png`, `learning_rate_schedule.png`
  - `outputs/training/training_report.md` & `outputs/training/experiment.json`
- **Dependencies**: Phase 2.
- **Exit Criteria**:
  - [x] Model successfully trains for target epochs without error (15 epochs ran; early stopping triggered).
  - [x] Validation accuracy reaches threshold $\ge 95.0\%$ on subject-split validation set (`98.25%` achieved).
  - [x] Training loss and validation loss curves logged and saved to `outputs/training/`.

---

## Phase 4 — Model Evaluation

- **Status**: `COMPLETED`
- **Goal**: Evaluate trained model on test set (Subject `09`), compute classification metrics, generate confusion matrices, and analyze misclassified samples.
- **Deliverables**:
  - `src/evaluation/metrics.py`: Metrics calculator for accuracy, precision, recall, and macro F1-score.
  - `src/evaluation/evaluate.py`: Evaluation execution script saving metric artifacts.
  - `src/evaluation/visualize.py`: Misclassified sample extractor and diagnostic visualizer.
- **Generated Artifacts (`outputs/evaluation/`)**:
  - `outputs/evaluation/classification_report.json`
  - `outputs/evaluation/confusion_matrix.png`
  - `outputs/evaluation/precision_recall_breakdown.png`
  - `outputs/evaluation/misclassified_samples.png`
  - `outputs/evaluation/error_analysis_report.md`
- **Dependencies**: Phase 3.
- **Exit Criteria**:
  - Final test set classification accuracy $\ge 98.0\%$.
  - Macro F1-Score $\ge 0.98$ calculated and saved.
  - Confusion matrix heatmap ($10 \times 10$) generated.
  - Error analysis report published identifying misclassified gesture pairs.

---

## Phase 5 — Real-Time Inference

- **Status**: `COMPLETED`
- **Goal**: Implement modular local desktop webcam inference application with MediaPipe hand detection, live-switchable preprocessing modes, multi-stage prediction stabilization, interactive keyboard controls, and a developer debug window.
- **Deliverables**:
  - `src/config/inference_config.py`: Central runtime configuration (`InferenceConfig` dataclass + `PreprocessMode` enum).
  - `src/inference/camera.py`: `CameraStream` OpenCV wrapper with FPS telemetry.
  - `src/inference/hand_detector.py`: `HandDetector` MediaPipe Hands integration.
  - `src/inference/preprocess.py`: `HandROIPreprocessor` — 3-mode preprocessing (Gray / HistEq / CLAHE).
  - `src/inference/predictor.py`: `GesturePredictor` — GestureCNN forward pass + Top-3 output + latency measurement.
  - `src/inference/stabilizer.py`: `PredictionStabilizer` — majority vote + confidence gate + consecutive-frame gate.
  - `src/inference/overlay.py`: `OverlayRenderer` — dark-slate HUD, bounding box, landmark skeleton, history panel, developer window.
  - `src/inference/demo.py`: `run_demo()` main loop — full keyboard controls (Q/R/M/L/B/S/F/P/C/O/H/D/1/2/3/SPACE), session CSV logging, debug bundle saving, session summary.
  - `tests/unit/test_inference.py`: 32 unit tests (all passing).
- **Inference Pipeline**:
  `Webcam` → `CameraStream` → `HandDetector (MediaPipe)` → `HandROIPreprocessor` → `GesturePredictor (PyTorch CNN)` → `PredictionStabilizer` → `OverlayRenderer` → `OpenCV Display`.
- **Generated Artifacts**:
  - `outputs/inference/session_log.csv` (per-frame CSV log)
  - `outputs/inference/debug/` (debug bundles on `O` keypress)
  - `outputs/inference/screenshots/` (screenshots on `C`/`SPACE` keypress)
- **Dependencies**: Phase 4.
- **Exit Criteria**:
  - [x] All 7 inference modules implemented and verified.
  - [x] 32 unit tests pass (no camera/GPU required).
  - [x] Keyboard controls: Q, R, M, L, B, S, F, P, C, O, H, D, 1, 2, 3, SPACE.
  - [x] Developer Mode window: ROI, tensor heatmap, probability bars, Top-3 table.
  - [x] Live-switchable preprocessing modes (Gray / HistEq / CLAHE).
  - [x] Multi-stage stabilizer: majority vote + confidence gate (≥70%) + consecutive frame gate.
  - [x] Session CSV logging and debug bundle saving.

---

## Phase 6 — Documentation & Repository Finalization

- **Status**: `COMPLETED`
- **Goal**: Finalize project README, installation instructions, results summary, screenshots, project report, repository cleanup, reproducibility guide, and final GitHub structure.
- **Deliverables**:
  - Comprehensive `README.md` with project description, setup commands, benchmark tables, screenshot galleries, and embedded architecture diagrams.
  - Formal 17-section `PROJECT_REPORT.md`.
  - Step-by-step `REPRODUCIBILITY.md` protocol.
  - Clean `requirements.txt` and `LICENSE` (MIT).
  - Organized visual assets under `assets/screenshots/`, `assets/demo/`, `assets/architecture/`.
  - Final `docs/changelog.md` and `docs/memory.md` synchronization.
  - Final repository cleanup and verification.
- **Dependencies**: Phase 5.
- **Exit Criteria**:
  - [x] `README.md` fully documented with clear instructions, benchmark tables, and embedded training/evaluation plots.
  - [x] `PROJECT_REPORT.md` and `REPRODUCIBILITY.md` published.
  - [x] Clean repository state with zero scratch files or temporary logs.
  - [x] All 60 unit tests pass cleanly.

---

## Final Project Status

```
Phase 1 — COMPLETED
Phase 2 — COMPLETED
Phase 3 — COMPLETED
Phase 4 — COMPLETED
Phase 5 — COMPLETED
Phase 6 — COMPLETED

Project Status: COMPLETE — READY FOR ML INTERNSHIP SUBMISSION
```
