# Product Roadmap — GestureFlow

This document presents the Machine Learning project roadmap for GestureFlow.

---

```mermaid
timeline
    title GestureFlow 6-Phase Machine Learning Roadmap
    section Milestone 1
        Project Foundation : Phase 1
        Dataset Audit & Preprocessing : Phase 2
        Subject Split & PyTorch DataLoader : Phase 2
    section Milestone 2
        PyTorch CNN Architecture : Phase 3
        Trainer & Early Stopping : Phase 3
        Model Checkpoint Saving : Phase 3
    section Milestone 3
        Test Set Evaluation : Phase 4
        Confusion Matrix & Metrics : Phase 4
        Error Analysis Report : Phase 4
    section Milestone 4
        Offline Image Inference : Phase 5
        Real-Time Desktop Webcam Demo : Phase 5
        README & Final Project Release : Phase 6
```

---

## 6-Phase Technical Sequence

```mermaid
flowchart TD
    P1[Phase 1: Project Foundation & Documentation] --> P2[Phase 2: Dataset Audit & Preprocessing]
    P2 --> P3[Phase 3: CNN Development & Training]
    P3 --> P4[Phase 4: Model Evaluation & Error Analysis]
    P4 --> P5[Phase 5: Real-Time Desktop Inference Demo]
    P5 --> P6[Phase 6: Documentation & Repository Finalization]
```

---

## Milestone Deliverables

### Milestone 1: Project Foundation & Dataset Preprocessing (Phases 1 & 2)
- Complete single-source-of-truth documentation (`docs/`).
- Automated Dataset Audit (`src/dataset/audit.py`) and EDA outputs (`outputs/dataset/`).
- Enforced subject-aware splitter ($70\% / 15\% / 15\%$) isolating subjects `00`–`06` (Train), `07`–`08` (Val), and `09` (Test).
- PyTorch `GestureDataset` and `DataLoader` implementation with safe data augmentations.

### Milestone 2: Model Architecture & PyTorch Training (Phase 3)
- PyTorch Convolutional Neural Network (`GestureCNN`).
- Configurable trainer loop with AdamW optimizer, Cross-Entropy Loss, and Early Stopping.
- Saved best model checkpoint (`models/checkpoints/best_model.pth`).
- Training history logs and loss/accuracy curves (`outputs/training/training_curves.png`).

### Milestone 3: Evaluation & Error Analysis (Phase 4)
- Quantitative test set evaluation outputting classification report and macro F1-score ($\ge 0.98$).
- $10 \times 10$ confusion matrix heatmap and misclassified sample diagnostic grid (`outputs/evaluation/`).
- Comprehensive error analysis report published in `outputs/evaluation/error_analysis_report.md`.

### Milestone 4: Real-Time Inference Demo & Project Finalization (Phases 5 & 6)
- Standalone inference engine (`src/inference/predictor.py`) for single-image and batch predictions.
- Real-time desktop webcam demonstration application (`src/inference/webcam.py`) rendering live gesture overlay, confidence %, FPS counter, and latency metrics.
- MediaPipe utilized strictly for optional hand ROI localization; classification performed exclusively by PyTorch CNN.
- Comprehensive `README.md` containing project overview, quickstart setup, benchmark tables, and embedded result plots.
