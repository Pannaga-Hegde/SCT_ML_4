# Engineering Rules & Standards — GestureFlow

This document defines mandatory engineering standards for GestureFlow. All code, neural network architectures, dataset loaders, inference scripts, and documentation contributions must strictly adhere to these rules.

---

## 1. Architecture Rules
- **Modular Machine Learning Lifecycle**: Source code must be organized into single-purpose Python packages under `src/` (`config/`, `dataset/`, `models/`, `training/`, `evaluation/`, `inference/`, `utils/`).
- **Single Source of Truth**: Architectural designs, hyperparameters, and evaluation targets must originate from the `docs/` directory.
- **Standalone Local Execution**: The project must execute standalone locally without requiring web servers, REST endpoints, WebSockets, Docker, or external database services.

---

## 2. Mandatory Machine Learning Rules

### Enforced Subject-Aware Splitting Rule
> [!CAUTION]
> **Strict Prohibition**: Random image-level splitting across train, validation, and test sets is **permanently prohibited**.

- **Rule**: All images originating from a subject folder (`00` through `09`) must belong exclusively to a single split (Train, Validation, or Test).
- **Default Split**: Subjects `00`–`06` (Train), `07`–`08` (Validation), `09` (Test).
- **Rationale**: Prevents data leakage and guarantees true out-of-subject model generalization.
- **Reproducibility**: Set explicit random seeds across Python `random`, `numpy`, and `torch` (`seed=42`).

---

## 3. Real-Time Inference & MediaPipe Rules

- **OpenCV Display Loop**: Real-time webcam inference (`src/inference/webcam.py`) must run natively via OpenCV desktop windows displaying top-1 gesture prediction, confidence score (%), FPS counter, and inference latency (ms).
- **MediaPipe Usage Boundary**:
  - MediaPipe may be used **only** for hand localization and Region-of-Interest (ROI) bounding box extraction.
  - Hand gesture recognition **must always be performed exclusively by the trained PyTorch CNN model**.

---

## 4. Coding Standards
- **Python**: Follow PEP 8 guidelines strictly, enforce type annotations, format with `black`, and lint with `flake8`.
- **Docstrings**: Google-style docstrings for public classes and functions describing arguments, return values, and exceptions.

---

## 5. Documentation & Artifact Rules
- All dataset audit reports, statistics JSONs, and EDA plots must be generated in `outputs/dataset/`.
- All model checkpoints must be saved to `models/checkpoints/` (e.g. `models/checkpoints/best_model.pth`).
- All training logs and loss curves must be saved to `outputs/training/`.
- All evaluation reports and confusion matrices must be saved to `outputs/evaluation/`.
- All inference overlay samples and benchmarks must be saved to `outputs/inference/`.
- Update `docs/memory.md` and `docs/changelog.md` after completing any phase task.

---

## 6. Data Augmentation Rules
- **Safe Augmentations Allowed**: Subtle rotation ($\pm 10^\circ$), slight translation ($\pm 5\%$), brightness/contrast adjust ($\pm 15\%$).
- **Unsafe Augmentations Prohibited**:
  - `RandomVerticalFlip` is **strictly prohibited** as it inverts `05_thumb` (Thumb Up) into `10_down` (Hand Down), corrupting label semantics.

---

## 7. Definition of Done (DoD)
A task or phase is considered **DONE** only when:
1. Code is fully implemented according to specifications in `docs/prd.md` and `docs/architecture.md`.
2. Subject-aware dataset splitting rules are strictly verified.
3. Unit tests in `tests/` pass with zero errors.
4. Code passes style formatting (`black`) and linting (`flake8`).
5. All required output artifacts (plots, JSON stats, markdown reports) are generated.
6. `docs/memory.md` and `docs/changelog.md` are updated.
