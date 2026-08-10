# Architecture Decision Records (ADR) — GestureFlow

This document captures key architectural and technical decisions made throughout the lifecycle of the GestureFlow project.

---

## ADR Index

| ADR ID | Title | Date | Status |
| :--- | :--- | :--- | :--- |
| **ADR-0001** | Documentation-First Single Source of Truth Governance | 2026-08-05 | `Accepted` |
| **ADR-0002** | Adoption of LeapGestRecog Dataset for Baseline Model | 2026-08-05 | `Accepted` |
| **ADR-0003** | FastAPI and WebSocket Protocol for Real-Time Streaming (Superseded) | 2026-08-05 | `Superseded` |
| **ADR-0004** | Pure Vanilla HTML5/CSS3/JS Glassmorphic Frontend (Superseded) | 2026-08-05 | `Superseded` |
| **ADR-0005** | PyTorch Model Training with CPU Inference Execution | 2026-08-05 | `Accepted` |
| **ADR-0006** | Research-First Machine Learning Workflow | 2026-08-05 | `Accepted` |
| **ADR-0007** | Dataset Validation Before Model Development | 2026-08-05 | `Accepted` |
| **ADR-0008** | Machine Learning Focus with Desktop Inference Demonstration | 2026-08-06 | `Accepted` |
| **ADR-0009** | Adoption of Global Average Pooling for Lightweight GestureCNN Architecture | 2026-08-06 | `Accepted` |
| **ADR-0010** | Automated Experiment Tracking and Model Complexity Analysis Framework | 2026-08-06 | `Accepted` |

---

## ADR-0001: Documentation-First Single Source of Truth Governance

- **Date**: 2026-08-05
- **Status**: `Accepted`

### Context
Developing a complex ML system across sessions requires absolute consistency in architectural decisions, coding standards, UI specs, and phase transitions. Without centralized documentation, project intent drifts.

---

## ADR-0002: Adoption of LeapGestRecog Dataset for Baseline Model

- **Date**: 2026-08-05
- **Status**: `Accepted`

---

## ADR-0003: FastAPI and WebSocket Protocol for Real-Time Streaming (Superseded)

- **Date**: 2026-08-05
- **Status**: `Superseded by ADR-0008`

---

## ADR-0004: Pure Vanilla HTML5/CSS3/JS Glassmorphic Frontend (Superseded)

- **Date**: 2026-08-05
- **Status**: `Superseded by ADR-0008`

---

## ADR-0005: PyTorch Model Training with CPU Inference Execution

- **Date**: 2026-08-05
- **Status**: `Accepted`

---

## ADR-0006: Research-First Machine Learning Workflow

- **Date**: 2026-08-05
- **Status**: `Accepted`

---

## ADR-0007: Dataset Validation Before Model Development

- **Date**: 2026-08-05
- **Status**: `Accepted`

---

## ADR-0008: Machine Learning Focus with Desktop Inference Demonstration

- **Date**: 2026-08-06
- **Status**: `Accepted`

---

## ADR-0009: Adoption of Global Average Pooling for Lightweight GestureCNN Architecture

- **Date**: 2026-08-06
- **Status**: `Accepted`

### Context
When transitioning from spatial feature maps (e.g. Block 4 output shape $256 \times 8 \times 8$) to dense classification layers, traditional neural network architectures flatten feature maps into large linear vectors ($256 \times 8 \times 8 = 16,384$ features). Feeding 16,384 features into fully-connected hidden layers adds over $2.1\text{M}$ trainable parameters, creating a heavy model footprint ($\approx 10\text{ MB}$ checkpoint size), increasing spatial overfitting risks, and degrading real-time desktop CPU inference latency.

### Decision
Adopt **Global Average Pooling (`nn.AdaptiveAvgPool2d((1, 1))`)** in `GestureCNN` (`src/models/cnn.py`) prior to the fully-connected classification head.

### Alternatives Considered
- *Flatten Layer (`nn.Flatten()`) followed by Dense FC Layer*: Adds $\approx 2.1\text{M}$ parameters, increases spatial overfitting, slows CPU execution.
- *Max Pooling to $1\times 1$*: Tends to retain extreme feature activations rather than average spatial representations across spatial channels.

### Rationale
1. **Ultra-Lightweight Parameter Count**: Reduces total network parameters from $\approx 2.5\text{M}$ down to **422,506 parameters** ($\approx 1.6\text{ MB}$ FP32 checkpoint footprint).
2. **Spatial Translation Invariance**: Enforces feature map averaging across spatial dimensions, encouraging spatial generalization.
3. **Sub-5ms CPU Inference**: Enables rapid matrix operations during real-time desktop webcam inference loop execution.

### Consequences
- `GestureCNN` classifier head accepts input vector dimension of $256$ features (`Linear(256, 128)` -> `ReLU` -> `Dropout(0.4)` -> `Linear(128, 10)`).
- Model instantiates with 422,506 parameters, perfectly balancing high accuracy with ultra-fast real-time inference speed.

---

## ADR-0010: Automated Experiment Tracking and Model Complexity Analysis Framework

- **Date**: 2026-08-06
- **Status**: `Accepted`

### Context
Machine learning model development requires tracking hyperparameter choices, system configuration, git revisions, and detailed architectural complexity metrics to guarantee reproducibility across training runs and avoid manual overhead.

### Decision
Implement a zero-dependency, automated experiment tracking and model analysis framework (`ExperimentTracker`, `ModelComplexityAnalyzer`, `ModelSummaryGenerator`) operating before and during model training.

### Rationale
1. **Automated Governance**: Automatically tracks parameter counts, FP32 model memory size ($1,690,024\text{ bytes} / 1.612\text{ MB}$), input/output shapes, layer-by-layer MACs ($233,211,274$), and FLOPs ($466,422,548$).
2. **Standardized Artifact Structure**: Guarantees consistent creation of `experiment.json`, `training_history.json`, `training_history.csv`, `model_summary.txt`, `model_complexity.json`, and `training_log.txt` under `outputs/training/`.
3. **Reproducibility**: Links git commit SHA hashes, hyperparameters, seed settings, and dataset split metadata directly with training logs.

