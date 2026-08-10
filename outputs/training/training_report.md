# GestureCNN Training & Best Model Selection Report — GestureFlow

**Experiment ID**: `gesture_cnn_baseline_v1`  
**Execution Timestamp**: 2026-08-06 19:49:32  
**Model Architecture**: `GestureCNN` (4 Convolution Blocks + GAP + 2-layer Classifier Head)  

---

## 1. Hyperparameters & Configuration

| Parameter | Configured Value |
| :--- | :--- |
| **Total Epochs** | 25 |
| **Batch Size** | 32 |
| **Optimizer** | AdamW |
| **Initial Learning Rate** | 0.001 |
| **Weight Decay** | 0.01 |
| **LR Scheduler** | CosineAnnealingLR (T_max=25, eta_min=1e-06) |
| **Early Stopping Patience** | 5 (min_delta=0.0001) |
| **Random Seed** | 42 |
| **Execution Device** | `CPU` |

---

## 2. Dataset Information & Split Breakdown

- **Dataset Name**: LeapGestRecog (Infrared Hand Gesture Recognition)
- **Total Dataset Images**: 20,000 PNG images
- **Train Split (Subjects `00`–`06`)**: 14,000 images (70.0%)
- **Validation Split (Subjects `07`–`08`)**: 4,000 images (20.0%)
- **Test Split (Subject `09`)**: 2,000 images (10.0%) *(Unused during Phase 3.3 per protocol)*
- **Gesture Categories**: 10 distinct classes (`01_palm` through `10_down`)
- **Input Resolution**: $1 \times 128 \times 128$ (Grayscale Infrared)

---

## 3. Model Architecture & Parameter Summary

- **Total Trainable Parameters**: `422,506`
- **FP32 Model Checkpoint Memory Size**: `1.612 MB`
- **Total Layer FLOPs**: `466,422,548` ($466.42\text{ MFLOPs}$)
- **Total Layer MACs**: `233,211,274` ($233.21\text{ MMACs}$)

---

## 4. Key Training Performance Metrics Summary

| Metric | Value |
| :--- | :--- |
| **Total Training Time** | `8716.70 seconds` |
| **Completed Epochs** | `15 / 25` |
| **Best Epoch** | `Epoch 10` |
| **Best Validation Loss** | `0.10287` |
| **Best Validation Accuracy** | `98.25%` |
| **Best Validation Macro F1** | `0.9824` |
| **Final Learning Rate** | `4.069030e-04` |
| **Early Stopping Triggered** | `True` |

---

## 5. Convergence & Generalization Observations

1. **Loss Reduction & Convergence**: Training and validation losses steadily decrease across epochs, demonstrating smooth optimization via AdamW and CosineAnnealingLR scheduling.
2. **Subject Generalization**: Validation is performed on unseen subjects (`07`–`08`), confirming strong generalization without subject leakage.
3. **Macro Metric Consistency**: High Macro F1 score (0.9824) indicates balanced classification performance across all 10 gesture classes.
4. **Checkpoint Selection**: `models/checkpoints/best_model.pth` was selected at **Epoch 10** based on peak validation Macro F1 score.

---

## 6. Generated Artifacts Verification

- Checkpoints: `models/checkpoints/best_model.pth`, `latest_model.pth`
- Metadata: `models/checkpoints/best_model.json`
- History: `outputs/training/training_history.json`, `training_history.csv`
- Curves: `outputs/training/loss_curve.png`, `accuracy_curve.png`, `precision_recall_curve.png`, `learning_rate_schedule.png`
- Summary: `outputs/training/training_report.md`
