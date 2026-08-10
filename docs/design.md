# Design Specification & Visualization Aesthetics — GestureFlow

## Overview

In GestureFlow, design specifications govern the visual aesthetics of generated Machine Learning artifacts, plot graphs, confusion matrix heatmaps, training loss curves, and prediction overlay graphics rendered during real-time desktop webcam inference.

---

## 1. Visual Aesthetics & Plot Color Tokens

All generated charts (using Matplotlib / Seaborn) and real-time OpenCV prediction overlays use a cohesive dark slate palette:

| Token Name | Color Value | Usage in Artifacts & Desktop Overlays |
| :--- | :--- | :--- |
| **Canvas Background** | `#080b11` | Figure canvas and background plots |
| **Plot Area Background** | `#0f172a` | Axes background color |
| **Primary Accent Cyan** | `#00f2fe` | Bounding box ROI, training accuracy curves, primary charts |
| **Secondary Accent Blue** | `#4facfe` | Validation curves, secondary metric bars |
| **Success Emerald** | `#10b981` | High confidence predictions ($\ge 85\%$), verified metric indicators |
| **Warning Amber** | `#f59e0b` | Medium confidence predictions ($60\%–84\%$), class distribution warnings |
| **Error Rose** | `#ef4444` | Low confidence predictions ($< 60\%$), misclassified sample bounding boxes, loss curves |
| **Primary Text White** | `#f8fafc` | Text overlays, class labels, plot titles |
| **Secondary Text Slate**| `#94a3b8` | Subtitles, gridlines, FPS/latency telemetry text |

---

## 2. Graphic Visualization & Desktop Overlay Standards

### A. Distribution Plots (`outputs/dataset/`)
- Dark slate figure background (`#080b11`) with subtle gridlines (`#1e293b`).
- Numeric data callouts on bar charts for clear readability.
- Monospace metric callout typography.

### B. Training Curves (`outputs/training/`)
- Side-by-side plots displaying Loss (Train vs Validation) and Accuracy (Train vs Validation) over epochs.
- Minimum line width $2.0\text{pt}$ with cyan/rose color pairing.

### C. Confusion Matrix Heatmap (`outputs/evaluation/`)
- $10 \times 10$ matrix using `Viridis` or `Blues` colormap on dark background.
- Integer count and normalized percentage text inside each cell.

### D. Real-Time Desktop Webcam Overlays (`src/inference/webcam.py`)
- High-visibility cyan (`#00f2fe`) or green (`#10b981`) bounding box around detected hand ROI.
- Top-1 predicted gesture class label displayed in bold header text.
- Live telemetry badge showing Confidence Score (%), FPS Counter, and CPU Latency (ms).
