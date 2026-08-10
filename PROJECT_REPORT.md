# GestureFlow — End-to-End Machine Learning Project Report

**Project Title**: GestureFlow: Lightweight CNN Hand Gesture Recognition System  
**Author**: Machine Learning Engineering Intern  
**Date**: August 2026  
**Status**: Completed & Submitted  

---

## 1. Introduction

Hand Gesture Recognition (HGR) represents a core capability in modern computer vision and spatial computing, enabling natural human-computer interaction (HCI) across touchless interfaces, sign language translation, and augmented/virtual reality (AR/VR).

**GestureFlow** was developed as an end-to-end Machine Learning internship project. It demonstrates complete engineering rigor across the machine learning lifecycle—starting from scientific raw dataset auditing and subject-isolated partitioning to lightweight PyTorch CNN architecture design, empirical evaluation, error analysis, and a real-time desktop webcam demonstration application.

---

## 2. Problem Statement

Building real-time computer vision models for desktop hand gesture recognition requires overcoming three main obstacles:
1. **Subject Data Leakage**: In vision benchmark datasets, random image-level splitting places images of the same human subject into both training and test sets. Models memorize background clutter and subject skin characteristics, producing over-optimistic test metrics ($\approx 99\%$) that fail in real-world deployments.
2. **Computational Constraints**: Complex deep neural networks (e.g., ResNet, EfficientNet) contain tens of millions of parameters, demanding high-end GPU hardware and causing unacceptable latency ($> 50\text{ ms}$) on standard desktop CPUs.
3. **Real-Time Prediction Instability**: Frame-by-frame inference on live video streams causes prediction flickering due to bounding box jitter, lighting changes, and transient hand movements.

---

## 3. Objectives

- **Scientific Dataset Governance**: Audit 20,000 images from the `LeapGestRecog` dataset, verify zero corruptions/duplicate leaks, and compute exact channel normalization statistics.
- **Leak-Free Partitioning**: Enforce strict subject-isolated dataset partitioning ($70\%$ train / $20\%$ val / $10\%$ test) across Subjects `00` through `09`.
- **Lightweight Model Architecture**: Design a PyTorch `GestureCNN` neural network under 500,000 parameters and $< 2\text{ MB}$ memory footprint achieving $\ge 98.0\%$ test accuracy.
- **Sub-10ms CPU Inference**: Achieve average per-image CPU inference latency $< 10.0\text{ ms}$.
- **Desktop Webcam Application**: Implement a desktop demonstration pipeline integrating OpenCV frame capture, MediaPipe hand ROI localization, PyTorch CNN prediction, and temporal prediction stabilization.

---

## 4. Dataset Overview

GestureFlow uses the public **LeapGestRecog** dataset captured via the Leap Motion infrared sensor:
- **Total Samples**: 20,000 grayscale PNG images ($640 \times 240$ resolution).
- **Subjects**: 10 distinct human subjects (`00`, `01`, `02`, `03`, `04`, `05`, `06`, `07`, `08`, `09`).
- **Classes**: 10 gesture categories (`01_palm`, `02_l`, `03_fist`, `04_fist_moved`, `05_thumb`, `06_index`, `07_ok`, `08_palm_moved`, `09_c`, `10_down`).
- **Class Balance**: Exactly 2,000 images per class (perfectly balanced dataset distribution).

---

## 5. Dataset Analysis & Audit

Prior to model development, an automated dataset verification engine (`src/dataset/validator.py`) was executed:
- **Image Corruption Scan**: 20,000 images scanned; **0 corruptions** found.
- **SHA-256 Duplicate Image Scan**: 388 duplicate image clusters identified; **0 cross-split duplicates** and **0 cross-class duplicates** detected.
- **Channel Normalization Statistics**:
  - Global Pixel Mean ($\mu$): `0.2435`
  - Global Pixel Std ($\sigma$): `0.2330`

---

## 6. Data Preprocessing & Augmentation

Images undergo a deterministic transformation pipeline (`src/dataset/transforms.py`):
1. **Resizing**: Spatial downsizing from $640 \times 240$ to $128 \times 128$ single-channel grayscale tensor.
2. **Normalisation**: Standardized via $(I - \mu) / \sigma$ using calculated dataset statistics.
3. **Training Augmentations**:
   - Random Horizontal Flip ($p=0.5$)
   - Random Affine Rotation ($\pm 10^\circ$)
   - Random Translation ($\pm 5\%$)

---

## 7. Subject-Aware Dataset Split

To enforce zero subject data leakage, dataset partitioning strictly assigns human subject IDs (`00` through `09`):

```
Dataset Split Partitioning Policy
├── Training Set   : Subjects 00, 01, 02, 03, 04, 05, 06  (14,000 images / 70.0%)
├── Validation Set : Subjects 07, 08                 (4,000 images  / 20.0%)
└── Test Set       : Subject 09                      (2,000 images  / 10.0%)
```

---

## 8. CNN Architecture Design

The custom `GestureCNN` architecture (`src/models/cnn.py`) balances feature extraction capacity with ultra-low memory footprint:

```
Layer Breakdown:
Input (1, 128, 128)
├── Block 1: Conv2d(1->32, 3x3, p=1)   -> BatchNorm2d -> ReLU -> MaxPool2d(2x2) [Out: 32, 64, 64]
├── Block 2: Conv2d(32->64, 3x3, p=1)  -> BatchNorm2d -> ReLU -> MaxPool2d(2x2) [Out: 64, 32, 32]
├── Block 3: Conv2d(64->128, 3x3, p=1) -> BatchNorm2d -> ReLU -> MaxPool2d(2x2) [Out: 128, 16, 16]
├── Block 4: Conv2d(128->256, 3x3, p=1)-> BatchNorm2d -> ReLU -> MaxPool2d(2x2) [Out: 256, 8, 8]
├── Global Average Pooling: AdaptiveAvgPool2d((1, 1))                      [Out: 256, 1, 1]
├── Dense FC 1: Linear(256 -> 128)     -> ReLU -> Dropout(p=0.4)           [Out: 128]
└── Dense FC 2: Linear(128 -> 10)                                          [Out: 10]
```

### Key Architectural Rationale (ADR-0009):
By replacing traditional flattened fully-connected layers ($256 \times 8 \times 8 = 16,384$ features) with **Global Average Pooling (GAP)**, trainable parameters were reduced from $\approx 2.5\text{M}$ down to **422,506 parameters** ($1.625\text{ MB}$ memory footprint), preventing spatial overfitting and accelerating CPU inference.

---

## 9. Training Methodology

- **Optimizer**: AdamW (`lr=0.001`, `weight_decay=0.01`).
- **Loss Function**: Cross-Entropy Loss.
- **LR Scheduler**: CosineAnnealingLR (`T_max=25`, `eta_min=1e-6`).
- **Early Stopping**: Patience 5 epochs, `min_delta=0.0001`.
- **Training Progression**: Trained for 15 epochs; early stopping triggered at Epoch 15 due to validation loss stabilization (`0.10287`). Best validation accuracy reached **98.25%** at Epoch 10.

---

## 10. Model Evaluation

Evaluating `best_model.pth` on the held-out Subject `09` test split (2,000 images):

| Metric | Target | Verified Score |
| :--- | :---: | :---: |
| **Test Accuracy** | $\ge 98.0\%$ | **100.00%** |
| **Macro F1-Score** | $\ge 0.980$ | **1.0000** |
| **Macro Precision** | $\ge 0.980$ | **1.0000** |
| **Macro Recall** | $\ge 0.980$ | **1.0000** |
| **Test Loss** | — | **0.0088** |
| **Avg CPU Inference Latency** | $< 20.0\text{ ms}$ | **3.98 ms** |

---

## 11. Error Analysis

- **Misclassifications**: 0 samples misclassified out of 2,000 test set images.
- **Confidence Distribution**: Average prediction confidence across test samples was **99.17%**.
- **Domain Shift Risk**: While offline evaluation on Subject `09` achieved 100% accuracy, real-world deployment faces domain shift due to infrared vs RGB camera sensor differences.

---

## 12. Real-Time Desktop Webcam Inference

The live webcam application (`src/inference/demo.py`) implements a multi-stage real-time pipeline:

```
[Webcam Stream] ➔ [MediaPipe Hand Detector] ➔ [Hand ROI Preprocessor] ➔ [GestureCNN Model] ➔ [Prediction Stabilizer] ➔ [OpenCV Overlay HUD]
```

### Features:
- **Preprocessing Modes**: Grayscale, Global Histogram Equalization, and CLAHE.
- **Prediction Stabilizer**: 5-frame temporal sliding window majority voting with confidence gating ($\ge 70\%$).
- **Developer Debug Mode**: Real-time rendering of cropped hand ROI, normalized 2D tensor heatmap, and Top-3 probability bars.

---

## 13. Summary of Results

```
GestureFlow Benchmark Summary:
- CNN Architecture    : GestureCNN (422,506 parameters, 1.625 MB FP32 footprint)
- Subject Isolation   : Train (00-06), Val (07-08), Test (09)
- Best Val Accuracy   : 98.25% (Epoch 10)
- Test Set Accuracy   : 100.00% (2,000 images, Subject 09)
- CPU Inference Time  : 3.98 ms / image (~251 FPS capability)
- Real-Time Webcam HUD: ~30-60 FPS live rendering
- Unit Test Coverage  : 60/60 unit tests passing (100%)
```

---

## 14. Key Engineering Challenges & Solutions

1. **Overfitting Avoidance**: Solved via subject-isolated dataset splitting, Batch Normalization, Dropout ($p=0.4$), and Global Average Pooling.
2. **Prediction Flicker in Live Video**: Solved by engineering `PredictionStabilizer` using majority voting and confidence thresholds.
3. **Headless & Cross-Platform Testing**: Solved by setting non-interactive matplotlib backends (`Agg`) and adding a synthetic `--mock` mode to the webcam demo.

---

## 15. Limitations

1. **Infrared vs RGB Sensor Shift**: LeapGestRecog contains near-infrared imagery, while standard webcams capture visible-light RGB imagery.
2. **Lighting & Background Sensitivity**: Extreme low-light or cluttered skin-toned backgrounds can degrade MediaPipe ROI detection.
3. **Single Hand Constraint**: Bounding box extraction targets the single largest hand ROI.

---

## 16. Future Improvements

- **RGB Transfer Learning**: Fine-tune model on real-world RGB gesture benchmarks.
- **Domain Adaptation**: Use adversarial feature adaptation to align IR and RGB feature spaces.
- **Keypoint-Based Gesture Models**: Train lightweight GNN models directly on 3D hand keypoints for sub-1ms inference.

---

## 17. Conclusion

**GestureFlow** successfully fulfills all objectives of an end-to-end machine learning project. By maintaining scientific integrity through subject-aware dataset partitioning, lightweight CNN architecture design, thorough evaluation, and robust desktop webcam software engineering, GestureFlow stands as a polished, fully documented, reproducible ML project ready for GitHub publication and internship evaluation.
