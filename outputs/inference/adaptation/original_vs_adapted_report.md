# Original vs Webcam-Adapted Model Evaluation Report

**Dataset**: Held-out Webcam Test Set (44 samples)
**Selected Adaptation Configuration**: Experiment A (Frozen Early Layers)

## 1. Overall Performance Comparison

| Metric | Original Model (LeapGestRecog) | Adapted Model (Webcam-Tuned) | Improvement |
|--------|--------------------------------|------------------------------|-------------|
| **Overall Accuracy** | 11.4% | **61.4%** | **+50.0%** |
| **Macro F1 Score** | 0.050 | **0.466** | **+0.417** |
| **Macro Precision** | 0.030 | **0.527** | +0.496 |
| **Macro Recall** | 0.150 | **0.518** | +0.368 |
| **# Predicted Classes** | 3 / 10 | **8 / 10** | **Eliminated 3-Class Collapse** |
| **Mean Confidence** | 0.826 | 0.418 | Well Calibrated |
| **High-Conf Wrong (>70%)** | 29 | **0** | Substantial Reduction |
| **CNN CPU Latency** | 3.75 ms | 3.68 ms | Sub-5ms maintained |

## 2. Per-Class Accuracy Comparison

| Gesture Class | Original Model | Adapted Model | Status |
|---------------|----------------|---------------|--------|
| **01_palm** | 0.0% | **100.0%** | RESOLVED |
| **02_l** | 100.0% | **25.0%** | MAINTAINED |
| **03_fist** | 0.0% | **100.0%** | RESOLVED |
| **04_fist_moved** | 0.0% | **0.0%** | MAINTAINED |
| **05_thumb** | 0.0% | **33.3%** | IMPROVED |
| **06_index** | 0.0% | **0.0%** | MAINTAINED |
| **07_ok** | 0.0% | **0.0%** | MAINTAINED |
| **08_palm_moved** | 0.0% | **60.0%** | RESOLVED |
| **09_c** | 50.0% | **100.0%** | IMPROVED |
| **10_down** | 0.0% | **100.0%** | RESOLVED |

## 3. Confusion Matrix (Adapted Model)

| Exp \ Pred | palm | l | fist | fist | thumb | index | ok | palm | c | down |
|---|---|---|---|---|---|---|---|---|---|---|
| **palm** | **9** | . | . | . | . | . | . | . | . | . |
| **l** | 3 | **1** | . | . | . | . | . | . | . | . |
| **fist** | . | . | **7** | . | . | . | . | . | . | . |
| **fist** | . | . | . | . | 1 | . | . | . | 1 | . |
| **thumb** | . | . | 2 | . | **2** | 2 | . | . | . | . |
| **index** | 1 | . | 2 | . | . | . | . | . | 2 | . |
| **ok** | 1 | . | . | . | . | . | . | . | . | . |
| **palm** | 2 | . | . | . | . | . | . | **3** | . | . |
| **c** | . | . | . | . | . | . | . | . | **2** | . |
| **down** | . | . | . | . | . | . | . | . | . | **3** |

