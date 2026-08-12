# 3-Way Model Comparison Evaluation Report

**Test Set**: Held-out Real Webcam Captures (79 samples across all 10 classes)

## 1. Overall Performance Comparison

| Metric | Original Model (`best_model.pth`) | Previous Adapted Model | Newly Fine-Tuned Model (`webcam_fine_tuned_model.pth`) |
|--------|-----------------------------------|------------------------|---------------------------------------------------------|
| **Overall Accuracy** | 17.7% | 15.2% | **10.1%** |
| **Macro F1 Score** | 0.096 | 0.140 | **0.077** |
| **Macro Precision** | 0.076 | 0.143 | **0.110** |
| **Macro Recall** | 0.186 | 0.189 | **0.114** |
| **Active Predicted Classes** | 4 / 10 | 7 / 10 | **4 / 10** |
| **Mean Confidence** | 0.928 | 0.418 | **0.248** |
| **High-Conf Errors (>70%)** | 60 | 0 | **0** |
| **CPU Latency** | 4.00 ms | 4.42 ms | **5.15 ms** |

## 2. Per-Class Accuracy Comparison Across All 10 Classes

| Gesture Class | Original Model | Previous Adapted Model | Newly Fine-Tuned Model | Status |
|---------------|----------------|------------------------|-------------------------|--------|
| **01_palm** | 100.0% | 0.0% | **0.0%** | MAINTAINED |
| **02_l** | 57.1% | 0.0% | **71.4%** | IMPROVED |
| **03_fist** | 0.0% | 100.0% | **42.9%** | IMPROVED |
| **04_fist_moved** | 0.0% | 0.0% | **0.0%** | MAINTAINED |
| **05_thumb** | 0.0% | 0.0% | **0.0%** | MAINTAINED |
| **06_index** | 0.0% | 0.0% | **0.0%** | MAINTAINED |
| **07_ok** | 0.0% | 0.0% | **0.0%** | MAINTAINED |
| **08_palm_moved** | 0.0% | 60.0% | **0.0%** | MAINTAINED |
| **09_c** | 28.6% | 28.6% | **0.0%** | MAINTAINED |
| **10_down** | 0.0% | 0.0% | **0.0%** | MAINTAINED |

## 3. Confusion Matrix (Newly Fine-Tuned Model)

| Exp \ Pred | palm | l | fist | fist | thumb | index | ok | palm | c | down |
|---|---|---|---|---|---|---|---|---|---|---|
| **palm** | . | 6 | . | . | . | . | . | 2 | . | . |
| **l** | 2 | **5** | . | . | . | . | . | . | . | . |
| **fist** | . | 4 | **3** | . | . | . | . | . | . | . |
| **fist** | . | 11 | . | . | . | . | . | 2 | . | . |
| **thumb** | . | 12 | . | . | . | . | . | 8 | . | . |
| **index** | . | 4 | . | . | . | . | . | 4 | . | . |
| **ok** | . | 1 | . | . | . | . | . | . | . | . |
| **palm** | 3 | 2 | . | . | . | . | . | . | . | . |
| **c** | 2 | 5 | . | . | . | . | . | . | . | . |
| **down** | . | 2 | . | . | . | . | . | 1 | . | . |

