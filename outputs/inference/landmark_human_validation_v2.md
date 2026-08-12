# MediaPipe Landmark V2 Human Validation Report

**Decision Gate Outcome**: `IMPROVED LANDMARK V2 MODEL READY (MADE PRIMARY)`

## 1. Executive Summary & Core Metrics

- **Total Valid Predictions**: 188
- **Hand Tracking Failures**: 21
- **Landmark V2 Human-Validation Accuracy**: **56.9%** (vs Landmark V1 Baseline: 54.3% vs CNN V2: 13.3%)
- **Average Model Confidence**: 0.634
- **Average MediaPipe Latency**: 66.56 ms
- **Average Classifier Latency**: 23.77 ms

## 2. Per-Gesture Accuracy Comparison (Landmark V1 vs Landmark V2)

| Gesture Class | Correct / Total | Landmark V2 Acc | Baseline V1 Acc | Improvement |
|---|---|---|---|---|
| **01_palm** | 14 / 29 | **48.3%** | 51.7% | -3.4% |
| **02_l** | 21 / 31 | **67.7%** | 81.3% | -13.6% |
| **03_fist** | 15 / 22 | **68.2%** | 47.8% | +20.4% |
| **04_fist_moved** | 1 / 20 | **5.0%** | 40.0% | -35.0% |
| **05_thumb** | 21 / 24 | **87.5%** | 56.0% | +31.5% |
| **06_index** | 12 / 13 | **92.3%** | 38.5% | +53.8% |
| **07_ok** | 8 / 15 | **53.3%** | 53.3% | +0.0% |
| **08_palm_moved** | 10 / 12 | **83.3%** | 50.0% | +33.3% |
| **09_c** | 2 / 19 | **10.5%** | 45.0% | -34.5% |
| **10_down** | 3 / 3 | **100.0%** | 66.7% | +33.3% |

## 3. Confusion Matrix (Landmark V2)

```
     0  1  2  3  4  5  6  7  8  9
 0: 14  0  0  0  0  0  0 15  0  0
 1:  0 21  0  0  3  7  0  0  0  0
 2:  0  0 15  0  7  0  0  0  0  0
 3:  0  0 18  1  1  0  0  0  0  0
 4:  0  0  3  0 21  0  0  0  0  0
 5:  0  1  0  0  0 12  0  0  0  0
 6:  0  0  0  0  2  0  8  5  0  0
 7:  2  0  0  0  0  0  0 10  0  0
 8:  0  0  3  0  7  0  2  5  2  0
 9:  0  0  0  0  0  0  0  0  0  3
```

## 4. Final Recommendation & Primary Model Selection

- **Decision**: `IMPROVED LANDMARK V2 MODEL READY (MADE PRIMARY)`
- **Rationale**: Landmark V2 achieved **56.9%** accuracy.