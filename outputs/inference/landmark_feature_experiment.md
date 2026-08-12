# Landmark Feature Engineering Controlled Experiment Report

**Selected Candidate Model**: `D: Random Forest (86-feat)`

## 1. Summary Comparison Table (Validation & Held-Out Test)

| Experiment Variant | Val Acc | Val F1 | Test Acc | Test F1 | Active Classes | Latency |
|---|---|---|---|---|---|---|
| **A: Current 63-feature RBF SVM** | 20.0% | 0.3350 | **36.8%** | 0.2681 | 6 | 0.05 ms |
| **B: 63 coords + Engineered (86-feat) SVM** | 28.9% | 0.3115 | **36.8%** | 0.2446 | 6 | 0.03 ms |
| **C: Engineered features only (23-feat) SVM** | 28.9% | 0.3387 | **40.8%** | 0.3189 | 7 | 0.02 ms |
| **D: Random Forest (86-feat)** | 31.1% | 0.2810 | **32.9%** | 0.2898 | 7 | 0.31 ms |
| **E: Tuned RBF SVM (86-feat)** | 28.9% | 0.3743 | **36.8%** | 0.2870 | 7 | 0.02 ms |
| **F: Lightweight MLP (86-feat)** | 13.3% | 0.1254 | **30.3%** | 0.1929 | 7 | 0.02 ms |
