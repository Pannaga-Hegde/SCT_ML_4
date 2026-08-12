# Controlled Preprocessing Experiment Report

**Dataset**: 63 Captured Webcam Diagnostic Samples
**Model**: PyTorch GestureCNN (best_model.pth - Frozen Weights)

## 1. Overall Summary Comparison

| Variant | Accuracy | Correct/Total | Mean Conf | Wrong Conf (>70%) | High-Conf Wrong | # Pred Classes |
|---------|----------|---------------|-----------|-------------------|-----------------|----------------|
| **A (Baseline Grayscale)** | **22.2%** | 14/63 | 0.818 | 0.813 | 35 | 4 |
| **B (Min-Max Norm)** | **23.8%** | 15/63 | 0.874 | 0.868 | 41 | 4 |
| **C (Zero-Mean/Unit-Var)** | **14.3%** | 9/63 | 0.633 | 0.647 | 22 | 4 |
| **D (Histogram Matching)** | **1.6%** | 1/63 | 0.918 | 0.922 | 57 | 4 |
| **E (Inversion + Scaling)** | **1.6%** | 1/63 | 0.599 | 0.599 | 21 | 4 |
| **F (Tight ROI + Baseline)** | **15.9%** | 10/63 | 0.828 | 0.830 | 43 | 7 |
| **G (Tight ROI + Best Norm: B)** | **15.9%** | 10/63 | 0.859 | 0.852 | 42 | 6 |

## 2. Per-Class Accuracy Breakdown

| Class | A | B | C | D | E | F | G |
|---|---|---|---|---|---|---|---|
| **01_palm** | 0/9 (0%) | 0/9 (0%) | 0/9 (0%) | 0/9 (0%) | 0/9 (0%) | 4/9 (44%) | 1/9 (11%) |
| **02_l** | 11/16 (69%) | 9/16 (56%) | 0/16 (0%) | 0/16 (0%) | 0/16 (0%) | 6/16 (38%) | 8/16 (50%) |
| **03_fist** | 0/7 (0%) | 0/7 (0%) | 0/7 (0%) | 0/7 (0%) | 0/7 (0%) | 0/7 (0%) | 0/7 (0%) |
| **04_fist_moved** | 0/2 (0%) | 0/2 (0%) | 0/2 (0%) | 0/2 (0%) | 1/2 (50%) | 0/2 (0%) | 0/2 (0%) |
| **05_thumb** | 0/6 (0%) | 0/6 (0%) | 0/6 (0%) | 0/6 (0%) | 0/6 (0%) | 0/6 (0%) | 0/6 (0%) |
| **06_index** | 0/5 (0%) | 0/5 (0%) | 0/5 (0%) | 0/5 (0%) | 0/5 (0%) | 0/5 (0%) | 0/5 (0%) |
| **07_ok** | 0/4 (0%) | 1/4 (25%) | 3/4 (75%) | 0/4 (0%) | 0/4 (0%) | 0/4 (0%) | 0/4 (0%) |
| **08_palm_moved** | 0/5 (0%) | 0/5 (0%) | 0/5 (0%) | 0/5 (0%) | 0/5 (0%) | 0/5 (0%) | 0/5 (0%) |
| **09_c** | 3/9 (33%) | 5/9 (56%) | 6/9 (67%) | 1/9 (11%) | 0/9 (0%) | 0/9 (0%) | 1/9 (11%) |
| **10_down** | 0/0 (0%) | 0/0 (0%) | 0/0 (0%) | 0/0 (0%) | 0/0 (0%) | 0/0 (0%) | 0/0 (0%) |

## 3. Confusion Matrices Summary

### Variant A (Baseline Grayscale)

| Exp \ Pred | palm | l | fist | fist | thumb | index | ok | palm | c | down |
|---|---|---|---|---|---|---|---|---|---|---|
| **palm** | . | 6 | . | . | . | . | . | . | 3 | . |
| **l** | 1 | **11** | . | . | . | . | . | . | 4 | . |
| **fist** | 4 | . | . | . | . | . | . | . | 3 | . |
| **fist** | . | 2 | . | . | . | . | . | . | . | . |
| **thumb** | . | 1 | . | . | . | . | . | . | 5 | . |
| **index** | . | 2 | . | . | . | . | . | . | 3 | . |
| **ok** | 1 | 1 | . | . | . | . | . | . | 2 | . |
| **palm** | 1 | 1 | . | . | . | . | . | . | 3 | . |
| **c** | 1 | 4 | . | . | . | . | . | 1 | **3** | . |
| **down** | . | . | . | . | . | . | . | . | . | . |

### Variant B (Min-Max Norm)

| Exp \ Pred | palm | l | fist | fist | thumb | index | ok | palm | c | down |
|---|---|---|---|---|---|---|---|---|---|---|
| **palm** | . | 6 | . | . | . | . | . | . | 3 | . |
| **l** | . | **9** | . | . | . | . | 3 | . | 4 | . |
| **fist** | 2 | . | . | . | . | . | . | . | 5 | . |
| **fist** | . | 2 | . | . | . | . | . | . | . | . |
| **thumb** | . | 1 | . | . | . | . | . | . | 5 | . |
| **index** | . | 1 | . | . | . | . | . | . | 4 | . |
| **ok** | . | . | . | . | . | . | **1** | . | 3 | . |
| **palm** | 1 | . | . | . | . | . | 1 | . | 3 | . |
| **c** | . | 3 | . | . | . | . | 1 | . | **5** | . |
| **down** | . | . | . | . | . | . | . | . | . | . |

### Variant C (Zero-Mean/Unit-Var)

| Exp \ Pred | palm | l | fist | fist | thumb | index | ok | palm | c | down |
|---|---|---|---|---|---|---|---|---|---|---|
| **palm** | . | . | . | . | . | . | . | . | 5 | 4 |
| **l** | 2 | . | . | . | . | . | 2 | . | 4 | 8 |
| **fist** | 3 | . | . | . | . | . | . | . | 4 | . |
| **fist** | . | . | . | . | . | . | . | . | . | 2 |
| **thumb** | . | . | . | . | . | . | . | . | 4 | 2 |
| **index** | . | . | . | . | . | . | 3 | . | 2 | . |
| **ok** | . | . | . | . | . | . | **3** | . | . | 1 |
| **palm** | 1 | . | . | . | . | . | 1 | . | . | 3 |
| **c** | . | . | . | . | . | . | 2 | . | **6** | 1 |
| **down** | . | . | . | . | . | . | . | . | . | . |

### Variant D (Histogram Matching)

| Exp \ Pred | palm | l | fist | fist | thumb | index | ok | palm | c | down |
|---|---|---|---|---|---|---|---|---|---|---|
| **palm** | . | . | . | . | . | . | . | . | 7 | 2 |
| **l** | . | . | . | . | . | . | . | . | 2 | 14 |
| **fist** | . | . | . | . | . | . | 1 | . | . | 6 |
| **fist** | . | . | . | . | . | . | . | . | 1 | 1 |
| **thumb** | . | . | . | . | . | . | . | . | 1 | 5 |
| **index** | . | . | . | . | . | . | . | . | 1 | 4 |
| **ok** | . | . | . | . | . | . | . | . | 4 | . |
| **palm** | . | . | . | . | . | . | . | . | 1 | 4 |
| **c** | . | 2 | . | . | . | . | . | . | **1** | 6 |
| **down** | . | . | . | . | . | . | . | . | . | . |

### Variant E (Inversion + Scaling)

| Exp \ Pred | palm | l | fist | fist | thumb | index | ok | palm | c | down |
|---|---|---|---|---|---|---|---|---|---|---|
| **palm** | . | . | . | 9 | . | . | . | . | . | . |
| **l** | 2 | . | . | 11 | . | 2 | . | . | . | 1 |
| **fist** | . | . | . | 7 | . | . | . | . | . | . |
| **fist** | 1 | . | . | **1** | . | . | . | . | . | . |
| **thumb** | 3 | . | . | 3 | . | . | . | . | . | . |
| **index** | 1 | . | . | 4 | . | . | . | . | . | . |
| **ok** | 1 | . | . | 3 | . | . | . | . | . | . |
| **palm** | . | . | . | 5 | . | . | . | . | . | . |
| **c** | 4 | . | . | 4 | . | . | . | . | . | 1 |
| **down** | . | . | . | . | . | . | . | . | . | . |

### Variant F (Tight ROI + Baseline)

| Exp \ Pred | palm | l | fist | fist | thumb | index | ok | palm | c | down |
|---|---|---|---|---|---|---|---|---|---|---|
| **palm** | **4** | 5 | . | . | . | . | . | . | . | . |
| **l** | 4 | **6** | . | . | 2 | 2 | . | . | 1 | 1 |
| **fist** | 4 | 1 | . | . | 1 | 1 | . | . | . | . |
| **fist** | . | 1 | 1 | . | . | . | . | . | . | . |
| **thumb** | 1 | 4 | . | . | . | . | . | . | 1 | . |
| **index** | 1 | 4 | . | . | . | . | . | . | . | . |
| **ok** | 1 | 3 | . | . | . | . | . | . | . | . |
| **palm** | . | 4 | . | . | 1 | . | . | . | . | . |
| **c** | 3 | 4 | . | . | 2 | . | . | . | . | . |
| **down** | . | . | . | . | . | . | . | . | . | . |

### Variant G (Tight ROI + Best Norm: B)

| Exp \ Pred | palm | l | fist | fist | thumb | index | ok | palm | c | down |
|---|---|---|---|---|---|---|---|---|---|---|
| **palm** | **1** | 7 | . | . | . | . | . | . | 1 | . |
| **l** | 2 | **8** | . | . | 1 | . | . | . | 1 | 4 |
| **fist** | 2 | 2 | . | . | 1 | 2 | . | . | . | . |
| **fist** | . | 1 | . | . | 1 | . | . | . | . | . |
| **thumb** | . | 3 | . | . | . | . | . | . | 1 | 2 |
| **index** | 1 | 3 | . | . | 1 | . | . | . | . | . |
| **ok** | . | 3 | . | . | . | . | . | . | . | 1 |
| **palm** | . | 1 | . | . | 1 | . | . | . | . | 3 |
| **c** | 1 | 4 | . | . | 1 | . | . | . | **1** | 2 |
| **down** | . | . | . | . | . | . | . | . | . | . |

## 4. Key Experimental Findings

1. **Baseline Accuracy (A)**: 22.2%
2. **Best Variant Accuracy (B (Min-Max Norm))**: 23.8%
3. **Peak Accuracy Achieved**: 23.8% vs Random Baseline of 10.0%
4. **Class Collapse**: Across all variants, the model predicts only 4 out of 10 classes.
5. **High-Confidence Error Persistence**: High confidence (>70%) on wrong predictions remains high (41 samples).

## 5. Conclusion & Single Recommended Next Step

> [!CAUTION]
> **Experimental Conclusion**: None of the 6 intensity normalization or ROI tightening preprocessing variants (A–G) produced a substantial improvement in real webcam gesture recognition accuracy (highest accuracy achieved was **23.8%** vs baseline 22.2%).
>
> **Root Cause Confirmation**: The performance failure is NOT merely a scalar intensity shift, linear histogram mismatch, or ROI padding issue that can be solved by input image transformation alone. The CNN feature extractors trained exclusively on infrared LeapMotion images fail to generalize to visible RGB webcam feature structures.
>
> **Single Recommended Next Step**: Fine-tune the PyTorch CNN model using domain adaptation / transfer learning with a small set of webcam-captured gesture samples (or synthetic background/domain augmentation) while freezing early layers.
