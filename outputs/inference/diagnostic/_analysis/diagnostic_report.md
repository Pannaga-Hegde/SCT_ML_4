# Webcam Diagnostic Analysis Report

**Samples Analyzed**: 63

## Sample Distribution

| Class | Count |
|-------|-------|
| 01_palm | 9 |
| 02_l | 16 |
| 03_fist | 7 |
| 04_fist_moved | 2 |
| 05_thumb | 6 |
| 06_index | 5 |
| 07_ok | 4 |
| 08_palm_moved | 5 |
| 09_c | 9 |
| 10_down | 0 |
| **Total** | **63** |

## 1. Overall Accuracy by Preprocessing Mode

| Mode | Correct | Total | Accuracy |
|------|---------|-------|----------|
| gray | 14 | 63 | **22.2%** |
| hist_eq | 8 | 63 | **12.7%** |
| clahe | 14 | 63 | **22.2%** |

## 2. Per-Class Accuracy

| Class | gray % | hist_eq % | clahe % |
|-------|-------|-------|-------|
| 01_palm | 0/9 (0%) | 0/9 (0%) | 0/9 (0%) |
| 02_l | 11/16 (69%) | 4/16 (25%) | 6/16 (38%) |
| 03_fist | 0/7 (0%) | 0/7 (0%) | 0/7 (0%) |
| 04_fist_moved | 0/2 (0%) | 0/2 (0%) | 0/2 (0%) |
| 05_thumb | 0/6 (0%) | 0/6 (0%) | 0/6 (0%) |
| 06_index | 0/5 (0%) | 0/5 (0%) | 0/5 (0%) |
| 07_ok | 0/4 (0%) | 1/4 (25%) | 3/4 (75%) |
| 08_palm_moved | 0/5 (0%) | 0/5 (0%) | 0/5 (0%) |
| 09_c | 3/9 (33%) | 3/9 (33%) | 5/9 (56%) |
| 10_down | 0/0 (0%) | 0/0 (0%) | 0/0 (0%) |

## 3. Confusion Matrices

### GRAY Mode

| Expected \ Predicted | palm | l | fist | fist_m | thumb | index | ok | palm_m | c | down |
|---|---|---|---|---|---|---|---|---|---|---|
| **palm** | . | 6 | . | . | . | . | . | . | 3 | . |
| **l** | 1 | **11** | . | . | . | . | . | . | 4 | . |
| **fist** | 4 | . | . | . | . | . | . | . | 3 | . |
| **fist_m** | . | 2 | . | . | . | . | . | . | . | . |
| **thumb** | . | 1 | . | . | . | . | . | . | 5 | . |
| **index** | . | 2 | . | . | . | . | . | . | 3 | . |
| **ok** | 1 | 1 | . | . | . | . | . | . | 2 | . |
| **palm_m** | 1 | 1 | . | . | . | . | . | . | 3 | . |
| **c** | 1 | 4 | . | . | . | . | . | 1 | **3** | . |
| **down** | . | . | . | . | . | . | . | . | . | . |

### HIST_EQ Mode

| Expected \ Predicted | palm | l | fist | fist_m | thumb | index | ok | palm_m | c | down |
|---|---|---|---|---|---|---|---|---|---|---|
| **palm** | . | 3 | . | . | . | . | . | . | 6 | . |
| **l** | 3 | **4** | . | . | . | . | 1 | . | 8 | . |
| **fist** | 3 | 1 | . | . | . | . | . | . | 3 | . |
| **fist_m** | 1 | 1 | . | . | . | . | . | . | . | . |
| **thumb** | 2 | 1 | . | . | . | . | . | . | 3 | . |
| **index** | 1 | . | . | . | . | . | . | . | 4 | . |
| **ok** | . | . | . | . | . | . | **1** | . | 3 | . |
| **palm_m** | 1 | . | . | . | . | . | 1 | . | 3 | . |
| **c** | 3 | 2 | . | . | . | . | 1 | . | **3** | . |
| **down** | . | . | . | . | . | . | . | . | . | . |

### CLAHE Mode

| Expected \ Predicted | palm | l | fist | fist_m | thumb | index | ok | palm_m | c | down |
|---|---|---|---|---|---|---|---|---|---|---|
| **palm** | . | 5 | . | . | . | . | . | . | 4 | . |
| **l** | . | **6** | . | . | . | . | 7 | . | 3 | . |
| **fist** | . | 1 | . | . | . | . | . | . | 6 | . |
| **fist_m** | . | . | . | . | . | . | 2 | . | . | . |
| **thumb** | . | . | . | . | . | . | 2 | . | 4 | . |
| **index** | . | . | . | . | . | . | 2 | . | 3 | . |
| **ok** | . | . | . | . | . | . | **3** | . | 1 | . |
| **palm_m** | . | . | . | . | . | . | 2 | . | 3 | . |
| **c** | . | 1 | . | . | . | . | 3 | . | **5** | . |
| **down** | . | . | . | . | . | . | . | . | . | . |

## 4. Most Common Incorrect Predictions

### GRAY

| Expected | Predicted | Count |
|----------|-----------|-------|
| 01_palm | 02_l | 6 |
| 05_thumb | 09_c | 5 |
| 02_l | 09_c | 4 |
| 03_fist | 01_palm | 4 |
| 09_c | 02_l | 4 |
| 01_palm | 09_c | 3 |
| 03_fist | 09_c | 3 |
| 06_index | 09_c | 3 |
| 08_palm_moved | 09_c | 3 |
| 04_fist_moved | 02_l | 2 |

### HIST_EQ

| Expected | Predicted | Count |
|----------|-----------|-------|
| 02_l | 09_c | 8 |
| 01_palm | 09_c | 6 |
| 06_index | 09_c | 4 |
| 01_palm | 02_l | 3 |
| 02_l | 01_palm | 3 |
| 03_fist | 01_palm | 3 |
| 03_fist | 09_c | 3 |
| 05_thumb | 09_c | 3 |
| 07_ok | 09_c | 3 |
| 08_palm_moved | 09_c | 3 |

### CLAHE

| Expected | Predicted | Count |
|----------|-----------|-------|
| 02_l | 07_ok | 7 |
| 03_fist | 09_c | 6 |
| 01_palm | 02_l | 5 |
| 01_palm | 09_c | 4 |
| 05_thumb | 09_c | 4 |
| 02_l | 09_c | 3 |
| 06_index | 09_c | 3 |
| 08_palm_moved | 09_c | 3 |
| 09_c | 07_ok | 3 |
| 04_fist_moved | 07_ok | 2 |

## 5. 01_palm Prediction Bias Analysis

| Mode | Non-palm Predicted as palm | Total Non-palm | Rate |
|------|---------------------------|----------------|------|
| gray | 8 | 54 | 14.8% |
| hist_eq | 14 | 54 | 25.9% |
| clahe | 0 | 54 | 0.0% |

**gray** breakdown: {'02_l': 1, '03_fist': 4, '07_ok': 1, '08_palm_moved': 1, '09_c': 1}

**hist_eq** breakdown: {'02_l': 3, '03_fist': 3, '04_fist_moved': 1, '05_thumb': 2, '06_index': 1, '08_palm_moved': 1, '09_c': 3}

## 6. Confidence Distribution: Correct vs Incorrect

| Mode | Correct Mean±Std | Correct Range | Incorrect Mean±Std | Incorrect Range | #Correct | #Incorrect |
|------|------------------|---------------|--------------------|-----------------|---------|-----------|
| gray | 0.834±0.166 | [0.510, 1.000] | 0.813±0.180 | [0.403, 1.000] | 14 | 49 |
| hist_eq | 0.914±0.119 | [0.651, 0.999] | 0.831±0.162 | [0.481, 1.000] | 8 | 55 |
| clahe | 0.896±0.116 | [0.561, 0.998] | 0.813±0.177 | [0.481, 1.000] | 14 | 49 |

## 7. High-Confidence Wrong Predictions (≥70%)

### GRAY (35 cases)

| Sample | Expected | Predicted | Confidence |
|--------|----------|-----------|------------|
| sample_20260812_003114_610754_08_palm_mo... | 08_palm_moved | 01_palm | 1.000 |
| sample_20260812_003045_530036_06_index... | 06_index | 09_c | 0.999 |
| sample_20260812_003044_715050_06_index... | 06_index | 09_c | 0.999 |
| sample_20260812_003359_588690_02_l... | 02_l | 09_c | 0.998 |
| sample_20260812_002957_075392_01_palm... | 01_palm | 02_l | 0.990 |
| sample_20260812_002956_749846_01_palm... | 01_palm | 02_l | 0.989 |
| sample_20260812_003010_780739_03_fist... | 03_fist | 01_palm | 0.988 |
| sample_20260812_002957_302545_01_palm... | 01_palm | 02_l | 0.987 |
| sample_20260812_003036_737113_05_thumb... | 05_thumb | 09_c | 0.985 |
| sample_20260812_003017_200600_03_fist... | 03_fist | 01_palm | 0.983 |
| sample_20260812_003004_476548_02_l... | 02_l | 09_c | 0.976 |
| sample_20260812_003347_990710_09_c... | 09_c | 02_l | 0.974 |
| sample_20260812_003035_487383_05_thumb... | 05_thumb | 09_c | 0.968 |
| sample_20260812_002947_798578_01_palm... | 01_palm | 02_l | 0.967 |
| sample_20260812_003012_675236_03_fist... | 03_fist | 09_c | 0.966 |

### HIST_EQ (42 cases)

| Sample | Expected | Predicted | Confidence |
|--------|----------|-----------|------------|
| sample_20260812_003359_588690_02_l... | 02_l | 09_c | 1.000 |
| sample_20260812_003016_219909_03_fist... | 03_fist | 09_c | 0.998 |
| sample_20260812_003037_764665_05_thumb... | 05_thumb | 09_c | 0.997 |
| sample_20260812_003111_387411_08_palm_mo... | 08_palm_moved | 09_c | 0.996 |
| sample_20260812_003044_715050_06_index... | 06_index | 09_c | 0.995 |
| sample_20260812_003114_610754_08_palm_mo... | 08_palm_moved | 01_palm | 0.994 |
| sample_20260812_003003_544970_02_l... | 02_l | 01_palm | 0.994 |
| sample_20260812_003108_713899_08_palm_mo... | 08_palm_moved | 09_c | 0.991 |
| sample_20260812_003344_823922_09_c... | 09_c | 02_l | 0.989 |
| sample_20260812_003102_130895_07_ok... | 07_ok | 09_c | 0.988 |
| sample_20260812_002957_075392_01_palm... | 01_palm | 02_l | 0.988 |
| sample_20260812_002949_068563_01_palm... | 01_palm | 09_c | 0.986 |
| sample_20260812_003048_852369_06_index... | 06_index | 01_palm | 0.981 |
| sample_20260812_002957_302545_01_palm... | 01_palm | 02_l | 0.981 |
| sample_20260812_003112_535531_08_palm_mo... | 08_palm_moved | 09_c | 0.980 |

### CLAHE (34 cases)

| Sample | Expected | Predicted | Confidence |
|--------|----------|-----------|------------|
| sample_20260812_003016_219909_03_fist... | 03_fist | 09_c | 1.000 |
| sample_20260812_003012_675236_03_fist... | 03_fist | 09_c | 1.000 |
| sample_20260812_003045_530036_06_index... | 06_index | 09_c | 0.999 |
| sample_20260812_003359_588690_02_l... | 02_l | 09_c | 0.999 |
| sample_20260812_003011_867618_03_fist... | 03_fist | 09_c | 0.999 |
| sample_20260812_003037_764665_05_thumb... | 05_thumb | 09_c | 0.998 |
| sample_20260812_003038_905781_05_thumb... | 05_thumb | 09_c | 0.998 |
| sample_20260812_003013_480696_03_fist... | 03_fist | 09_c | 0.998 |
| sample_20260812_003018_233793_03_fist... | 03_fist | 09_c | 0.997 |
| sample_20260812_003402_717997_02_l... | 02_l | 07_ok | 0.996 |
| sample_20260812_003044_715050_06_index... | 06_index | 09_c | 0.996 |
| sample_20260812_003401_646661_02_l... | 02_l | 07_ok | 0.994 |
| sample_20260812_003110_050748_08_palm_mo... | 08_palm_moved | 07_ok | 0.991 |
| sample_20260812_002949_068563_01_palm... | 01_palm | 09_c | 0.987 |
| sample_20260812_003035_487383_05_thumb... | 05_thumb | 09_c | 0.986 |

## 8. Preprocessing Mode Agreement (Same ROIs)

- **All 3 modes correct**: 5/63 (7.9%)
- **Some modes correct**: 15/63 (23.8%)
- **No mode correct**: 43/63 (68.3%)
- **Mode exclusive wins**: {'gray': 9, 'clahe': 9, 'hist_eq': 3}

## 9. ROI Dimension Analysis

| Metric | All | Correct (gray) | Incorrect (gray) |
|--------|-----|----------------|------------------|
| Count | 63 | 14 | 49 |
| Width (mean) | 1055 | 1122 | 1036 |
| Height (mean) | 505 | 561 | 489 |
| Aspect Ratio (mean) | 2.14 | 2.02 | 2.18 |
| Width range | [688, 1280] | - | - |
| Height range | [290, 706] | - | - |
| Training target ratio | 2.67 (640:240) | - | - |

## 10. Webcam vs Training Data Statistical Comparison

| Metric | Training (LeapGestRecog) | Webcam GRAY | Webcam HIST_EQ | Webcam CLAHE |
|--------|------------------------|-------------|----------------|-------------|
| N samples | 180 | 63 | 63 | 63 |
| Mean pixel intensity | 29.6 | 133.8 | 128.6 | 136.2 |
| Std of per-image means | 6.4 | 8.5 | 0.2 | 6.9 |
| Mean per-image std | 58.1 | 66.4 | 73.3 | 65.2 |
| Bhattacharyya dist (gray vs training) | — | 0.8459 | | |
| Bhattacharyya distance vs training | — | 0.8459 | 0.6655 | 0.8216 | 

> Bhattacharyya distance: 0 = identical distributions, 1 = completely different

## 11. Raw CNN vs Stabilization Errors

The diagnostic tool captures **raw CNN predictions** (no stabilizer applied).
All errors reported above are **raw model errors**, not stabilization artifacts.
The stabilizer (majority-vote + consecutive-frame filter) would only affect the live demo, not these single-frame captures.

## 12. Representative Failure Examples (GRAY mode)

### 01_palm

| Sample | Predicted | Confidence | Top-2 Probabilities |
|--------|-----------|------------|---------------------|
| sample_20260812_002945_034827_01_pa... | 09_c | 0.587 | 09_c: 0.587, 02_l: 0.391 |
| sample_20260812_002946_419115_01_pa... | 09_c | 0.603 | 09_c: 0.603, 02_l: 0.390 |
| sample_20260812_002947_798578_01_pa... | 02_l | 0.967 | 02_l: 0.967, 09_c: 0.032 |
| sample_20260812_002949_068563_01_pa... | 09_c | 0.525 | 09_c: 0.525, 02_l: 0.475 |
| sample_20260812_002950_037776_01_pa... | 02_l | 0.935 | 02_l: 0.935, 09_c: 0.061 |

### 02_l

| Sample | Predicted | Confidence | Top-2 Probabilities |
|--------|-----------|------------|---------------------|
| sample_20260812_003000_352228_02_l... | 09_c | 0.494 | 09_c: 0.494, 02_l: 0.474 |
| sample_20260812_003003_544970_02_l... | 01_palm | 0.862 | 01_palm: 0.862, 02_l: 0.120 |
| sample_20260812_003004_476548_02_l... | 09_c | 0.976 | 09_c: 0.976, 02_l: 0.016 |
| sample_20260812_003358_067994_02_l... | 09_c | 0.530 | 09_c: 0.530, 02_l: 0.456 |
| sample_20260812_003359_588690_02_l... | 09_c | 0.998 | 09_c: 0.998, 07_ok: 0.001 |

### 03_fist

| Sample | Predicted | Confidence | Top-2 Probabilities |
|--------|-----------|------------|---------------------|
| sample_20260812_003010_780739_03_fi... | 01_palm | 0.988 | 01_palm: 0.988, 06_index: 0.009 |
| sample_20260812_003011_867618_03_fi... | 09_c | 0.702 | 09_c: 0.702, 08_palm_moved: 0.181 |
| sample_20260812_003012_675236_03_fi... | 09_c | 0.966 | 09_c: 0.966, 01_palm: 0.031 |
| sample_20260812_003013_480696_03_fi... | 01_palm | 0.851 | 01_palm: 0.851, 09_c: 0.149 |
| sample_20260812_003016_219909_03_fi... | 01_palm | 0.744 | 01_palm: 0.744, 09_c: 0.255 |

### 04_fist_moved

| Sample | Predicted | Confidence | Top-2 Probabilities |
|--------|-----------|------------|---------------------|
| sample_20260812_003023_215980_04_fi... | 02_l | 0.842 | 02_l: 0.842, 09_c: 0.087 |
| sample_20260812_003029_479961_04_fi... | 02_l | 0.728 | 02_l: 0.728, 09_c: 0.148 |

### 05_thumb

| Sample | Predicted | Confidence | Top-2 Probabilities |
|--------|-----------|------------|---------------------|
| sample_20260812_003033_714335_05_th... | 02_l | 0.950 | 02_l: 0.950, 09_c: 0.047 |
| sample_20260812_003034_582736_05_th... | 09_c | 0.852 | 09_c: 0.852, 02_l: 0.102 |
| sample_20260812_003035_487383_05_th... | 09_c | 0.968 | 09_c: 0.968, 07_ok: 0.017 |
| sample_20260812_003036_737113_05_th... | 09_c | 0.985 | 09_c: 0.985, 08_palm_moved: 0.011 |
| sample_20260812_003037_764665_05_th... | 09_c | 0.952 | 09_c: 0.952, 08_palm_moved: 0.048 |

### 06_index

| Sample | Predicted | Confidence | Top-2 Probabilities |
|--------|-----------|------------|---------------------|
| sample_20260812_003043_595266_06_in... | 02_l | 0.951 | 02_l: 0.951, 09_c: 0.037 |
| sample_20260812_003044_715050_06_in... | 09_c | 0.999 | 09_c: 0.999, 07_ok: 0.001 |
| sample_20260812_003045_530036_06_in... | 09_c | 0.999 | 09_c: 0.999, 07_ok: 0.001 |
| sample_20260812_003047_466840_06_in... | 02_l | 0.640 | 02_l: 0.640, 09_c: 0.299 |
| sample_20260812_003048_852369_06_in... | 09_c | 0.919 | 09_c: 0.919, 07_ok: 0.080 |

### 07_ok

| Sample | Predicted | Confidence | Top-2 Probabilities |
|--------|-----------|------------|---------------------|
| sample_20260812_003057_315198_07_ok... | 01_palm | 0.624 | 01_palm: 0.624, 09_c: 0.373 |
| sample_20260812_003058_018415_07_ok... | 02_l | 0.676 | 02_l: 0.676, 07_ok: 0.301 |
| sample_20260812_003058_992842_07_ok... | 09_c | 0.570 | 09_c: 0.570, 02_l: 0.343 |
| sample_20260812_003102_130895_07_ok... | 09_c | 0.745 | 09_c: 0.745, 01_palm: 0.245 |

### 08_palm_moved

| Sample | Predicted | Confidence | Top-2 Probabilities |
|--------|-----------|------------|---------------------|
| sample_20260812_003108_713899_08_pa... | 09_c | 0.895 | 09_c: 0.895, 02_l: 0.074 |
| sample_20260812_003110_050748_08_pa... | 02_l | 0.422 | 02_l: 0.422, 07_ok: 0.330 |
| sample_20260812_003111_387411_08_pa... | 09_c | 0.687 | 09_c: 0.687, 02_l: 0.264 |
| sample_20260812_003112_535531_08_pa... | 09_c | 0.555 | 09_c: 0.555, 02_l: 0.388 |
| sample_20260812_003114_610754_08_pa... | 01_palm | 1.000 | 01_palm: 1.000, 09_c: 0.000 |

### 09_c

| Sample | Predicted | Confidence | Top-2 Probabilities |
|--------|-----------|------------|---------------------|
| sample_20260812_003132_445684_09_c... | 01_palm | 0.555 | 01_palm: 0.555, 09_c: 0.390 |
| sample_20260812_003339_934085_09_c... | 02_l | 0.915 | 02_l: 0.915, 09_c: 0.073 |
| sample_20260812_003343_354967_09_c... | 08_palm_moved | 0.403 | 08_palm_moved: 0.403, 09_c: 0.378 |
| sample_20260812_003344_823922_09_c... | 02_l | 0.904 | 02_l: 0.904, 01_palm: 0.078 |
| sample_20260812_003346_778663_09_c... | 02_l | 0.779 | 02_l: 0.779, 07_ok: 0.140 |

## 13. Root Cause Analysis

### Evidence Summary

1. **Best overall accuracy**: gray at 22.2%
2. **Training data mean pixel intensity**: 29.6
3. **Webcam GRAY mean pixel intensity**: 133.8 (gap: 104.2)
4. **Training data mean std**: 58.1
5. **Webcam GRAY mean std**: 66.4
6. **Bhattacharyya distance (gray)**: 0.8459
6. **Bhattacharyya distance (hist_eq)**: 0.6655
6. **Bhattacharyya distance (clahe)**: 0.8216
7. **ROI aspect ratio**: mean 2.14 vs training target 2.67
8. **No mode correct**: 43/63 samples (68.3%)

### Conclusion

The evidence strongly indicates a **distribution mismatch** between training data and webcam input:

- The LeapGestRecog dataset (infrared, dark background, mean intensity ~30) differs substantially from the webcam captures (visible light, complex background, mean intensity ~134).
- Bhattacharyya distances confirm the pixel intensity distributions are significantly different.
- The model has never seen inputs with this intensity distribution during training.
- This is a **training-inference distribution gap** — the CNN learned features specific to infrared imagery.

### Recommended Next Action

*(Based strictly on the evidence above — no changes have been made)*

1. The most impactful fix would be to **align the webcam input distribution** to match training data characteristics.
2. Investigate whether histogram equalization or CLAHE can close the gap (compare Bhattacharyya distances above).
3. If preprocessing alone is insufficient, consider **fine-tuning** the CNN on a small set of webcam-captured samples.
4. Verify the ROI cropping is not introducing excessive background — check the saved `roi_bgr.jpg` files visually.
