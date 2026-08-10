# Dataset Audit & Research Report - GestureFlow Phase 2

**Dataset Name**: `LeapGestRecog`  
**Version**: `1.0.0`  
**Audit Execution Time**: `2026-08-05T11:47:36.607840+00:00`  

---

## 1. Executive Summary

The Phase 2 scientific Dataset Audit successfully scanned **20000** valid images across **10** subjects and **10** gesture classes in `archive/leapGestRecog/`.

- **Total Valid Images**: `20,000`
- **Total Corrupted Files**: `0`
- **Exact Duplicate Clusters**: `0`
- **Image Dimensions**: $640 \times 240$ pixels (Grayscale)
- **Subject Distribution**: Exactly $2,000$ images per subject ($10.0\%$ each)
- **Class Distribution**: Exactly $2,000$ images per gesture class ($10.0\%$ each)

---

## 2. Image Resolution & Channel Statistics

| Statistic | Width (px) | Height (px) | Aspect Ratio | File Size (Bytes) |
| :--- | :---: | :---: | :---: | :---: |
| **Min** | 640.0 | 240.0 | 2.6667 | 52681.0 |
| **Max** | 640.0 | 240.0 | 2.6667 | 62979.0 |
| **Mean** | 640.0 | 240.0 | 2.67 | 57241.63 |
| **Median** | 640.0 | 240.0 | 2.67 | 57152.0 |

**Color Modes Detected**: `['L']`  
**Channel Count**: `[1]` (Single-Channel Grayscale)

---

## 3. Per-Class Image Distribution

| Class ID | Folder Name | Gesture Name | Image Count | Ratio Percentage |
| :---: | :--- | :--- | :---: | :---: |
| `0` | `01_palm` | Open Palm | 2000 | 10.0% |
| `1` | `02_l` | L-Shape | 2000 | 10.0% |
| `2` | `03_fist` | Fist | 2000 | 10.0% |
| `3` | `04_fist_moved` | Fist Moved | 2000 | 10.0% |
| `4` | `05_thumb` | Thumb Up | 2000 | 10.0% |
| `5` | `06_index` | Index Point | 2000 | 10.0% |
| `6` | `07_ok` | OK Sign | 2000 | 10.0% |
| `7` | `08_palm_moved` | Palm Moved | 2000 | 10.0% |
| `8` | `09_c` | C-Shape | 2000 | 10.0% |
| `9` | `10_down` | Hand Down | 2000 | 10.0% |

---

## 4. Per-Subject Image Distribution

| Subject ID | Image Count | Share Percentage | Status |
| :---: | :---: | :---: | :---: |
| `00` | 2000 | 10.0% | PASSED |
| `01` | 2000 | 10.0% | PASSED |
| `02` | 2000 | 10.0% | PASSED |
| `03` | 2000 | 10.0% | PASSED |
| `04` | 2000 | 10.0% | PASSED |
| `05` | 2000 | 10.0% | PASSED |
| `06` | 2000 | 10.0% | PASSED |
| `07` | 2000 | 10.0% | PASSED |
| `08` | 2000 | 10.0% | PASSED |
| `09` | 2000 | 10.0% | PASSED |

---

## 5. Visual Artifacts Generated

- `class_distribution.png`: Bar chart of image counts across all 10 gesture classes.
- `subject_distribution.png`: Bar chart of image counts across all 10 subject folders.
- `resolution_distribution.png`: Histograms of image dimensions and aspect ratios.
- `sample_grid.png`: $2 \times 5$ sample grid displaying representative gesture images.