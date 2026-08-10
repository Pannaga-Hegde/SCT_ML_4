# Dataset Specification & Governance — GestureFlow

This document is the authoritative specification and governance reference for all datasets used in GestureFlow. It defines dataset structures, gesture class definitions, subject segregation rules, limitation profiles, validation checklists, splitting policies, data augmentation strategies, and benchmark evaluation protocols.

---

## 1. Baseline Dataset Overview

- **Dataset Name**: LeapGestRecog (Leap Motion Hand Gesture Recognition Dataset)
- **Source**: Kaggle / Hand Gesture Recognition Database (Leap Motion)
- **License**: Open Access for Academic & Non-Commercial Research
- **Citation**: Hand Gesture Recognition Database, Leap Motion Infrared Sensor Dataset
- **Version**: 1.0.0
- **Download / Workspace Location**: `archive/leapGestRecog/`
- **Expected Directory Structure**:
  ```
  archive/leapGestRecog/
  ├── 00/
  ├── 01/
  ├── 02/
  ├── 03/
  ├── 04/
  ├── 05/
  ├── 06/
  ├── 07/
  ├── 08/
  └── 09/
  ```

---

## 2. Dataset Hierarchy & Structure

The dataset is organized hierarchically by subject subdirectories (`00` to `09`). Each subject folder contains 10 gesture class directories (`01_palm` through `10_down`).

```
archive/leapGestRecog/
└── {subject_id}/             # 10 Subjects (00, 01, 02, 03, 04, 05, 06, 07, 08, 09)
    ├── 01_palm/              # ~200 images per subject
    ├── 02_l/                 # ~200 images per subject
    ├── 03_fist/              # ~200 images per subject
    ├── 04_fist_moved/        # ~200 images per subject
    ├── 05_thumb/             # ~200 images per subject
    ├── 06_index/             # ~200 images per subject
    ├── 07_ok/                # ~200 images per subject
    ├── 08_palm_moved/        # ~200 images per subject
    ├── 09_c/                 # ~200 images per subject
    └── 10_down/              # ~200 images per subject
```

Expected image counts:
- Images per class per subject: $\approx 200$ images.
- Images per subject: $\approx 2,000$ images across 10 classes.
- Total expected dataset size: $20,000$ images across 10 subjects.

---

## 3. Gesture Class Catalog

| Class ID | Folder Name | Human-Readable Name | Detailed Description | Research Focus |
| :---: | :--- | :--- | :--- | :--- |
| `0` | `01_palm` | Open Palm | Open hand with all five fingers fully extended upward | Static gesture |
| `1` | `02_l` | L-Shape | Thumb and Index finger extended at right angle; other fingers curled | Directional asymmetric gesture |
| `2` | `03_fist` | Fist | All fingers clenched tightly into a fist | Static gesture |
| `3` | `04_fist_moved` | Fist Moved | Fist gesture captured across motion trajectories | Trajectory variant |
| `4` | `05_thumb` | Thumb Up | Clenched fist with thumb extended vertically upward | Directional gesture (vertical) |
| `5` | `06_index` | Index Point | Index finger pointing straight upward/forward; others closed | Static gesture |
| `6` | `07_ok` | OK Sign | Circle formed by thumb and index tip; remaining three fingers extended | Fine finger interaction |
| `7` | `08_palm_moved` | Palm Moved | Open palm captured across horizontal or vertical motion | Trajectory variant |
| `8` | `09_c` | C-Shape | Curved fingers and thumb forming a 'C' shape | Curved geometry gesture |
| `9` | `10_down` | Hand Down | Palm facing downward with fingers pointing downward | Directional gesture (vertical) |

---

## 4. Subject Segregation & Generalization

### Why Subject Folders Exist
The `LeapGestRecog` dataset contains data captured from 10 distinct individuals (`00` through `09`). Each subject exhibits unique hand geometry, finger proportions, skin tone characteristics, and personal movement dynamics.

### Mandatory Subject Isolation Rule
> [!CAUTION]
> **Strict Engineering Prohibition**: Random image-level splitting across train, validation, and test sets is **permanently prohibited**.

If images from Subject `00` were split randomly between training and testing sets, the neural network could memorize Subject `00`'s specific hand morphology (skin texture, finger length, wrist angle) rather than learning generalized gesture feature representations.

### Enforced Partitioning Policy
- **Train Set (70%)**: Subjects `00`, `01`, `02`, `03`, `04`, `05`, `06` (7 Subjects, 14,000 images).
- **Validation Set (15%)**: Subjects `07`, `08` (2 Subjects, 4,000 images).
- **Test Set (15%)**: Subject `09` (1 Subject, 2,000 images).

This guarantees that model evaluation measures true out-of-subject generalization.

---

## 5. Dataset Statistics (Audit Baseline Target)

The Phase 2 scientific Dataset Audit computes the following baseline metrics, saved as a formal JSON artifact (`outputs/dataset/dataset_statistics.json`):

| Metric | Target Standard | Audit Status |
| :--- | :--- | :---: |
| **Total Image Count** | $20,000$ files | PENDING SCAN |
| **Class Distribution Balance**| Exactly $2,000$ images per class ($10.0\%$) | PENDING SCAN |
| **Subject Distribution Balance**| Exactly $2,000$ images per subject ($10.0\%$) | PENDING SCAN |
| **Original Resolution** | $640 \times 240$ pixels | PENDING SCAN |
| **Color Space / Channels** | Single-channel Grayscale (Infrared) | PENDING SCAN |
| **Corrupted / Truncated Files**| $0$ files ($0.0\%$) | PENDING SCAN |
| **Exact Duplicate Hashes** | $0$ exact duplicates | PENDING SCAN |

---

## 6. Data Augmentation Policy

### Safe Augmentations (Allowed)
- **Random Rotation**: $\pm 10^\circ$ subtle tilt to simulate slight hand tilt.
- **Random Translation**: Shift image laterally/vertically by $\pm 5\%$.
- **Brightness & Contrast Adjustments**: Range $\pm 15\%$ to improve lighting robustness.
- **Gaussian Blur / Sensor Noise**: Kernel size $3\times3$ to simulate webcam sensor noise.

### Unsafe Augmentations (Prohibited)
- **Vertical Flipping (`RandomVerticalFlip`)**: Inverts `05_thumb` (Thumb Up) into `10_down` (Hand Down), causing severe label corruption. **Strictly prohibited**.

---

## 7. Benchmark Protocol & Evaluation Metrics

To guarantee scientific rigor, model performance is evaluated against a standardized benchmark protocol.

### Primary Metrics
1. **Top-1 Accuracy**:
   $$\text{Accuracy} = \frac{\sum_{i=1}^{C} \text{TP}_i}{N_{\text{total}}}$$
2. **Macro F1-Score**: Harmonic mean of Precision and Recall averaged across all 10 gesture classes:
   $$\text{F1}_{\text{macro}} = \frac{1}{C} \sum_{i=1}^{C} 2 \cdot \frac{\text{Precision}_i \cdot \text{Recall}_i}{\text{Precision}_i + \text{Recall}_i}$$

### Secondary Metrics
- **Per-Class Precision & Recall Breakdown**.
- **Confusion Matrix ($10 \times 10$)**: Heatmap visualization highlighting inter-class confusion.
- **PyTorch / CPU Inference Latency**: Benchmarked over 1,000 execution runs ($P_{95}$, $P_{99}$ latency in milliseconds).
- **Model Checkpoint Size**: Saved FP32 PyTorch checkpoint (`models/checkpoints/best_model.pth`).
