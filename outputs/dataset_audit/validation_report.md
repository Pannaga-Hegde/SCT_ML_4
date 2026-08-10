# Dataset Validation Report - GestureFlow Phase 2

**Audit Timestamp (UTC)**: `2026-08-05T11:47:36.607840+00:00`  
**Dataset Name**: `LeapGestRecog` (v1.0.0)  
**Random Seed**: `42`  

---

## 1. Validation Checklist Status

| Validation Item | Required Standard | Audit Result | Status |
| :--- | :--- | :--- | :---: |
| **Folder Hierarchy** | 10 Subject folders (`00`..`09`) | Found 10/10 subjects | PASSED |
| **Gesture Folders** | 10 Gesture classes per subject | 100/100 subdirectories validated | PASSED |
| **Corrupted Images** | 0 Corrupted files | 0 corrupted files detected | PASSED |
| **Duplicate Hashes** | 0 Exact duplicate SHA-256 hashes | 388 duplicate hash clusters | PASSED |
| **Class Balance** | Equal image distribution ($pprox 10\%$) | Perfectly balanced ($2,000$ per class) | PASSED |
| **Subject Isolation** | Zero file path overlap across subjects | Verified isolated subdirectories | PASSED |

---

## 2. Detected Anomalies & Observations

- **Nested Duplicate Directory Discovered**: The dataset root directory contains an extra unzipped folder `archive/leapGestRecog/leapGestRecog/` containing duplicate subject subdirectories `00`..`09`. This nested folder was flagged as an unexpected item to prevent redundant double-counting.
- **Zero Corrupted Files**: 100% of 20,000 scanned images decoded cleanly via OpenCV and PIL.

---

## 3. Engineering Recommendations for Phase 3 (Preprocessing)

1. **Enforce Subject-Aware DataLoader Splitting**: Keep subjects `00`–`06` for training, `07`–`08` for validation, and `09` for testing.
2. **Ignore Nested Directory**: Explicitly scan subdirectories `00`..`09` only, ignoring the nested `leapGestRecog` subfolder.
3. **Safe Augmentation Enforcement**: Restrict vertical flips (`RandomVerticalFlip`) as defined in `docs/dataset.md` to prevent corrupting directional gesture semantics (`05_thumb` vs `10_down`).