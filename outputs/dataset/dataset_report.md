# Dataset Audit & Preprocessing Report — GestureFlow

## 1. Executive Summary
- **Dataset Name**: LeapGestRecog
- **Total Images**: 20000
- **Total Subjects**: 10 (`00` to `09`)
- **Total Classes**: 10
- **Dataset Size**: 1091.80 MB
- **Audit Status**: FAILED

## 2. Subject-Aware Partitioning Summary
- **Train Split (70%)**: Subjects `['00', '01', '02', '03', '04', '05', '06']` (14000 images)
- **Val Split (15%)**: Subjects `['07', '08']` (4000 images)
- **Test Split (15%)**: Subjects `['09']` (2000 images)
- **Subject Leakage**: Verified ZERO subject overlap between splits.

## 3. Generated Artifacts Index
- `dataset_statistics.json`
- `dataset_statistics.csv`
- `dataset_metadata.json`
- `normalization.json`
- `classes.json`
- `dataset_split.json`
- `validation_report.md`
- `class_distribution.png`
- `subject_distribution.png`
- `resolution_distribution.png`
- `sample_grid.png`
