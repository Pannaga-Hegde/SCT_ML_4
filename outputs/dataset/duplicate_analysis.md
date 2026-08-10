# SHA-256 Duplicate Image Investigation Report — GestureFlow

## 1. Summary
- **Total Scanned Images**: 20000
- **Total Duplicate Clusters**: 388
- **Total Duplicate Image Files**: 780
- **Duplicate Percentage**: 3.90%

## 2. Duplicate Cluster Categorization Breakdown

| Category | Description | Cluster Count | Risk Assessment |
| :--- | :--- | :---: | :--- |
| **Category A** | Same Subject, Same Gesture (Sequential frame capture) | **246** | SAFE (Expected recording artifact) |
| **Category B** | Cross-Subject Duplicate (Appears across subject IDs) | **142** | WARNING |
| **Category C** | Cross-Class Duplicate (Appears in different gesture classes) | **0** | SAFE |
| **Category D** | Cross-Split Duplicate (Appears in Train AND Val/Test) | **0** | SAFE (Zero Leakage) |

## 3. Representative Top Duplicate Clusters

### Cluster #1 (Hash: `e68abbdc5872...`, Count: 4 files)
- **Category**: Category B (Cross-Subject)
- **Subject**: 02
- **Gesture Class**: 08_palm_moved
- **Split**: train
- **Sample Paths**:
  - `02/08_palm_moved/frame_02_08_0085.png`
  - `02/08_palm_moved/frame_02_08_0087.png`
  - `05/08_palm_moved/frame_05_08_0029.png`

### Cluster #2 (Hash: `496581b947b1...`, Count: 4 files)
- **Category**: Category B (Cross-Subject)
- **Subject**: 02
- **Gesture Class**: 08_palm_moved
- **Split**: train
- **Sample Paths**:
  - `02/08_palm_moved/frame_02_08_0086.png`
  - `02/08_palm_moved/frame_02_08_0088.png`
  - `05/08_palm_moved/frame_05_08_0030.png`

### Cluster #3 (Hash: `8ce3a26edbb9...`, Count: 2 files)
- **Category**: Category A (Sequential Duplicate)
- **Subject**: 00
- **Gesture Class**: 01_palm
- **Split**: train
- **Sample Paths**:
  - `00/01_palm/frame_00_01_0002.png`
  - `00/01_palm/frame_00_01_0003.png`

### Cluster #4 (Hash: `ea326e4a38fa...`, Count: 2 files)
- **Category**: Category A (Sequential Duplicate)
- **Subject**: 00
- **Gesture Class**: 01_palm
- **Split**: train
- **Sample Paths**:
  - `00/01_palm/frame_00_01_0004.png`
  - `00/01_palm/frame_00_01_0005.png`

### Cluster #5 (Hash: `18ea552cb87b...`, Count: 2 files)
- **Category**: Category A (Sequential Duplicate)
- **Subject**: 00
- **Gesture Class**: 01_palm
- **Split**: train
- **Sample Paths**:
  - `00/01_palm/frame_00_01_0006.png`
  - `00/01_palm/frame_00_01_0007.png`

## 4. Conclusion & Scientific Finding
> **Scientific Finding**: Non-Category-A duplicates detected requiring review.
