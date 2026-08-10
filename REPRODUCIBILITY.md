# GestureFlow — Reproducibility & Replication Protocol

This document provides exact, deterministic instructions to reproduce all dataset validation, model training, evaluation metrics, and inference demonstrations for the **GestureFlow** project.

---

## ⚙️ 1. Environment & Hardware Requirements

### System Configuration
- **Python Version**: `Python 3.10`, `3.11`, or `3.12` (Tested on `Python 3.12.6`).
- **Operating System**: Windows 10/11, Ubuntu 22.04 LTS, or macOS 13+.
- **Execution Device**: `CPU` (x86_64 architecture). GPU acceleration is optional.
- **Random Seed**: `42` (Fixed across Python, NumPy, PyTorch, and DataLoader random number generators).

---

## 📦 2. Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Pannaga-Hegde/GestureFlow.git
   cd GestureFlow
   ```

2. **Create & Activate Virtual Environment**:
   ```bash
   # Windows PowerShell
   python -m venv .venv
   .venv\Scripts\activate

   # macOS / Linux
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## 📁 3. Dataset Acquisition & Directory Structure

GestureFlow uses the **LeapGestRecog** hand gesture recognition dataset:
- **Download Link**: [Kaggle - LeapGestRecog](https://www.kaggle.com/datasets/gti-upm/leapgestrecog)
- **Destination Path**: `archive/leapGestRecog/`

### Expected Directory Layout:
```
archive/leapGestRecog/
├── 00/
│   ├── 01_palm/
│   ├── 02_l/
│   ├── ...
│   └── 10_down/
├── 01/
├── ...
└── 09/
```

---

## 🔬 4. Step-by-Step Reproduction Pipeline

### Step 4.1: Run Unit Test Suite
Verify environment integrity by running all 60 automated unit tests:
```powershell
$env:PYTHONPATH="."; pytest tests/
```
*Expected Output*: `60 passed in ~10.0s`.

### Step 4.2: Dataset Audit & Manifest Generation
Audit dataset integrity, calculate SHA-256 duplicate scans, compute dataset normalization statistics, and generate canonical dataset split definitions:
```powershell
$env:PYTHONPATH="."; python -m src.dataset.verify_integrity
```
*Generated Artifacts*:
- `outputs/dataset/manifest.csv` (20,000 canonical rows)
- `outputs/dataset/dataset_metadata.json` (`mean: 0.2435`, `std: 0.2330`)
- `outputs/dataset/dataset_split.json`

### Step 4.3: Subject-Aware Dataset Partitioning Protocol
GestureFlow enforces subject isolation across splits to prevent data leakage:

```python
train_subjects = ["00", "01", "02", "03", "04", "05", "06"]  # 14,000 images (70%)
val_subjects   = ["07", "08"]                                #  4,000 images (20%)
test_subjects  = ["09"]                                      #  2,000 images (10%)
```

### Step 4.4: Model Training Execution
Train the PyTorch `GestureCNN` model for 15 epochs:
```powershell
$env:PYTHONPATH="."; python -m src.training.train
```
*Hyperparameters*:
- **Epochs**: 25 (Early stopping triggered at Epoch 15)
- **Batch Size**: 32
- **Optimizer**: AdamW (`lr=0.001`, `weight_decay=0.01`)
- **Scheduler**: CosineAnnealingLR (`T_max=25`, `eta_min=1e-06`)
- **Early Stopping**: Patience 5, `min_delta=0.0001`
- **Best Validation Accuracy**: `98.25%` (Epoch 10)

*Generated Artifacts*:
- Checkpoint: `models/checkpoints/best_model.pth`
- History: `outputs/training/training_history.csv`
- Report: `outputs/training/training_report.md`

### Step 4.5: Model Evaluation Execution
Evaluate `models/checkpoints/best_model.pth` on held-out Subject `09` test split (2,000 images):
```powershell
$env:PYTHONPATH="."; python -m src.evaluation.evaluate
```
*Expected Metrics*:
- **Test Accuracy**: `100.00%` (`1.0000`)
- **Macro F1**: `1.0000`
- **Macro Precision**: `1.0000`
- **Macro Recall**: `1.0000`
- **CPU Latency**: `3.98 ms` per image

*Generated Artifacts*:
- `outputs/evaluation/evaluation_summary.json`
- `outputs/evaluation/confusion_matrix.png`
- `outputs/evaluation/predictions.csv`

### Step 4.6: Real-Time Desktop Webcam Inference Demo
Launch the interactive desktop webcam application:
```powershell
# Live Webcam Execution
$env:PYTHONPATH="."; python -m src.inference.demo

# Headless / Synthetic Mock Verification (Runs 50 frames without camera)
$env:PYTHONPATH="."; python -m src.inference.demo --mock --frames 50
```

---

## 🔒 5. Verification Checklist

| Reproduction Step | Executed Command | Expected Output | Status |
| :--- | :--- | :--- | :---: |
| **Unit Tests** | `pytest tests/` | `60 passed` | ✅ Passed |
| **Dataset Verification** | `python -m src.dataset.verify_integrity` | `CERTIFIED FOR PHASE 3` | ✅ Passed |
| **Model Training** | `python -m src.training.train` | Best Val Acc `98.25%` | ✅ Passed |
| **Test Evaluation** | `python -m src.evaluation.evaluate` | Test Acc `100.00%` | ✅ Passed |
| **Mock Inference Demo** | `python -m src.inference.demo --mock --frames 50` | `Session summary exported` | ✅ Passed |

---

## 📧 Support & Issue Reporting

For questions regarding environment replication or metric validation, open an issue on the project GitHub repository.
