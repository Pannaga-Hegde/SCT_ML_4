import os
import sys
import json
import time
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference.feature_engineering import extract_rich_geometric_features

CLASS_NAMES = [
    "01_palm", "02_l", "03_fist", "04_fist_moved", "05_thumb",
    "06_index", "07_ok", "08_palm_moved", "09_c", "10_down"
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

class LandmarkMLP86(nn.Module):
    def __init__(self, in_features=86, num_classes=10):
        super(LandmarkMLP86, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        return self.net(x)

def run_controlled_experiments():
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cpu")
    
    splits_file = Path("outputs/landmark_dataset/landmark_splits.json")
    with open(splits_file, "r") as f:
        splits = json.load(f)
        
    train_samples = splits["train"]
    val_samples = splits["val"]
    test_samples = splits["test"]
    
    print("==========================================================================")
    print(" CONTROLLED FEATURE ENGINEERING & CLASSIFIER EXPERIMENTS (A TO F)")
    print("==========================================================================")
    print(f"Dataset Splits: Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)}")
    
    # Extract features for all splits
    # 63 features (raw coords normalized)
    X63_train = np.array([s["features_63"] for s in train_samples], dtype=np.float32)
    X63_val = np.array([s["features_63"] for s in val_samples], dtype=np.float32)
    X63_test = np.array([s["features_63"] for s in test_samples], dtype=np.float32)
    
    # 86 features (rich geometric)
    X86_train = np.array([extract_rich_geometric_features(np.array(s["raw_landmarks_21x3"])) for s in train_samples], dtype=np.float32)
    X86_val = np.array([extract_rich_geometric_features(np.array(s["raw_landmarks_21x3"])) for s in val_samples], dtype=np.float32)
    X86_test = np.array([extract_rich_geometric_features(np.array(s["raw_landmarks_21x3"])) for s in test_samples], dtype=np.float32)
    
    # 23 engineered features only
    X23_train = X86_train[:, 63:]
    X23_val = X86_val[:, 63:]
    X23_test = X86_test[:, 63:]
    
    y_train = np.array([s["class_idx"] for s in train_samples])
    y_val = np.array([s["class_idx"] for s in val_samples])
    y_test = np.array([s["class_idx"] for s in test_samples])
    
    experiments = {}
    
    # --------------------------------------------------------------------------
    # Experiment A: Current 63-feature RBF SVM baseline
    # --------------------------------------------------------------------------
    exp_A = SVC(probability=True, kernel="rbf", C=5.0, random_state=42)
    exp_A.fit(X63_train, y_train)
    val_pred_A = exp_A.predict(X63_val)
    val_acc_A = accuracy_score(y_val, val_pred_A)
    val_f1_A = precision_recall_fscore_support(y_val, val_pred_A, average="macro", zero_division=0)[2]
    experiments["A: Current 63-feature RBF SVM"] = {"model": exp_A, "X_tr": X63_train, "X_v": X63_val, "X_te": X63_test, "val_acc": val_acc_A, "val_f1": val_f1_A}

    # --------------------------------------------------------------------------
    # Experiment B: 63 coords + engineered geometric features (86 features) RBF SVM
    # --------------------------------------------------------------------------
    exp_B = SVC(probability=True, kernel="rbf", C=5.0, random_state=42)
    exp_B.fit(X86_train, y_train)
    val_pred_B = exp_B.predict(X86_val)
    val_acc_B = accuracy_score(y_val, val_pred_B)
    val_f1_B = precision_recall_fscore_support(y_val, val_pred_B, average="macro", zero_division=0)[2]
    experiments["B: 63 coords + Engineered (86-feat) SVM"] = {"model": exp_B, "X_tr": X86_train, "X_v": X86_val, "X_te": X86_test, "val_acc": val_acc_B, "val_f1": val_f1_B}

    # --------------------------------------------------------------------------
    # Experiment C: 23 Engineered features only RBF SVM
    # --------------------------------------------------------------------------
    exp_C = SVC(probability=True, kernel="rbf", C=5.0, random_state=42)
    exp_C.fit(X23_train, y_train)
    val_pred_C = exp_C.predict(X23_val)
    val_acc_C = accuracy_score(y_val, val_pred_C)
    val_f1_C = precision_recall_fscore_support(y_val, val_pred_C, average="macro", zero_division=0)[2]
    experiments["C: Engineered features only (23-feat) SVM"] = {"model": exp_C, "X_tr": X23_train, "X_v": X23_val, "X_te": X23_test, "val_acc": val_acc_C, "val_f1": val_f1_C}

    # --------------------------------------------------------------------------
    # Experiment D: Random Forest using 86 engineered features
    # --------------------------------------------------------------------------
    exp_D = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42)
    exp_D.fit(X86_train, y_train)
    val_pred_D = exp_D.predict(X86_val)
    val_acc_D = accuracy_score(y_val, val_pred_D)
    val_f1_D = precision_recall_fscore_support(y_val, val_pred_D, average="macro", zero_division=0)[2]
    experiments["D: Random Forest (86-feat)"] = {"model": exp_D, "X_tr": X86_train, "X_v": X86_val, "X_te": X86_test, "val_acc": val_acc_D, "val_f1": val_f1_D}

    # --------------------------------------------------------------------------
    # Experiment E: Tuned RBF SVM using 86 features (C=10.0, gamma='scale')
    # --------------------------------------------------------------------------
    exp_E = SVC(probability=True, kernel="rbf", C=10.0, gamma="scale", random_state=42)
    exp_E.fit(X86_train, y_train)
    val_pred_E = exp_E.predict(X86_val)
    val_acc_E = accuracy_score(y_val, val_pred_E)
    val_f1_E = precision_recall_fscore_support(y_val, val_pred_E, average="macro", zero_division=0)[2]
    experiments["E: Tuned RBF SVM (86-feat)"] = {"model": exp_E, "X_tr": X86_train, "X_v": X86_val, "X_te": X86_test, "val_acc": val_acc_E, "val_f1": val_f1_E}

    # --------------------------------------------------------------------------
    # Experiment F: Lightweight MLP using 86 features
    # --------------------------------------------------------------------------
    mlp = LandmarkMLP86(in_features=86, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=0.001, weight_decay=1e-4)
    
    best_val_loss = float("inf")
    best_mlp_state = copy.deepcopy(mlp.state_dict())
    
    train_ds = torch.utils.data.TensorDataset(torch.tensor(X86_train), torch.tensor(y_train))
    val_ds = torch.utils.data.TensorDataset(torch.tensor(X86_val), torch.tensor(y_val))
    tr_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
    
    for epoch in range(1, 80):
        mlp.train()
        for bx, by in tr_loader:
            optimizer.zero_grad()
            out = mlp(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            
        mlp.eval()
        v_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                out = mlp(bx)
                v_loss += criterion(out, by).item() * len(bx)
        v_loss /= len(val_ds)
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_mlp_state = copy.deepcopy(mlp.state_dict())
            
    mlp.load_state_dict(best_mlp_state)
    mlp.eval()
    with torch.no_grad():
        val_logits = mlp(torch.tensor(X86_val))
        val_pred_F = val_logits.argmax(dim=1).numpy()
    val_acc_F = accuracy_score(y_val, val_pred_F)
    val_f1_F = precision_recall_fscore_support(y_val, val_pred_F, average="macro", zero_division=0)[2]
    experiments["F: Lightweight MLP (86-feat)"] = {"model": mlp, "X_tr": X86_train, "X_v": X86_val, "X_te": X86_test, "val_acc": val_acc_F, "val_f1": val_f1_F}

    # --------------------------------------------------------------------------
    # MODEL SELECTION BASED EXCLUSIVELY ON VALIDATION PERFORMANCE
    # --------------------------------------------------------------------------
    print("\n--------------------------------------------------------------------------")
    print(" VALIDATION PERFORMANCE MODEL SELECTION SUMMARY")
    print("--------------------------------------------------------------------------")
    print(f"{'Experiment Variant':<45} | {'Val Acc':<10} | {'Val Macro F1':<12}")
    print("-" * 75)
    for exp_name, data in sorted(experiments.items(), key=lambda item: item[1]["val_acc"], reverse=True):
        print(f"{exp_name:<45} | {data['val_acc']*100:<9.1f}% | {data['val_f1']:<12.4f}")

    best_exp_name = max(experiments.keys(), key=lambda k: (experiments[k]["val_acc"], experiments[k]["val_f1"]))
    best_data = experiments[best_exp_name]
    print(f"\nSELECTED CANDIDATE MODEL BASED ON VALIDATION: {best_exp_name} (Val Acc: {best_data['val_acc']*100:.1f}%)")

    # Save selected model to models/checkpoints/landmark_classifier_v2.joblib
    v2_model_path = Path("models/checkpoints/landmark_classifier_v2.joblib")
    joblib.dump(best_data["model"], v2_model_path)
    print(f"Saved selected candidate checkpoint to: {v2_model_path}")

    # --------------------------------------------------------------------------
    # SINGLE FINAL EVALUATION ON CLEAN HELD-OUT TEST SET
    # --------------------------------------------------------------------------
    print("\n==========================================================================")
    print(" FINAL HELD-OUT TEST EVALUATION FOR ALL EXPERIMENTS")
    print("==========================================================================")
    
    test_report_rows = []
    
    for exp_name, data in experiments.items():
        m = data["model"]
        X_te = data["X_te"]
        
        t0 = time.time()
        if exp_name.startswith("F:"):
            with torch.no_grad():
                logits = m(torch.tensor(X_te))
                probs = torch.softmax(logits, dim=1).numpy()
                preds = probs.argmax(axis=1)
        else:
            probs = m.predict_proba(X_te)
            preds = probs.argmax(axis=1)
        t1 = time.time()
        
        lat_ms = (t1 - t0) * 1000.0 / len(X_te)
        acc = accuracy_score(y_test, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average="macro", zero_division=0)
        cm = confusion_matrix(y_test, preds, labels=list(range(10)))
        
        confs = np.max(probs, axis=1)
        high_conf_errs = sum(1 for p, l, c in zip(preds, y_test, confs) if p != l and c >= 0.70)
        active_cls = len(set(preds))
        
        test_report_rows.append({
            "experiment": exp_name,
            "val_acc": data["val_acc"],
            "val_f1": data["val_f1"],
            "test_acc": acc,
            "test_f1": f1,
            "test_prec": prec,
            "test_rec": rec,
            "mean_conf": float(np.mean(confs)),
            "high_conf_errors": high_conf_errs,
            "active_classes": active_cls,
            "latency_ms": lat_ms,
            "confusion_matrix": cm.tolist()
        })

    print(f"\n{'Experiment':<45} | {'Val Acc':<9} | {'Test Acc':<9} | {'Test F1':<9} | {'Active Cls':<10} | {'Latency':<8}")
    print("-" * 100)
    for r in sorted(test_report_rows, key=lambda x: x["val_acc"], reverse=True):
        print(f"{r['experiment']:<45} | {r['val_acc']*100:<8.1f}% | {r['test_acc']*100:<8.1f}% | {r['test_f1']:<9.4f} | {r['active_classes']:<10} | {r['latency_ms']:<7.2f} ms")

    # Generate landmark_feature_comparison.csv
    csv_file = Path("outputs/inference/landmark_feature_comparison.csv")
    with open(csv_file, "w") as f:
        f.write("experiment,val_acc,val_f1,test_acc,test_f1,test_precision,test_recall,mean_conf,high_conf_errors,active_classes,latency_ms\n")
        for r in test_report_rows:
            f.write(f"{r['experiment']},{r['val_acc']:.4f},{r['val_f1']:.4f},{r['test_acc']:.4f},{r['test_f1']:.4f},{r['test_prec']:.4f},{r['test_rec']:.4f},{r['mean_conf']:.4f},{r['high_conf_errors']},{r['active_classes']},{r['latency_ms']:.4f}\n")

    # Save Confusion Matrix Plot (landmark_feature_confusion_matrix.png)
    best_row = max(test_report_rows, key=lambda x: x["val_acc"])
    best_cm = np.array(best_row["confusion_matrix"])
    
    plt.figure(figsize=(9, 7))
    sns.heatmap(best_cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f"Confusion Matrix: {best_row['experiment']}")
    plt.xlabel("Predicted Gesture")
    plt.ylabel("Expected Gesture")
    plt.tight_layout()
    plt.savefig("outputs/inference/landmark_feature_confusion_matrix.png", dpi=300)
    plt.close()
    
    print(f"\nSaved CSV report to: {csv_file}")
    print(f"Saved confusion matrix plot to: outputs/inference/landmark_feature_confusion_matrix.png")

    # Save Markdown report landmark_feature_experiment.md
    md_file = Path("outputs/inference/landmark_feature_experiment.md")
    with open(md_file, "w") as f:
        f.write("# Landmark Feature Engineering Controlled Experiment Report\n\n")
        f.write(f"**Selected Candidate Model**: `{best_row['experiment']}`\n\n")
        f.write("## 1. Summary Comparison Table (Validation & Held-Out Test)\n\n")
        f.write("| Experiment Variant | Val Acc | Val F1 | Test Acc | Test F1 | Active Classes | Latency |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in test_report_rows:
            f.write(f"| **{r['experiment']}** | {r['val_acc']*100:.1f}% | {r['val_f1']:.4f} | **{r['test_acc']*100:.1f}%** | {r['test_f1']:.4f} | {r['active_classes']} | {r['latency_ms']:.2f} ms |\n")

if __name__ == "__main__":
    run_controlled_experiments()
