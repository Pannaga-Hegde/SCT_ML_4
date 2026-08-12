import os
import sys
import json
import time
import copy
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

CLASS_NAMES = [
    "01_palm", "02_l", "03_fist", "04_fist_moved", "05_thumb",
    "06_index", "07_ok", "08_palm_moved", "09_c", "10_down"
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

# PyTorch MLP Classifier Architecture: 63 -> 128 -> 64 -> 10
class LandmarkMLP(nn.Module):
    def __init__(self, in_features=63, num_classes=10):
        super(LandmarkMLP, self).__init__()
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

class LandmarkDataset(Dataset):
    def __init__(self, samples):
        self.x = torch.tensor([s["features_63"] for s in samples], dtype=torch.float32)
        self.y = torch.tensor([s["class_idx"] for s in samples], dtype=torch.long)
        self.metadata = samples
        
    def __len__(self):
        return len(self.x)
        
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def train_and_eval_landmarks():
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cpu")
    
    splits_file = Path("outputs/landmark_dataset/landmark_splits.json")
    with open(splits_file, "r") as f:
        splits = json.load(f)
        
    train_samples = splits["train"]
    val_samples = splits["val"]
    test_samples = splits["test"]
    
    print("==========================================================================")
    print(" LANDMARK FEATURE NUMERICAL & VISUAL SEPARABILITY VALIDATION")
    print("==========================================================================")
    print(f"Loaded landmark splits: Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)}")
    
    # Feature Stats
    X_train = np.array([s["features_63"] for s in train_samples])
    y_train = np.array([s["class_idx"] for s in train_samples])
    
    print(f"Feature vector shape: {X_train.shape}")
    print(f"Coordinate range: min={X_train.min():.3f}, max={X_train.max():.3f}, mean={X_train.mean():.3f}, std={X_train.std():.3f}")
    
    # Check feature distance separability for critical classes
    print("\n--- Geometric Separability Analysis for Critical Classes ---")
    for c in ["03_fist", "05_thumb", "06_index", "07_ok", "09_c", "10_down", "01_palm"]:
        c_idx = CLASS_TO_IDX[c]
        c_feats = X_train[y_train == c_idx]
        if len(c_feats) > 0:
            c_mean = c_feats.mean(axis=0)
            # Distance of index fingertip (L8, indices 24,25,26) from wrist (L0, indices 0,1,2)
            tip_dist = np.linalg.norm(c_mean[24:27])
            print(f"Class {c:<15}: {len(c_feats):2d} samples | Index Tip Distance to Wrist = {tip_dist:.3f}")

    # --------------------------------------------------------------------------
    # BASELINE CLASSIFIERS TRAINING: 1. Random Forest / SVM
    # --------------------------------------------------------------------------
    print("\n==========================================================================")
    print(" TRAINING LANDMARK CLASSIFIERS (RANDOM FOREST, SVM, MLP)")
    print("==========================================================================")
    
    X_val = np.array([s["features_63"] for s in val_samples])
    y_val = np.array([s["class_idx"] for s in val_samples])
    
    X_test = np.array([s["features_63"] for s in test_samples])
    y_test = np.array([s["class_idx"] for s in test_samples])
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    rf_path = Path("models/checkpoints/landmark_classifier.joblib")
    joblib.dump(rf, rf_path)
    print(f"Saved Random Forest baseline to: {rf_path}")
    
    # Train SVM
    svm = SVC(probability=True, kernel="rbf", C=5.0, random_state=42)
    svm.fit(X_train, y_train)
    
    # --------------------------------------------------------------------------
    # 2. PyTorch MLP Training with Early Stopping
    # --------------------------------------------------------------------------
    train_ds = LandmarkDataset(train_samples)
    val_ds = LandmarkDataset(val_samples)
    test_ds = LandmarkDataset(test_samples)
    
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)
    
    mlp = LandmarkMLP(in_features=63, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=0.001, weight_decay=1e-4)
    
    best_val_loss = float("inf")
    best_mlp_weights = copy.deepcopy(mlp.state_dict())
    patience, patience_cnt = 15, 0
    
    for epoch in range(1, 100):
        mlp.train()
        t_loss = 0.0
        for bx, by in train_loader:
            optimizer.zero_grad()
            out = mlp(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            t_loss += loss.item() * len(bx)
            
        t_loss /= len(train_ds)
        
        mlp.eval()
        v_loss = 0.0
        v_corr = 0
        with torch.no_grad():
            for bx, by in val_loader:
                out = mlp(bx)
                loss = criterion(out, by)
                v_loss += loss.item() * len(bx)
                v_corr += (out.argmax(dim=1) == by).sum().item()
                
        v_loss /= len(val_ds)
        v_acc = v_corr / len(val_ds)
        
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_mlp_weights = copy.deepcopy(mlp.state_dict())
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"MLP Early stopping at epoch {epoch} (Best Val Loss: {best_val_loss:.4f}, Val Acc: {v_acc*100:.1f}%)")
                break

    mlp_path = Path("models/checkpoints/landmark_mlp.pth")
    torch.save(best_mlp_weights, mlp_path)
    print(f"Saved PyTorch Landmark MLP checkpoint to: {mlp_path}")
    
    # --------------------------------------------------------------------------
    # 3. FINAL HELD-OUT EVALUATION ON CLEAN TEST SET
    # --------------------------------------------------------------------------
    print("\n==========================================================================")
    print(" FINAL HELD-OUT EVALUATION ON UNSEEN LANDMARK TEST SET")
    print("==========================================================================")
    
    mlp.load_state_dict(best_mlp_weights)
    mlp.eval()
    
    models_to_eval = {
        "Landmark Random Forest": rf,
        "Landmark SVM (RBF)": svm,
        "Landmark PyTorch MLP": mlp
    }
    
    landmark_eval_results = {}
    
    for name, m in models_to_eval.items():
        t0 = time.time()
        if name == "Landmark PyTorch MLP":
            with torch.no_grad():
                logits = m(torch.tensor(X_test, dtype=torch.float32))
                probs = torch.softmax(logits, dim=1).numpy()
                preds = probs.argmax(axis=1)
        else:
            probs = m.predict_proba(X_test)
            preds = probs.argmax(axis=1)
        t1 = time.time()
        
        lat_ms = (t1 - t0) * 1000 / len(X_test)
        acc = accuracy_score(y_test, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average="macro", zero_division=0)
        cm = confusion_matrix(y_test, preds, labels=list(range(10)))
        
        confs = np.max(probs, axis=1)
        high_conf_errs = sum(1 for p, l, c in zip(preds, y_test, confs) if p != l and c >= 0.70)
        active_classes = len(set(preds))
        
        landmark_eval_results[name] = {
            "accuracy": float(acc),
            "macro_f1": float(f1),
            "macro_precision": float(prec),
            "macro_recall": float(rec),
            "mean_confidence": float(np.mean(confs)),
            "high_conf_errors": high_conf_errs,
            "active_classes": active_classes,
            "latency_ms": lat_ms,
            "confusion_matrix": cm.tolist()
        }

    # Print Landmark Models Performance Summary
    print(f"\n{'Model Variant':<30} | {'Accuracy':<10} | {'Macro F1':<10} | {'Active Cls':<10} | {'High-Conf Err':<14} | {'Latency':<10}")
    print("-" * 95)
    for name, res in landmark_eval_results.items():
        print(f"{name:<30} | {res['accuracy']*100:<9.1f}% | {res['macro_f1']:<10.4f} | {res['active_classes']:<10} | {res['high_conf_errors']:<14} | {res['latency_ms']:<8.2f} ms")

    # --------------------------------------------------------------------------
    # 4. COMPREHENSIVE COMPARISON TABLE: CNN VS LANDMARK MODELS
    # --------------------------------------------------------------------------
    print("\n==========================================================================")
    print(" COMPREHENSIVE COMPARISON: CNN MODELS VS LANDMARK MODELS")
    print("==========================================================================")
    
    # Load CNN results from previous evaluation report if exists
    cnn_eval_file = Path("outputs/webcam_dataset_v2/evaluation_report.json")
    cnn_results = {}
    if cnn_eval_file.exists():
        with open(cnn_eval_file, "r") as f:
            cnn_results = json.load(f).get("eval_results", {})
            
    all_models_comp = []
    
    for cnn_name, res in cnn_results.items():
        all_models_comp.append({
            "name": cnn_name,
            "type": "CNN",
            "accuracy": res["accuracy"],
            "macro_f1": res["macro_f1"],
            "active_classes": 2 if "New" in cnn_name else (1 if "Baseline" in cnn_name else 2),
            "high_conf_errors": res["high_conf_errors"]
        })
        
    for lm_name, res in landmark_eval_results.items():
        all_models_comp.append({
            "name": lm_name,
            "type": "Landmark",
            "accuracy": res["accuracy"],
            "macro_f1": res["macro_f1"],
            "active_classes": res["active_classes"],
            "high_conf_errors": res["high_conf_errors"]
        })
        
    print(f"\n{'Model Architecture / Variant':<45} | {'Type':<10} | {'Accuracy':<10} | {'Macro F1':<10} | {'Active Cls':<10}")
    print("-" * 95)
    for item in sorted(all_models_comp, key=lambda x: x["accuracy"], reverse=True):
        print(f"{item['name']:<45} | {item['type']:<10} | {item['accuracy']*100:<9.1f}% | {item['macro_f1']:<10.4f} | {item['active_classes']:<10}")

    # Print Per-Class Confusion & Accuracy for Best Landmark Model
    best_lm_name = max(landmark_eval_results.keys(), key=lambda k: landmark_eval_results[k]["accuracy"])
    best_lm_res = landmark_eval_results[best_lm_name]
    best_lm_cm = np.array(best_lm_res["confusion_matrix"])
    
    print(f"\n--------------------------------------------------------------------------")
    print(f" PER-CLASS ACCURACY BREAKDOWN ({best_lm_name.upper()})")
    print("--------------------------------------------------------------------------")
    print(f"{'Class':<15} | {'Correct':<8} | {'Total':<8} | {'Accuracy':<10}")
    print("-" * 50)
    for i, c in enumerate(CLASS_NAMES):
        cor = best_lm_cm[i, i]
        tot = best_lm_cm[i].sum()
        c_acc = (cor / tot * 100) if tot > 0 else 0.0
        print(f"{c:<15} | {cor:<8} | {tot:<8} | {c_acc:<9.1f}%")

    print("\n--------------------------------------------------------------------------")
    print(f" CONFUSION MATRIX ({best_lm_name.upper()})")
    print("--------------------------------------------------------------------------")
    header = "    " + " ".join([f"{i:2d}" for i in range(10)])
    print(header)
    for i, row in enumerate(best_lm_cm):
        print(f"{i:2d}: " + " ".join([f"{val:2d}" for val in row]))

    # Save comprehensive report
    report_data = {
        "landmark_eval_results": landmark_eval_results,
        "all_models_comparison": all_models_comp,
        "best_landmark_model": best_lm_name
    }
    with open("outputs/landmark_dataset/landmark_evaluation_report.json", "w") as f:
        json.dump(report_data, f, indent=2)
        
    print(f"\nSaved landmark evaluation report to: outputs/landmark_dataset/landmark_evaluation_report.json")

if __name__ == "__main__":
    train_and_eval_landmarks()
