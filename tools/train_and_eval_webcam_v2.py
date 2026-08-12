import os
import sys
import json
import time
import copy
import shutil
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.cnn import GestureCNN
from src.config.inference_config import inference_config

CLASS_NAMES = [
    "01_palm", "02_l", "03_fist", "04_fist_moved", "05_thumb",
    "06_index", "07_ok", "08_palm_moved", "09_c", "10_down"
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

class WebcamV2Dataset(Dataset):
    def __init__(self, root_dir, split="train"):
        self.root_dir = Path(root_dir) / split
        self.samples = []
        for c in CLASS_NAMES:
            c_dir = self.root_dir / c
            if c_dir.exists():
                for f in c_dir.glob("*.png"):
                    self.samples.append((f, CLASS_TO_IDX[c]))
                    
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with Image.open(path) as img:
            img_gray = img.convert("L")
            arr = np.array(img_gray, dtype=np.float32) / 255.0
            norm_arr = (arr - 0.5) / 0.5
            tensor = torch.from_numpy(norm_arr).unsqueeze(0)
        return tensor, label, str(path.name)

def load_weights_safely(model, ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    return model

def train_and_eval():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("==========================================================================")
    print(" TRAINING WEBCAM-ADAPTED MODEL V2 & EVALUATION")
    print("==========================================================================")
    print(f"Device: {device}")
    
    data_dir = Path("outputs/webcam_dataset_v2")
    train_dataset = WebcamV2Dataset(data_dir, "train")
    val_dataset = WebcamV2Dataset(data_dir, "val")
    test_dataset = WebcamV2Dataset(data_dir, "test")
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)} | Test samples: {len(test_dataset)}")
    
    ckpt_adapted = Path("models/checkpoints/webcam_adapted_model.pth")
    ckpt_best = Path("models/checkpoints/best_model.pth")
    
    model = GestureCNN(num_classes=10)
    if ckpt_adapted.exists():
        print(f"Loading weights from existing adapted checkpoint: {ckpt_adapted}")
        load_weights_safely(model, ckpt_adapted, device)
    elif ckpt_best.exists():
        print(f"Loading weights from baseline checkpoint: {ckpt_best}")
        load_weights_safely(model, ckpt_best, device)
        
    model.to(device)
    
    for name, param in model.named_parameters():
        if "block1" in name or "block2" in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
            
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0003, weight_decay=1e-4)
    
    best_val_loss = float("inf")
    best_weights = copy.deepcopy(model.state_dict())
    patience = 8
    patience_counter = 0
    epochs = 30
    
    print("\nStarting fine-tuning with early stopping...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * x.size(0)
            preds = out.argmax(dim=1)
            train_correct += (preds == y).sum().item()
            
        train_loss /= len(train_dataset)
        train_acc = train_correct / len(train_dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        with torch.no_grad():
            for x, y, _ in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                val_loss += loss.item() * x.size(0)
                preds = out.argmax(dim=1)
                val_correct += (preds == y).sum().item()
                
        val_loss /= len(val_dataset)
        val_acc = val_correct / len(val_dataset)
        
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} Acc: {train_acc*100:.1f}% | Val Loss: {val_loss:.4f} Acc: {val_acc*100:.1f}%")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break
                
    new_ckpt_path = Path("models/checkpoints/webcam_adapted_model_v2.pth")
    torch.save(best_weights, new_ckpt_path)
    print(f"\nSaved fine-tuned checkpoint to: {new_ckpt_path}")
    
    # --------------------------------------------------------------------------
    # EVALUATION ON CLEAN HELD-OUT TEST SET
    # --------------------------------------------------------------------------
    print("\n==========================================================================")
    print(" COMPARATIVE EVALUATION ON CLEAN UNSEEN WEBCAM TEST SET")
    print("==========================================================================")
    
    checkpoints_to_eval = {
        "Baseline (best_model.pth)": ckpt_best,
        "Previous Adapted (webcam_adapted_model.pth)": ckpt_adapted,
        "New Adapted V2 (webcam_adapted_model_v2.pth)": new_ckpt_path
    }
    
    eval_results = {}
    
    for name, ckpt_p in checkpoints_to_eval.items():
        if not ckpt_p.exists():
            continue
        eval_model = GestureCNN(num_classes=10)
        load_weights_safely(eval_model, ckpt_p, device)
        eval_model.to(device)
        eval_model.eval()
        
        all_preds = []
        all_labels = []
        all_confs = []
        latencies = []
        high_conf_errors = 0
        
        with torch.no_grad():
            for x, y, _ in test_loader:
                x, y = x.to(device), y.to(device)
                t0 = time.time()
                out = eval_model(x)
                t1 = time.time()
                latencies.append((t1 - t0) * 1000 / x.size(0))
                
                probs = torch.softmax(out, dim=1)
                confs, preds = torch.max(probs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
                all_confs.extend(confs.cpu().numpy())
                
                for p, l, c in zip(preds.cpu().numpy(), y.cpu().numpy(), confs.cpu().numpy()):
                    if p != l and c >= 0.70:
                        high_conf_errors += 1
                        
        acc = accuracy_score(all_labels, all_preds)
        prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro", zero_division=0)
        cm = confusion_matrix(all_labels, all_preds, labels=list(range(10)))
        
        eval_results[name] = {
            "accuracy": float(acc),
            "macro_f1": float(f1),
            "macro_precision": float(prec),
            "macro_recall": float(rec),
            "mean_confidence": float(np.mean(all_confs)),
            "high_conf_errors": high_conf_errors,
            "avg_latency_ms": float(np.mean(latencies)),
            "confusion_matrix": cm.tolist()
        }
        
    print(f"\n{'Model Variant':<45} | {'Accuracy':<10} | {'Macro F1':<10} | {'Mean Conf':<10} | {'High-Conf Errors':<16}")
    print("-" * 100)
    for name, res in eval_results.items():
        print(f"{name:<45} | {res['accuracy']*100:<9.1f}% | {res['macro_f1']:<10.4f} | {res['mean_confidence']:<10.3f} | {res['high_conf_errors']:<16}")

    print("\n--------------------------------------------------------------------------")
    print(" PER-CLASS ACCURACY BREAKDOWN ON CLEAN TEST SET (NEW ADAPTED V2)")
    print("--------------------------------------------------------------------------")
    v2_res = eval_results["New Adapted V2 (webcam_adapted_model_v2.pth)"]
    v2_cm = np.array(v2_res["confusion_matrix"])
    
    print(f"{'Class':<15} | {'Correct':<8} | {'Total':<8} | {'Accuracy':<10}")
    print("-" * 50)
    for i, c in enumerate(CLASS_NAMES):
        cor = v2_cm[i, i]
        tot = v2_cm[i].sum()
        c_acc = (cor / tot * 100) if tot > 0 else 0.0
        print(f"{c:<15} | {cor:<8} | {tot:<8} | {c_acc:<9.1f}%")

    print("\n--------------------------------------------------------------------------")
    print(" CONFUSION MATRIX (NEW ADAPTED V2)")
    print("--------------------------------------------------------------------------")
    header = "    " + " ".join([f"{i:2d}" for i in range(10)])
    print(header)
    for i, row in enumerate(v2_cm):
        print(f"{i:2d}: " + " ".join([f"{val:2d}" for val in row]))

    print("\n==========================================================================")
    print(" MODEL SELECTION DECISION")
    print("==========================================================================")
    prev_acc = eval_results.get("Previous Adapted (webcam_adapted_model.pth)", {}).get("accuracy", 0.0)
    new_acc = v2_res["accuracy"]
    
    if new_acc > prev_acc:
        print(f"SELECTION PASSED: New Adapted V2 model accuracy ({new_acc*100:.1f}%) outperforms Previous Adapted model ({prev_acc*100:.1f}%).")
        print(f"Updating adapted checkpoint copy to: models/checkpoints/webcam_adapted_model.pth")
        shutil.copyfile(new_ckpt_path, ckpt_adapted)
    else:
        print(f"SELECTION REJECTED: New model did not exceed previous model test accuracy.")

    report_data = {
        "eval_results": eval_results,
        "class_names": CLASS_NAMES,
        "selected_model": "webcam_adapted_model_v2.pth" if new_acc > prev_acc else "webcam_adapted_model.pth"
    }
    with open("outputs/webcam_dataset_v2/evaluation_report.json", "w") as f:
        json.dump(report_data, f, indent=2)

if __name__ == "__main__":
    train_and_eval()
