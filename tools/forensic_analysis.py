import os
import sys
import json
import cv2
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import confusion_matrix, accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.cnn import GestureCNN
from src.config.inference_config import inference_config, PreprocessMode
from src.inference.preprocess import HandROIPreprocessor

CLASS_NAMES = [
    "01_palm", "02_l", "03_fist", "04_fist_moved", "05_thumb",
    "06_index", "07_ok", "08_palm_moved", "09_c", "10_down"
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

def load_weights_safely(model, ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    return model

def run_forensic_analysis():
    device = torch.device("cpu")
    v2_ckpt = Path("models/checkpoints/webcam_adapted_model_v2.pth")
    if not v2_ckpt.exists():
        v2_ckpt = Path("models/checkpoints/webcam_adapted_model.pth")
        
    model = GestureCNN(num_classes=10)
    load_weights_safely(model, v2_ckpt, device)
    model.eval()
    
    test_dir = Path("outputs/webcam_dataset_v2/test")
    
    print("==========================================================================")
    print(" FORENSIC DIAGNOSTIC ANALYSIS OF WEBCAM_ADAPTED_MODEL_V2")
    print("==========================================================================")
    print(f"Checkpoint evaluated: {v2_ckpt}")
    print(f"Test Directory: {test_dir}\n")
    
    per_sample_results = []
    
    # Analyze all 60 test images
    for expected_cls in CLASS_NAMES:
        c_dir = test_dir / expected_cls
        if not c_dir.exists(): continue
        
        for img_path in sorted(c_dir.glob("*.png")):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            
            # Prepare tensor
            arr = img.astype(np.float32) / 255.0
            norm_arr = (arr - 0.5) / 0.5
            tensor = torch.from_numpy(norm_arr).unsqueeze(0).unsqueeze(0).to(device)
            
            with torch.no_grad():
                out = model(tensor)
                probs = torch.softmax(out, dim=1).squeeze(0).numpy()
                pred_idx = int(np.argmax(probs))
                pred_cls = CLASS_NAMES[pred_idx]
                conf = float(probs[pred_idx])
                
            # Image visual stats
            mean_brightness = float(img.mean())
            std_contrast = float(img.std())
            blur_laplacian = float(cv2.Laplacian(img, cv2.CV_64F).var())
            fg_ratio = float((img > 80).mean())
            
            per_sample_results.append({
                "filename": img_path.name,
                "expected_class": expected_cls,
                "expected_idx": CLASS_TO_IDX[expected_cls],
                "predicted_class": pred_cls,
                "predicted_idx": pred_idx,
                "confidence": conf,
                "probabilities": {CLASS_NAMES[i]: float(probs[i]) for i in range(10)},
                "stats": {
                    "mean_brightness": mean_brightness,
                    "std_contrast": std_contrast,
                    "blur_laplacian": blur_laplacian,
                    "fg_ratio": fg_ratio
                }
            })
            
    # Verify Confusion Matrix and Per-Class Metrics
    expected_indices = [s["expected_idx"] for s in per_sample_results]
    predicted_indices = [s["predicted_idx"] for s in per_sample_results]
    
    cm = confusion_matrix(expected_indices, predicted_indices, labels=list(range(10)))
    
    print("--------------------------------------------------------------------------")
    print(" CONFUSION MATRIX INDEX VERIFICATION (60 HELD-OUT TEST SAMPLES)")
    print("--------------------------------------------------------------------------")
    print("    " + " ".join([f"{i:2d}" for i in range(10)]))
    for i, row in enumerate(cm):
        cls_name = CLASS_NAMES[i]
        row_str = " ".join([f"{val:2d}" for val in row])
        print(f"{i:2d} ({cls_name:<13}): {row_str}")
        
    print("\n--------------------------------------------------------------------------")
    print(" RESOLVING PER-CLASS METRIC & CONFUSION MATRIX CONTRADICTION")
    print("--------------------------------------------------------------------------")
    for i, cls_name in enumerate(CLASS_NAMES):
        cor = cm[i, i]
        tot = cm[i].sum()
        acc = (cor / tot * 100) if tot > 0 else 0.0
        most_freq_pred = CLASS_NAMES[np.argmax(cm[i])]
        print(f"Class {i:2d} ({cls_name:<13}): Correct={cor}/{tot} ({acc:.1f}%). Primary Prediction: {most_freq_pred}")

    # Check 09_c prediction behavior:
    print("\n--------------------------------------------------------------------------")
    print(" ANALYSIS OF ATTRACTOR CLASSES: 09_c, 07_ok, AND 10_down")
    print("--------------------------------------------------------------------------")
    pred_counts = defaultdict(int)
    for p in predicted_indices:
        pred_counts[CLASS_NAMES[p]] += 1
        
    for cls_name, count in sorted(pred_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"Predicted as {cls_name:<15}: {count:2d} / 60 samples ({count/60*100:.1f}%)")
        
    # --------------------------------------------------------------------------
    # THREE CONTROLLED INFERENCE COMPARISONS (A / B / C)
    # --------------------------------------------------------------------------
    print("\n==========================================================================")
    print(" CONTROLLED ROI PREPROCESSING INFERENCE COMPARISONS (A / B / C)")
    print("==========================================================================")
    
    # Load original diagnostic raw BGR frames from diagnostic dataset
    diag_dir = Path("outputs/inference/diagnostic")
    sample_dirs = sorted([d for d in diag_dir.iterdir() if d.is_dir() and d.name.startswith("sample_")])
    
    cfg = inference_config
    preprocessor = HandROIPreprocessor(cfg)
    
    # Test on raw BGR diagnostic frames where bounding boxes and frames exist
    variant_results = {"A": [], "B": [], "C": []} # List of (expected_idx, pred_idx)
    
    for d in sample_dirs:
        parts = d.name.split("_")
        if len(parts) < 5: continue
        exp_cls = "_".join(parts[4:])
        if exp_cls not in CLASS_TO_IDX: continue
        exp_idx = CLASS_TO_IDX[exp_cls]
        
        roi_bgr_file = d / "roi_bgr.jpg"
        frame_file = d / "frame.jpg"
        
        if not roi_bgr_file.exists(): continue
        roi_bgr = cv2.imread(str(roi_bgr_file))
        if roi_bgr is None or roi_bgr.size == 0: continue
        
        # Variant A: Current 8:3 Padded ROI
        gray_A = cv2.resize(cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY), (128, 128), interpolation=cv2.INTER_AREA)
        t_A = torch.from_numpy(((gray_A.astype(np.float32)/255.0) - 0.5)/0.5).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_A = int(model(t_A).argmax(dim=1).item())
        variant_results["A"].append((exp_idx, pred_A))
        
        # Variant B: Tight hand ROI (center square crop without 8:3 forcing)
        h, w = roi_bgr.shape[:2]
        min_dim = min(h, w)
        cy, cx = h // 2, w // 2
        tight_crop = roi_bgr[max(0, cy-min_dim//2):min(h, cy+min_dim//2), max(0, cx-min_dim//2):min(w, cx+min_dim//2)]
        if tight_crop.size > 0:
            gray_B = cv2.resize(cv2.cvtColor(tight_crop, cv2.COLOR_BGR2GRAY), (128, 128), interpolation=cv2.INTER_AREA)
            t_B = torch.from_numpy(((gray_B.astype(np.float32)/255.0) - 0.5)/0.5).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                pred_B = int(model(t_B).argmax(dim=1).item())
            variant_results["B"].append((exp_idx, pred_B))
            
        # Variant C: Tight ROI + 10% controlled padding
        pad_dim = int(min_dim * 1.10)
        padded_crop = roi_bgr[max(0, cy-pad_dim//2):min(h, cy+pad_dim//2), max(0, cx-pad_dim//2):min(w, cx+pad_dim//2)]
        if padded_crop.size > 0:
            gray_C = cv2.resize(cv2.cvtColor(padded_crop, cv2.COLOR_BGR2GRAY), (128, 128), interpolation=cv2.INTER_AREA)
            t_C = torch.from_numpy(((gray_C.astype(np.float32)/255.0) - 0.5)/0.5).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                pred_C = int(model(t_C).argmax(dim=1).item())
            variant_results["C"].append((exp_idx, pred_C))

    print(f"\nVariant A (Current 8:3 Padded ROI) Accuracy : {accuracy_score([x[0] for x in variant_results['A']], [x[1] for x in variant_results['A']]) * 100:.1f}% ({len(variant_results['A'])} samples)")
    print(f"Variant B (Tight Hand ROI) Accuracy        : {accuracy_score([x[0] for x in variant_results['B']], [x[1] for x in variant_results['B']]) * 100:.1f}% ({len(variant_results['B'])} samples)")
    print(f"Variant C (Tight ROI + 10% Pad) Accuracy   : {accuracy_score([x[0] for x in variant_results['C']], [x[1] for x in variant_results['C']]) * 100:.1f}% ({len(variant_results['C'])} samples)")

    # Save detailed JSON forensic data
    forensic_payload = {
        "per_sample_results": per_sample_results,
        "confusion_matrix": cm.tolist(),
        "variant_accuracies": {
            "A_current_8x3": float(accuracy_score([x[0] for x in variant_results['A']], [x[1] for x in variant_results['A']])),
            "B_tight_roi": float(accuracy_score([x[0] for x in variant_results['B']], [x[1] for x in variant_results['B']])),
            "C_tight_padded": float(accuracy_score([x[0] for x in variant_results['C']], [x[1] for x in variant_results['C']]))
        }
    }
    
    out_file = Path("outputs/webcam_dataset_v2/forensic_analysis_report.json")
    with open(out_file, "w") as f:
        json.dump(forensic_payload, f, indent=2)
        
    print(f"\nForensic analysis JSON saved to: {out_file}")

if __name__ == "__main__":
    run_forensic_analysis()
