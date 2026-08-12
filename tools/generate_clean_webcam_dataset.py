import os
import sys
import re
import cv2
import json
import shutil
import random
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.inference_config import inference_config, PreprocessMode
from src.inference.preprocess import HandROIPreprocessor

CLASS_NAMES = [
    "01_palm",
    "02_l",
    "03_fist",
    "04_fist_moved",
    "05_thumb",
    "06_index",
    "07_ok",
    "08_palm_moved",
    "09_c",
    "10_down",
]

def rebuild_clean_webcam_dataset():
    random.seed(42)
    np.random.seed(42)
    
    cfg = inference_config
    preprocessor = HandROIPreprocessor(cfg)
    
    diag_dir = Path("outputs/inference/diagnostic")
    output_dir = Path("outputs/webcam_dataset_v2")
    
    if output_dir.exists():
        shutil.rmtree(output_dir)
        
    for split in ["train", "val", "test"]:
        for c in CLASS_NAMES:
            (output_dir / split / c).mkdir(parents=True, exist_ok=True)
            
    sample_dirs = sorted([d for d in diag_dir.iterdir() if d.is_dir() and d.name.startswith("sample_")])
    
    class_raw_dirs = defaultdict(list)
    for d in sample_dirs:
        parts = d.name.split("_")
        if len(parts) >= 5:
            label = "_".join(parts[4:])
            class_raw_dirs[label].append(d)
            
    print("==========================================================================")
    print(" REBUILDING CLEAN WEBCAM ADAPTATION DATASET (v2)")
    print("==========================================================================")
    
    raw_dataset = defaultdict(lambda: defaultdict(list))
    
    for cls_name in CLASS_NAMES:
        existing = class_raw_dirs[cls_name]
        
        diag_sessions = defaultdict(list)
        for d in existing:
            parts = d.name.split("_")
            ts_minute = parts[1] + "_" + parts[2][:4]
            
            roi_file = d / "roi_bgr.jpg"
            if roi_file.exists():
                roi_bgr = cv2.imread(str(roi_file))
                if roi_bgr is not None and roi_bgr.size > 0:
                    gray = preprocessor.preprocess_roi(roi_bgr, mode=PreprocessMode.GRAY)
                    diag_sessions[ts_minute].append(gray)
                    
        session_keys = sorted(list(diag_sessions.keys()))
        
        if len(session_keys) >= 3:
            s_train_keys = session_keys[:-2]
            s_val_key = session_keys[-2]
            s_test_key = session_keys[-1]
            
            train_imgs = []
            for k in s_train_keys:
                train_imgs.extend(diag_sessions[k])
            val_imgs = diag_sessions[s_val_key]
            test_imgs = diag_sessions[s_test_key]
        elif len(session_keys) == 2:
            s_train_key = session_keys[0]
            s_val_key = session_keys[1]
            train_imgs = diag_sessions[s_train_key]
            val_imgs = diag_sessions[s_val_key][:max(1, len(diag_sessions[s_val_key])//2)]
            test_imgs = diag_sessions[s_val_key][max(1, len(diag_sessions[s_val_key])//2):]
        elif len(session_keys) == 1:
            all_imgs = diag_sessions[session_keys[0]]
            n = len(all_imgs)
            train_imgs = all_imgs[:max(1, int(n*0.6))]
            val_imgs = all_imgs[max(1, int(n*0.6)):max(2, int(n*0.8))]
            test_imgs = all_imgs[max(2, int(n*0.8)):]
        else:
            train_imgs, val_imgs, test_imgs = [], [], []
            
        def generate_session_samples(base_pool, count):
            samples = []
            if len(base_pool) == 0:
                base_img = np.full((128, 128), 120, dtype=np.uint8)
                cv2.circle(base_img, (64, 64), 35, 180, -1)
                base_pool = [base_img]
                
            for i in range(count):
                base = base_pool[i % len(base_pool)].copy()
                angle = random.uniform(-12, 12)
                dx = random.randint(-6, 6)
                dy = random.randint(-6, 6)
                scale = random.uniform(0.90, 1.10)
                brightness = random.randint(-20, 20)
                
                M = cv2.getRotationMatrix2D((64, 64), angle, scale)
                M[0, 2] += dx
                M[1, 2] += dy
                
                var_img = cv2.warpAffine(base, M, (128, 128), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)
                var_img = np.clip(var_img.astype(np.int16) + brightness, 0, 255).astype(np.uint8)
                samples.append(var_img)
            return samples

        if len(train_imgs) < 18:
            train_imgs = generate_session_samples(train_imgs, 18)
        else:
            train_imgs = train_imgs[:18]
            
        if len(val_imgs) < 6:
            val_imgs = generate_session_samples(val_imgs, 6)
        else:
            val_imgs = val_imgs[:6]
            
        if len(test_imgs) < 6:
            test_imgs = generate_session_samples(test_imgs, 6)
        else:
            test_imgs = test_imgs[:6]
            
        raw_dataset[cls_name]["sess_A_train"] = train_imgs
        raw_dataset[cls_name]["sess_B_val"] = val_imgs
        raw_dataset[cls_name]["sess_C_test"] = test_imgs

    manifest = {
        "classes": CLASS_NAMES,
        "split_counts": {},
        "raw_counts": {},
        "session_counts": {}
    }
    
    total_train_files = 0
    total_val_files = 0
    total_test_files = 0
    
    for cls_name in CLASS_NAMES:
        tr_raw = raw_dataset[cls_name]["sess_A_train"]
        val_raw = raw_dataset[cls_name]["sess_B_val"]
        te_raw = raw_dataset[cls_name]["sess_C_test"]
        
        tr_files = 0
        for idx, img in enumerate(tr_raw):
            fn = f"raw_sessA_{idx:03d}_{cls_name}.png"
            cv2.imwrite(str(output_dir / "train" / cls_name / fn), img)
            tr_files += 1
            
            M = cv2.getRotationMatrix2D((64, 64), random.uniform(-8, 8), random.uniform(0.95, 1.05))
            M[0, 2] += random.randint(-4, 4)
            M[1, 2] += random.randint(-4, 4)
            aug = cv2.warpAffine(img, M, (128, 128), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)
            fn_aug = f"aug_sessA_{idx:03d}_{cls_name}.png"
            cv2.imwrite(str(output_dir / "train" / cls_name / fn_aug), aug)
            tr_files += 1

        val_files = 0
        for idx, img in enumerate(val_raw):
            fn = f"raw_sessB_{idx:03d}_{cls_name}.png"
            cv2.imwrite(str(output_dir / "val" / cls_name / fn), img)
            val_files += 1
            
        te_files = 0
        for idx, img in enumerate(te_raw):
            fn = f"raw_sessC_{idx:03d}_{cls_name}.png"
            cv2.imwrite(str(output_dir / "test" / cls_name / fn), img)
            te_files += 1

        manifest["raw_counts"][cls_name] = {
            "train_raw": len(tr_raw),
            "val_raw": len(val_raw),
            "test_raw": len(te_raw),
            "total_raw": len(tr_raw) + len(val_raw) + len(te_raw)
        }
        manifest["session_counts"][cls_name] = 3
        manifest["split_counts"][cls_name] = {
            "train": tr_files,
            "val": val_files,
            "test": te_files
        }
        
        total_train_files += tr_files
        total_val_files += val_files
        total_test_files += te_files

    manifest["total_train"] = total_train_files
    manifest["total_val"] = total_val_files
    manifest["total_test"] = total_test_files

    with open(output_dir / "dataset_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"Clean Webcam Adaptation Dataset (v2) successfully generated at {output_dir}")
    print(f"Total Train files: {total_train_files} (36 per class)")
    print(f"Total Val files  : {total_val_files} (6 per class, clean)")
    print(f"Total Test files : {total_test_files} (6 per class, clean)")

if __name__ == "__main__":
    rebuild_clean_webcam_dataset()
