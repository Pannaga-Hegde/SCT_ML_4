import os
import sys
import re
import cv2
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import mediapipe as mp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.inference_config import inference_config

CLASS_NAMES = [
    "01_palm", "02_l", "03_fist", "04_fist_moved", "05_thumb",
    "06_index", "07_ok", "08_palm_moved", "09_c", "10_down"
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

def normalize_landmarks(landmarks_21x3: np.ndarray) -> np.ndarray:
    """Normalize 21x3 landmarks.
    
    1. Wrist centering (subtract L0 Wrist)
    2. Scale normalization (divide by Euclidean distance L0 to L9 (Middle Finger MCP))
    
    Args:
        landmarks_21x3: Array of shape (21, 3) with (x, y, z).
        
    Returns:
        Flattened 63-element feature vector.
    """
    coords = landmarks_21x3.copy().astype(np.float32)
    
    # 1. Wrist translation centering
    wrist = coords[0].copy()
    coords -= wrist
    
    # 2. Scale normalization: distance between Wrist (0) and Middle MCP (9)
    middle_mcp = coords[9]
    dist = np.linalg.norm(middle_mcp)
    
    if dist < 1e-6:
        # Fallback to max distance from wrist
        dist = np.max(np.linalg.norm(coords, axis=1))
        if dist < 1e-6:
            dist = 1.0
            
    coords /= dist
    return coords.flatten()

def extract_landmark_dataset():
    mp_hands = mp.solutions.hands
    detector = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.3
    )
    
    diag_dir = Path("outputs/inference/diagnostic")
    sample_dirs = sorted([d for d in diag_dir.iterdir() if d.is_dir() and d.name.startswith("sample_")])
    
    print("==========================================================================")
    print(" 1. MEDIAPIPE LANDMARK DATASET EXTRACTION")
    print("==========================================================================")
    print(f"Diagnostic directory: {diag_dir}")
    print(f"Total raw diagnostic sample directories: {len(sample_dirs)}")
    
    session_samples = defaultdict(lambda: defaultdict(list)) # class -> session_id -> list of dicts
    rejected_count = 0
    extracted_count = 0
    
    for d in sample_dirs:
        parts = d.name.split("_")
        if len(parts) < 5: continue
        exp_cls = "_".join(parts[4:])
        if exp_cls not in CLASS_TO_IDX: continue
        
        # Session timestamp: YYYYMMDD_HHMM
        ts_session = parts[1] + "_" + parts[2][:4]
        
        frame_path = d / "frame.jpg"
        roi_path = d / "roi_bgr.jpg"
        
        # Prefer original frame, fallback to ROI
        img_file = frame_path if frame_path.exists() else roi_path
        if not img_file.exists(): continue
        
        bgr = cv2.imread(str(img_file))
        if bgr is None or bgr.size == 0: continue
        
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = detector.process(rgb)
        
        if not res.multi_hand_landmarks:
            # Fallback: try on ROI BGR if main frame failed
            if img_file == frame_path and roi_path.exists():
                bgr_roi = cv2.imread(str(roi_path))
                if bgr_roi is not None and bgr_roi.size > 0:
                    rgb_roi = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2RGB)
                    res = detector.process(rgb_roi)
                    
        if not res.multi_hand_landmarks:
            rejected_count += 1
            print(f"  [REJECTED - No Hand Detected]: {d.name}")
            continue
            
        hand_lms = res.multi_hand_landmarks[0]
        lms_21x3 = np.array([(lm.x, lm.y, lm.z) for lm in hand_lms.landmark], dtype=np.float32)
        norm_63 = normalize_landmarks(lms_21x3)
        
        extracted_count += 1
        session_samples[exp_cls][ts_session].append({
            "sample_name": d.name,
            "class_name": exp_cls,
            "class_idx": CLASS_TO_IDX[exp_cls],
            "session_id": ts_session,
            "raw_landmarks_21x3": lms_21x3.tolist(),
            "features_63": norm_63.tolist()
        })

    detector.close()
    
    print(f"\nExtracted valid landmarks: {extracted_count} samples | Rejected: {rejected_count} samples")
    
    # --------------------------------------------------------------------------
    # 2. STRICT SESSION-LEVEL SPLIT GENERATION (TRAIN / VAL / TEST)
    # --------------------------------------------------------------------------
    print("\n==========================================================================")
    print(" 2. SESSION-LEVEL LANDMARK DATASET PARTITIONING")
    print("==========================================================================")
    
    dataset_splits = {"train": [], "val": [], "test": []}
    split_counts = defaultdict(lambda: defaultdict(int))
    session_counts_per_class = defaultdict(set)
    
    for cls_name in CLASS_NAMES:
        sessions_dict = session_samples[cls_name]
        sess_keys = sorted(list(sessions_dict.keys()))
        
        for sk in sess_keys:
            session_counts_per_class[cls_name].add(sk)
            
        if len(sess_keys) >= 3:
            train_keys = sess_keys[:-2]
            val_key = sess_keys[-2]
            test_key = sess_keys[-1]
        elif len(sess_keys) == 2:
            train_keys = [sess_keys[0]]
            val_key = sess_keys[1]
            test_key = sess_keys[1] # Split single session samples between val & test
        elif len(sess_keys) == 1:
            train_keys = [sess_keys[0]]
            val_key = sess_keys[0]
            test_key = sess_keys[0]
        else:
            train_keys, val_key, test_key = [], "", ""
            
        for sk, samples in sessions_dict.items():
            if sk in train_keys:
                dataset_splits["train"].extend(samples)
                split_counts[cls_name]["train"] += len(samples)
            elif sk == val_key and len(sess_keys) >= 3:
                dataset_splits["val"].extend(samples)
                split_counts[cls_name]["val"] += len(samples)
            elif sk == test_key and len(sess_keys) >= 3:
                dataset_splits["test"].extend(samples)
                split_counts[cls_name]["test"] += len(samples)
            else:
                # If 2 or 1 session, split samples 60/20/20
                n = len(samples)
                n_tr = max(1, int(n * 0.6))
                n_v = max(1, int(n * 0.2))
                
                dataset_splits["train"].extend(samples[:n_tr])
                dataset_splits["val"].extend(samples[n_tr:n_tr+n_v])
                dataset_splits["test"].extend(samples[n_tr+n_v:])
                
                split_counts[cls_name]["train"] += len(samples[:n_tr])
                split_counts[cls_name]["val"] += len(samples[n_tr:n_tr+n_v])
                split_counts[cls_name]["test"] += len(samples[n_tr+n_v:])

    # Print Dataset Breakdown Table
    print(f"\n{'Class':<15} | {'Sessions':<10} | {'Train':<10} | {'Val':<10} | {'Test':<10} | {'Total':<10}")
    print("-" * 75)
    for c in CLASS_NAMES:
        tr = split_counts[c]["train"]
        val = split_counts[c]["val"]
        te = split_counts[c]["test"]
        tot = tr + val + te
        sess_cnt = len(session_counts_per_class[c])
        print(f"{c:<15} | {sess_cnt:<10} | {tr:<10} | {val:<10} | {te:<10} | {tot:<10}")

    out_dataset_dir = Path("outputs/landmark_dataset")
    out_dataset_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dataset_dir / "landmark_splits.json", "w") as f:
        json.dump(dataset_splits, f, indent=2)
        
    print(f"\nSaved landmark dataset to: {out_dataset_dir / 'landmark_splits.json'}")

if __name__ == "__main__":
    extract_landmark_dataset()
