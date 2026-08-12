import os
import re
import cv2
import glob
import numpy as np
from pathlib import Path
from collections import defaultdict

webcam_dir = Path("outputs/webcam_dataset")

def analyze_dataset():
    splits = ["train", "val", "test"]
    classes = [
        "01_palm", "02_l", "03_fist", "04_fist_moved", "05_thumb",
        "06_index", "07_ok", "08_palm_moved", "09_c", "10_down"
    ]
    
    # 1. Raw Samples & Sessions
    raw_samples = defaultdict(dict)
    augmented_counts = defaultdict(dict)
    session_map = defaultdict(lambda: defaultdict(set))
    all_sample_splits = defaultdict(lambda: defaultdict(list))

    for split in splits:
        for c in classes:
            c_dir = webcam_dir / split / c
            if not c_dir.exists():
                augmented_counts[c][split] = 0
                continue
            files = list(c_dir.glob("*.png"))
            augmented_counts[c][split] = len(files)
            
            bases = set()
            for f in files:
                m = re.match(r"(sample_\d{8}_\d{6}_\d+)", f.name)
                if m:
                    sid = m.group(1)
                    bases.add(sid)
                    all_sample_splits[sid][c].append((split, f.name))
                    # Extract timestamp YYYYMMDD_HHMMSS -> session (date + hour + minute first 3 digits)
                    ts = sid.split("_")[1] + "_" + sid.split("_")[2][:3]
                    session_map[c][split].add(ts)
            raw_samples[c][split] = list(bases)

    print("==========================================================================================")
    print(" 1. SAMPLE COUNTS AND SESSION BREAKDOWN PER CLASS")
    print("==========================================================================================")
    print(f"{'Class':<15} | {'Raw Train':<10} | {'Raw Val':<8} | {'Raw Test':<9} | {'Total Raw':<10} | {'Train Aug':<10} | {'Sessions Count':<15}")
    print("-" * 90)
    
    for c in classes:
        tr_raw = len(raw_samples[c].get("train", []))
        val_raw = len(raw_samples[c].get("val", []))
        te_raw = len(raw_samples[c].get("test", []))
        tot_raw = len(set(raw_samples[c].get("train", []) + raw_samples[c].get("val", []) + raw_samples[c].get("test", [])))
        tr_aug = augmented_counts[c].get("train", 0)
        
        all_sess = set()
        for s in splits:
            all_sess.update(session_map[c].get(s, set()))
        
        print(f"{c:<15} | {tr_raw:<10} | {val_raw:<8} | {te_raw:<9} | {tot_raw:<10} | {tr_aug:<10} | {len(all_sess):<15}")

    print("\n==========================================================================================")
    print(" 2. SPLIT LEAKAGE & OVERLAP SUMMARY")
    print("==========================================================================================")
    leakage_by_class = defaultdict(int)
    total_leakage = 0
    for sid, cls_dict in all_sample_splits.items():
        for c, occurrences in cls_dict.items():
            splits_found = set(s for s, fn in occurrences)
            if len(splits_found) > 1:
                leakage_by_class[c] += 1
                total_leakage += 1
    
    for c in classes:
        print(f"{c:<15} : {leakage_by_class[c]} samples leaked across splits (e.g. val & test or train)")
    print(f"Total leaked raw samples across splits: {total_leakage}")

if __name__ == "__main__":
    analyze_dataset()
