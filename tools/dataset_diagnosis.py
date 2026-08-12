import os
import re
import cv2
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

webcam_dir = Path("outputs/webcam_dataset")

# Collect stats
print("==========================================================================")
print(" WEBCAM ADAPTATION DATASET DIAGNOSIS")
print("==========================================================================")

class_raw_samples = defaultdict(set)
class_sessions = defaultdict(set)
class_split_raw = defaultdict(lambda: defaultdict(set))
class_split_augmented = defaultdict(lambda: defaultdict(int))
sample_to_splits = defaultdict(lambda: defaultdict(set))

for split in ["train", "val", "test"]:
    split_dir = webcam_dir / split
    if not split_dir.exists():
        continue
    for cls_dir in sorted(split_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        cls_name = cls_dir.name
        for f in cls_dir.glob("*.png"):
            m = re.match(r"(sample_\d{8}_\d{6}_\d+)", f.name)
            if m:
                sample_id = m.group(1)
                ts_session = sample_id.split("_")[1] + "_" + sample_id.split("_")[2][:4]
                class_raw_samples[cls_name].add(sample_id)
                class_sessions[cls_name].add(ts_session)
                class_split_raw[cls_name][split].add(sample_id)
                sample_to_splits[sample_id][cls_name].add(split)
            class_split_augmented[cls_name][split] += 1

print(f"{'Class':<15} | {'Raw Samples':<12} | {'Sessions':<10} | {'Train Files':<12} | {'Val Files':<10} | {'Test Files':<10}")
print("-" * 80)
for c in sorted(class_split_augmented.keys()):
    raw_cnt = len(class_raw_samples[c])
    sess_cnt = len(class_sessions[c])
    tr = class_split_augmented[c]["train"]
    val = class_split_augmented[c]["val"]
    te = class_split_augmented[c]["test"]
    print(f"{c:<15} | {raw_cnt:<12} | {sess_cnt:<10} | {tr:<12} | {val:<10} | {te:<10}")

print("\n==========================================================================")
print(" SPLIT LEAKAGE & OVERLAP CHECK")
print("==========================================================================")
leakage_count = 0
for sample_id, classes_dict in sample_to_splits.items():
    for c, splits in classes_dict.items():
        if len(splits) > 1:
            leakage_count += 1
            if leakage_count <= 10:
                print(f"Sample {sample_id} in {c} appears in multiple splits: {splits}")
print(f"Total overlapping samples across splits: {leakage_count}")

print("\n==========================================================================")
print(" IMAGE STATS & VISUAL INTENSITY ANALYSIS (128x128)")
print("==========================================================================")

for split in ["train", "val"]:
    print(f"\n--- Split: {split} ---")
    print(f"{'Class':<15} | {'Mean Brightness':<16} | {'Std Dev':<10} | {'Min Pixel':<10} | {'Max Pixel':<10} | {'Foreground (>100)':<20}")
    print("-" * 90)
    for c in sorted(class_split_augmented.keys()):
        cls_dir = webcam_dir / split / c
        if not cls_dir.exists(): continue
        means, stds, mins, maxs, fg_ratios = [], [], [], [], []
        for img_path in cls_dir.glob("*.png"):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            means.append(img.mean())
            stds.append(img.std())
            mins.append(img.min())
            maxs.append(img.max())
            fg_ratios.append((img > 100).mean())
        if means:
            print(f"{c:<15} | {np.mean(means):<16.2f} | {np.mean(stds):<10.2f} | {np.mean(mins):<10.1f} | {np.mean(maxs):<10.1f} | {np.mean(fg_ratios)*100:<19.2f}%")
