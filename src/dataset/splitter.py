"""Subject-aware dataset splitter module for GestureFlow.

Enforces configuration-driven subject partitioning across train, validation, and test sets.
Guarantees zero subject overlap between splits to prevent subject data leakage.
"""

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Tuple

from src.config.config import DatasetConfig, config


@dataclass
class SplitResult:
    """Dataclass holding dataset split partitioning results."""

    train_subjects: List[str]
    val_subjects: List[str]
    test_subjects: List[str]
    train_files: List[str] = field(default_factory=list)
    val_files: List[str] = field(default_factory=list)
    test_files: List[str] = field(default_factory=list)
    train_class_counts: Dict[str, int] = field(default_factory=dict)
    val_class_counts: Dict[str, int] = field(default_factory=dict)
    test_class_counts: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert split results to dictionary for dataset_split.json artifact."""
        return {
            "train_subjects": self.train_subjects,
            "val_subjects": self.val_subjects,
            "test_subjects": self.test_subjects,
            "train_images_count": len(self.train_files),
            "val_images_count": len(self.val_files),
            "test_images_count": len(self.test_files),
            "train_class_counts": self.train_class_counts,
            "val_class_counts": self.val_class_counts,
            "test_class_counts": self.test_class_counts,
        }


class SubjectAwareSplitter:
    """Partitioner enforcing subject isolation across dataset splits."""

    def __init__(self, cfg: DatasetConfig = config) -> None:
        """Initialize SubjectAwareSplitter with config.

        Args:
            cfg: Dataset configuration dataclass.
        """
        self.cfg = cfg
        self.dataset_root = Path(cfg.dataset_root)
        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._validate_subject_config()

    def _validate_subject_config(self) -> None:
        """Assert zero overlap between train, validation, and test subject sets."""
        train_set = set(self.cfg.train_subjects)
        val_set = set(self.cfg.val_subjects)
        test_set = set(self.cfg.test_subjects)

        assert train_set.isdisjoint(val_set), (
            f"Subject leakage detected between Train and Val sets! Overlap: {train_set & val_set}"
        )
        assert train_set.isdisjoint(test_set), (
            f"Subject leakage detected between Train and Test sets! Overlap: {train_set & test_set}"
        )
        assert val_set.isdisjoint(test_set), (
            f"Subject leakage detected between Val and Test sets! Overlap: {val_set & test_set}"
        )

    def split_dataset(self) -> SplitResult:
        """Partition image files based on subject subdirectory IDs.

        Returns:
            SplitResult dataclass containing isolated file path lists and per-split class distributions.
        """
        result = SplitResult(
            train_subjects=sorted(self.cfg.train_subjects),
            val_subjects=sorted(self.cfg.val_subjects),
            test_subjects=sorted(self.cfg.test_subjects),
        )

        manifest_path = self.output_dir / "manifest.csv"
        if manifest_path.exists():
            import pandas as pd
            df = pd.read_csv(manifest_path)
            for _, row in df.iterrows():
                rel_str = str(row["relative_path"])
                split = str(row["dataset_split"])
                cls = str(row["gesture_class"])
                if split == "train":
                    result.train_files.append(rel_str)
                    result.train_class_counts[cls] = result.train_class_counts.get(cls, 0) + 1
                elif split == "val":
                    result.val_files.append(rel_str)
                    result.val_class_counts[cls] = result.val_class_counts.get(cls, 0) + 1
                elif split == "test":
                    result.test_files.append(rel_str)
                    result.test_class_counts[cls] = result.test_class_counts.get(cls, 0) + 1

            split_json_path = self.output_dir / "dataset_split.json"
            with open(split_json_path, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, indent=4)
            return result

        if not self.dataset_root.exists():
            return result

        train_subj_set = set(result.train_subjects)
        val_subj_set = set(result.val_subjects)
        test_subj_set = set(result.test_subjects)

        image_files = []
        for subj in sorted(self.cfg.all_subjects):
            subj_dir = self.dataset_root / subj
            if subj_dir.exists():
                image_files.extend(sorted(list(subj_dir.glob("*/*.png")) + list(subj_dir.glob("*/*.jpg"))))

        train_classes: Counter = Counter()
        val_classes: Counter = Counter()
        test_classes: Counter = Counter()

        for img_path in image_files:
            rel_parts = img_path.relative_to(self.dataset_root).parts
            if len(rel_parts) >= 2:
                subj = rel_parts[0]
                cls = rel_parts[1]
                rel_str = str(img_path.relative_to(self.dataset_root))

                if subj in train_subj_set:
                    result.train_files.append(rel_str)
                    train_classes[cls] += 1
                elif subj in val_subj_set:
                    result.val_files.append(rel_str)
                    val_classes[cls] += 1
                elif subj in test_subj_set:
                    result.test_files.append(rel_str)
                    test_classes[cls] += 1

        result.train_class_counts = dict(sorted(train_classes.items()))
        result.val_class_counts = dict(sorted(val_classes.items()))
        result.test_class_counts = dict(sorted(test_classes.items()))

        # Save dataset_split.json artifact
        split_json_path = self.output_dir / "dataset_split.json"
        with open(split_json_path, "w") as f:
            json.dump(result.to_dict(), f, indent=4)

        return result
