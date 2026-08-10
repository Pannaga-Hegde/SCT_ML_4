"""Dataset metadata generator module for GestureFlow.

Generates authoritative metadata artifacts: dataset_metadata.json, classes.json, and normalization.json.
Enforces alphabetical class sorting before assigning label indices to guarantee reproducible class indexing.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from src.config.config import DatasetConfig, config
from src.dataset.statistics import StatisticsSummary


class DatasetMetadataGenerator:
    """Generator for dataset metadata artifacts and class index mappings."""

    def __init__(self, cfg: DatasetConfig = config) -> None:
        """Initialize metadata generator with config.

        Args:
            cfg: Dataset configuration dataclass.
        """
        self.cfg = cfg
        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_classes_json(self, class_names: List[str]) -> Dict[int, str]:
        """Sort class names alphabetically and map integer label indices to class names.

        Args:
            class_names: List of folder names corresponding to gesture classes.

        Returns:
            Dictionary mapping integer label ID to string class name.
        """
        sorted_classes = sorted(class_names)
        class_mapping = {idx: cls_name for idx, cls_name in enumerate(sorted_classes)}

        classes_json_path = self.output_dir / "classes.json"
        with open(classes_json_path, "w") as f:
            json.dump(class_mapping, f, indent=4)

        return class_mapping

    def generate_normalization_json(self) -> Dict[str, List[float]]:
        """Generate normalization values JSON artifact.

        Returns:
            Dictionary containing mean and std values.
        """
        norm_data = {
            "mean": self.cfg.mean,
            "std": self.cfg.std,
            "channels": self.cfg.channels,
            "target_image_size": list(self.cfg.target_image_size),
        }
        norm_path = self.output_dir / "normalization.json"
        with open(norm_path, "w") as f:
            json.dump(norm_data, f, indent=4)

        return norm_data

    def generate_dataset_metadata(
        self, stats: StatisticsSummary, split_info: Dict
    ) -> Dict:
        """Generate authoritative dataset_metadata.json file.

        Args:
            stats: Computed StatisticsSummary metrics.
            split_info: Dataset split mapping info.

        Returns:
            Metadata dictionary written to outputs/dataset/dataset_metadata.json.
        """
        classes = sorted(list(stats.class_counts.keys()))
        subjects = sorted(list(stats.subject_counts.keys()))

        metadata = {
            "dataset_name": "LeapGestRecog",
            "dataset_version": "1.0.0",
            "project_name": "GestureFlow",
            "project_version": "0.1.0",
            "generation_timestamp": datetime.now().isoformat(),
            "total_images": stats.total_images,
            "num_subjects": stats.total_subjects,
            "num_classes": stats.total_classes,
            "subject_ids": subjects,
            "class_names": classes,
            "raw_image_size": list(self.cfg.raw_image_size),
            "target_image_size": list(self.cfg.target_image_size),
            "channels": self.cfg.channels,
            "color_space": "Infrared Grayscale",
            "random_seed": self.cfg.seed,
            "normalization": {
                "mean": self.cfg.mean,
                "std": self.cfg.std,
            },
            "dataset_split": {
                "train_subjects": split_info.get("train_subjects", []),
                "val_subjects": split_info.get("val_subjects", []),
                "test_subjects": split_info.get("test_subjects", []),
                "train_images": split_info.get("train_images_count", 0),
                "val_images": split_info.get("val_images_count", 0),
                "test_images": split_info.get("test_images_count", 0),
            },
        }

        metadata_path = self.output_dir / "dataset_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

        return metadata
