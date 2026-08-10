"""Dataset validator module for GestureFlow.

Executes structural integrity checks, single-pass file loadability verification using PIL,
zero-byte/corrupt image detection, duplicate filename detection, and SHA-256 duplicate image hash scanning.
"""

import hashlib
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set

from PIL import Image

from src.config.config import DatasetConfig, config


@dataclass
class ValidationReport:
    """Dataclass holding validation scan results."""

    total_images_scanned: int = 0
    valid_images_count: int = 0
    corrupt_images: List[str] = field(default_factory=list)
    zero_byte_images: List[str] = field(default_factory=list)
    missing_subject_folders: List[str] = field(default_factory=list)
    unexpected_subject_folders: List[str] = field(default_factory=list)
    missing_class_folders: List[str] = field(default_factory=list)
    duplicate_filenames: List[str] = field(default_factory=list)
    duplicate_hash_clusters: Dict[str, List[str]] = field(default_factory=dict)
    subject_folder_counts: Dict[str, int] = field(default_factory=dict)
    class_folder_counts: Dict[str, int] = field(default_factory=dict)
    is_valid: bool = True

    def to_dict(self) -> Dict:
        """Convert validation report to dictionary representation."""
        return {
            "total_images_scanned": self.total_images_scanned,
            "valid_images_count": self.valid_images_count,
            "corrupt_images_count": len(self.corrupt_images),
            "corrupt_images": self.corrupt_images,
            "zero_byte_images_count": len(self.zero_byte_images),
            "zero_byte_images": self.zero_byte_images,
            "missing_subject_folders": self.missing_subject_folders,
            "unexpected_subject_folders": self.unexpected_subject_folders,
            "missing_class_folders": self.missing_class_folders,
            "duplicate_filenames_count": len(self.duplicate_filenames),
            "duplicate_hash_clusters_count": len(self.duplicate_hash_clusters),
            "is_valid": self.is_valid,
        }


class DatasetValidator:
    """Engine for validating dataset hierarchy, loadability, and duplicate scans."""

    def __init__(self, cfg: DatasetConfig = config) -> None:
        """Initialize DatasetValidator with configuration.

        Args:
            cfg: Configuration object containing dataset paths and expected subjects.
        """
        self.cfg = cfg
        self.dataset_root = Path(cfg.dataset_root)

    def validate_structure(self) -> ValidationReport:
        """Validate directory hierarchy and expected subject/class subfolders.

        Returns:
            ValidationReport detailing subject/class folder status.
        """
        report = ValidationReport()

        if not self.dataset_root.exists():
            report.is_valid = False
            report.missing_subject_folders = self.cfg.all_subjects
            return report

        existing_subject_dirs = sorted(
            [d.name for d in self.dataset_root.iterdir() if d.is_dir()]
        )

        expected_subjects = set(self.cfg.all_subjects)
        found_subjects = set(existing_subject_dirs)

        report.missing_subject_folders = sorted(list(expected_subjects - found_subjects))
        report.unexpected_subject_folders = sorted(list(found_subjects - expected_subjects))

        expected_classes = [
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

        for subj in existing_subject_dirs:
            subj_path = self.dataset_root / subj
            class_dirs = [d.name for d in subj_path.iterdir() if d.is_dir()]
            for cls in expected_classes:
                if cls not in class_dirs:
                    report.missing_class_folders.append(f"{subj}/{cls}")

        if report.missing_subject_folders or report.missing_class_folders:
            report.is_valid = False

        return report

    def validate_full_dataset(self, check_hashes: bool = True) -> ValidationReport:
        """Execute ultra-fast single-pass validation scan over all dataset images.

        Args:
            check_hashes: If True, computes SHA-256 hashes to detect duplicate images.

        Returns:
            ValidationReport detailing scan results.
        """
        report = self.validate_structure()

        if not self.dataset_root.exists():
            return report

        filename_set: Set[str] = set()
        hash_to_paths: Dict[str, List[str]] = {}

        image_files = []
        for subj in sorted(self.cfg.all_subjects):
            subj_dir = self.dataset_root / subj
            if subj_dir.exists():
                image_files.extend(sorted(list(subj_dir.glob("*/*.png")) + list(subj_dir.glob("*/*.jpg"))))

        report.total_images_scanned = len(image_files)

        for idx, img_path in enumerate(image_files, 1):
            if idx % 5000 == 0 or idx == len(image_files):
                print(f"    ... scanned {idx}/{len(image_files)} images ({idx/len(image_files)*100:.1f}%)")

            rel_path = str(img_path.relative_to(self.dataset_root))

            # Duplicate filename check
            if img_path.name in filename_set:
                report.duplicate_filenames.append(rel_path)
            else:
                filename_set.add(img_path.name)

            # Single-pass read bytes
            try:
                data = img_path.read_bytes()
            except Exception as e:
                report.corrupt_images.append(f"{rel_path}: {str(e)}")
                report.is_valid = False
                continue

            # Zero byte check
            if len(data) == 0:
                report.zero_byte_images.append(rel_path)
                report.is_valid = False
                continue

            # Loadability and corruption check via PIL BytesIO (In-memory, single pass)
            try:
                with Image.open(io.BytesIO(data)) as img:
                    img.load()
                report.valid_images_count += 1
            except Exception as e:
                report.corrupt_images.append(f"{rel_path}: {str(e)}")
                report.is_valid = False
                continue

            # SHA-256 hash duplication scan
            if check_hashes:
                file_hash = hashlib.sha256(data).hexdigest()
                if file_hash not in hash_to_paths:
                    hash_to_paths[file_hash] = []
                hash_to_paths[file_hash].append(rel_path)

        if check_hashes:
            report.duplicate_hash_clusters = {
                h: paths for h, paths in hash_to_paths.items() if len(paths) > 1
            }

        return report
