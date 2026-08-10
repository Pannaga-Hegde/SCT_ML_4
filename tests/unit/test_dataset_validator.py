"""Unit tests for DatasetValidator."""

import shutil
from pathlib import Path
from PIL import Image
import pytest

from src.config.config import DatasetConfig
from src.dataset.validator import DatasetValidator


@pytest.fixture
def temp_dataset_dir(tmp_path: Path) -> Path:
    """Create temporary mock dataset hierarchy for testing."""
    dataset_root = tmp_path / "leapGestRecog"
    subjects = ["00", "01", "02"]
    classes = [
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

    for subj in subjects:
        for cls in classes:
            dir_path = dataset_root / subj / cls
            dir_path.mkdir(parents=True, exist_ok=True)
            # Create valid dummy PNG image
            img = Image.new("L", (100, 100), color=128)
            img.save(dir_path / "sample.png")

    return dataset_root


def test_validator_structure_valid(temp_dataset_dir: Path, tmp_path: Path) -> None:
    """Test validator when dataset structure matches configuration."""
    cfg = DatasetConfig(
        dataset_root=temp_dataset_dir,
        output_dir=tmp_path / "output",
        train_subjects=["00"],
        val_subjects=["01"],
        test_subjects=["02"],
    )
    validator = DatasetValidator(cfg)
    report = validator.validate_structure()

    assert report.is_valid
    assert len(report.missing_subject_folders) == 0
    assert len(report.missing_class_folders) == 0


def test_validator_detects_corrupt_image(temp_dataset_dir: Path, tmp_path: Path) -> None:
    """Test validator detection of corrupt/truncated image bytes."""
    corrupt_file = temp_dataset_dir / "00" / "01_palm" / "corrupt.png"
    with open(corrupt_file, "wb") as f:
        f.write(b"not an image byte stream")

    cfg = DatasetConfig(
        dataset_root=temp_dataset_dir,
        output_dir=tmp_path / "output",
        train_subjects=["00"],
        val_subjects=["01"],
        test_subjects=["02"],
    )
    validator = DatasetValidator(cfg)
    report = validator.validate_full_dataset(check_hashes=False)

    assert not report.is_valid
    assert len(report.corrupt_images) == 1
    assert "corrupt.png" in report.corrupt_images[0]


def test_validator_detects_duplicate_hashes(temp_dataset_dir: Path, tmp_path: Path) -> None:
    """Test detection of identical SHA-256 image file hashes."""
    cfg = DatasetConfig(
        dataset_root=temp_dataset_dir,
        output_dir=tmp_path / "output",
        train_subjects=["00"],
        val_subjects=["01"],
        test_subjects=["02"],
    )
    validator = DatasetValidator(cfg)
    report = validator.validate_full_dataset(check_hashes=True)

    # Identical sample.png files were created across directories
    assert len(report.duplicate_hash_clusters) >= 1
