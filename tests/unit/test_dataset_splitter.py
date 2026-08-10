"""Unit tests for SubjectAwareSplitter."""

from pathlib import Path
from PIL import Image
import pytest

from src.config.config import DatasetConfig
from src.dataset.splitter import SubjectAwareSplitter


@pytest.fixture
def mock_dataset(tmp_path: Path) -> Path:
    """Create mock dataset structure for subject split testing."""
    ds_root = tmp_path / "leapGestRecog"
    subjects = ["00", "01", "02"]
    for subj in subjects:
        for cls in ["01_palm", "02_l"]:
            p = ds_root / subj / cls
            p.mkdir(parents=True, exist_ok=True)
            img = Image.new("L", (100, 100))
            img.save(p / "img.png")
    return ds_root


def test_subject_aware_splitter(mock_dataset: Path, tmp_path: Path) -> None:
    """Test subject-aware dataset partitioning with zero subject overlap."""
    cfg = DatasetConfig(
        dataset_root=mock_dataset,
        output_dir=tmp_path / "output",
        train_subjects=["00"],
        val_subjects=["01"],
        test_subjects=["02"],
    )
    splitter = SubjectAwareSplitter(cfg)
    result = splitter.split_dataset()

    assert len(result.train_files) == 2
    assert len(result.val_files) == 2
    assert len(result.test_files) == 2

    # Assert zero subject leakage
    train_subjs = {f.split("/")[0] for f in result.train_files}
    val_subjs = {f.split("/")[0] for f in result.val_files}
    test_subjs = {f.split("/")[0] for f in result.test_files}

    assert train_subjs.isdisjoint(val_subjs)
    assert train_subjs.isdisjoint(test_subjs)
    assert val_subjs.isdisjoint(test_subjs)


def test_splitter_raises_on_overlapping_config(tmp_path: Path) -> None:
    """Test that overlapping subject config raises AssertionError."""
    with pytest.raises(AssertionError):
        DatasetConfig(
            dataset_root=tmp_path,
            output_dir=tmp_path / "output",
            train_subjects=["00", "01"],
            val_subjects=["01"],  # Overlap!
            test_subjects=["02"],
        )
