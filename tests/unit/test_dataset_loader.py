"""Unit tests for GestureDataset and PyTorch DataLoaders."""

from pathlib import Path
import torch
from PIL import Image
import pytest

from src.config.config import DatasetConfig
from src.dataset.loader import GestureDataset, create_dataloaders


@pytest.fixture
def mock_dataset(tmp_path: Path) -> Path:
    """Create mock dataset structure for loader unit testing."""
    ds_root = tmp_path / "leapGestRecog"
    subjects = ["00", "01", "02"]
    classes = ["01_palm", "02_l"]

    for subj in subjects:
        for cls in classes:
            p = ds_root / subj / cls
            p.mkdir(parents=True, exist_ok=True)
            for i in range(5):
                img = Image.new("L", (200, 200), color=i * 40)
                img.save(p / f"sample_{i}.png")
    return ds_root


def test_gesture_dataset_indexing(mock_dataset: Path) -> None:
    """Test GestureDataset sample indexing and metadata dict contents."""
    rel_paths = ["00/01_palm/sample_0.png", "01/02_l/sample_1.png"]
    class_to_idx = {"01_palm": 0, "02_l": 1}

    ds = GestureDataset(
        image_relative_paths=rel_paths,
        class_to_idx=class_to_idx,
        dataset_root=mock_dataset,
    )

    assert len(ds) == 2
    tensor_img, label, meta = ds[0]

    assert isinstance(tensor_img, torch.Tensor)
    assert label == 0
    assert meta["subject"] == "00"
    assert meta["class_name"] == "01_palm"
    assert meta["filename"] == "sample_0.png"


def test_create_dataloaders(mock_dataset: Path, tmp_path: Path) -> None:
    """Test create_dataloaders factory and batch extraction."""
    cfg = DatasetConfig(
        dataset_root=mock_dataset,
        output_dir=tmp_path / "output",
        train_subjects=["00"],
        val_subjects=["01"],
        test_subjects=["02"],
        batch_size=4,
        num_workers=0,
    )

    train_loader, val_loader, test_loader = create_dataloaders(cfg)

    assert len(train_loader.dataset) == 10
    assert len(val_loader.dataset) == 10
    assert len(test_loader.dataset) == 10

    # Extract one batch
    images, labels, meta = next(iter(train_loader))
    assert images.shape == (4, 1, 128, 128)
    assert labels.shape == (4,)
