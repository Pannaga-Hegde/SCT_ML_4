"""Unit tests for DatasetStatistics."""

from pathlib import Path
from PIL import Image
import pytest

from src.config.config import DatasetConfig
from src.dataset.statistics import DatasetStatistics


@pytest.fixture
def mock_dataset(tmp_path: Path) -> Path:
    """Create mock dataset directory with sample images."""
    ds_root = tmp_path / "leapGestRecog"
    for subj in ["00", "01"]:
        for cls in ["01_palm", "02_l"]:
            p = ds_root / subj / cls
            p.mkdir(parents=True, exist_ok=True)
            img = Image.new("L", (640, 240), color=200)
            img.save(p / "img1.png")
    return ds_root


def test_statistics_calculation(mock_dataset: Path, tmp_path: Path) -> None:
    """Test calculation of image counts, resolutions, and file sizes."""
    cfg = DatasetConfig(
        dataset_root=mock_dataset,
        output_dir=tmp_path / "output",
        train_subjects=["00"],
        val_subjects=["01"],
        test_subjects=[],
    )
    stats_engine = DatasetStatistics(cfg)
    summary = stats_engine.compute_statistics()

    assert summary.total_images == 4
    assert summary.total_subjects == 2
    assert summary.total_classes == 2
    assert summary.resolution_counts.get("640x240") == 4


def test_plot_generation(mock_dataset: Path, tmp_path: Path) -> None:
    """Test generation of EDA visualization plot artifacts."""
    out_dir = tmp_path / "output"
    cfg = DatasetConfig(
        dataset_root=mock_dataset,
        output_dir=out_dir,
        train_subjects=["00"],
        val_subjects=["01"],
        test_subjects=[],
    )
    stats_engine = DatasetStatistics(cfg)
    summary = stats_engine.compute_statistics()
    plots = stats_engine.generate_plots(summary)

    assert len(plots) == 4
    for plot_path in plots:
        assert plot_path.exists()
        assert plot_path.stat().st_size > 0
