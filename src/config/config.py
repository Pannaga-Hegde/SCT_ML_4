"""Dataset and system configuration dataclass for GestureFlow.

Centralizes dataset paths, image dimensions, subject partition lists, random seeds,
and DataLoader options without hardcoding values in operational modules.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple


@dataclass
class DatasetConfig:
    """Central configuration parameters for dataset processing and model training."""

    # Project Root & Path Configs
    project_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent.parent
    )
    dataset_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent.parent
        / "archive"
        / "leapGestRecog"
    )
    output_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent.parent
        / "outputs"
        / "dataset"
    )
    models_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent.parent
        / "models"
        / "checkpoints"
    )

    # Subject Split Definitions (Configuration-Driven, 70% / 15% / 15%)
    train_subjects: List[str] = field(
        default_factory=lambda: ["00", "01", "02", "03", "04", "05", "06"]
    )
    val_subjects: List[str] = field(default_factory=lambda: ["07", "08"])
    test_subjects: List[str] = field(default_factory=lambda: ["09"])

    # Image Input Specs
    raw_image_size: Tuple[int, int] = (640, 240)  # (width, height)
    target_image_size: Tuple[int, int] = (128, 128)  # (width, height)
    channels: int = 1  # Grayscale

    # Normalization (Grayscale channel mean & std)
    mean: List[float] = field(default_factory=lambda: [0.5])
    std: List[float] = field(default_factory=lambda: [0.5])

    # Data Loader Hyperparameters
    batch_size: int = 32
    num_workers: int = 0  # Safe default for cross-platform compatibility
    pin_memory: bool = True
    seed: int = 42

    # Augmentation Settings
    max_rotation_degrees: float = 10.0
    max_translation_fraction: float = 0.05
    brightness_factor_range: Tuple[float, float] = (0.85, 1.15)
    contrast_factor_range: Tuple[float, float] = (0.85, 1.15)

    def __post_init__(self) -> None:
        """Ensure directories exist and validate subject partition sets upon initialization."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        train_set = set(self.train_subjects)
        val_set = set(self.val_subjects)
        test_set = set(self.test_subjects)

        assert train_set.isdisjoint(val_set), (
            f"Subject leakage detected between Train and Val sets! Overlap: {train_set & val_set}"
        )
        assert train_set.isdisjoint(test_set), (
            f"Subject leakage detected between Train and Test sets! Overlap: {train_set & test_set}"
        )
        assert val_set.isdisjoint(test_set), (
            f"Subject leakage detected between Val and Test sets! Overlap: {val_set & test_set}"
        )

    @property
    def all_subjects(self) -> List[str]:
        """Return full list of configured subjects sorted alphabetically."""
        return sorted(self.train_subjects + self.val_subjects + self.test_subjects)


# Default global instance
config = DatasetConfig()
