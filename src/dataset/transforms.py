"""Preprocessing and data augmentation transforms module for GestureFlow.

Enforces safe data augmentations for training (subtle rotation, small translation, mild contrast/brightness adjust)
and strictly prohibits label-corrupting augmentations (vertical/horizontal flips).
"""

from torchvision import transforms

from src.config.config import DatasetConfig, config


def get_val_transforms(cfg: DatasetConfig = config) -> transforms.Compose:
    """Return deterministic evaluation/test preprocessing transforms pipeline.

    Pipeline: PIL Image -> Resize -> ToTensor -> Normalize

    Args:
        cfg: Dataset configuration dataclass.

    Returns:
        torchvision.transforms.Compose pipeline.
    """
    return transforms.Compose(
        [
            transforms.Resize(cfg.target_image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.mean, std=cfg.std),
        ]
    )


def get_train_transforms(cfg: DatasetConfig = config) -> transforms.Compose:
    """Return safe training data augmentation and preprocessing pipeline.

    Allowed Augmentations:
    - Subtle rotation (max +-10 deg)
    - Small translation (max +-5%)
    - Mild brightness & contrast adjustments (+-15%)

    Prohibited Augmentations:
    - RandomVerticalFlip (corrupts 05_thumb into 10_down)
    - RandomHorizontalFlip (distorts asymmetric 02_l L-Shape)
    - Perspective warp / finger-removing crops

    Args:
        cfg: Dataset configuration dataclass.

    Returns:
        torchvision.transforms.Compose pipeline.
    """
    return transforms.Compose(
        [
            transforms.Resize(cfg.target_image_size),
            transforms.RandomRotation(degrees=cfg.max_rotation_degrees),
            transforms.RandomAffine(
                degrees=0,
                translate=(
                    cfg.max_translation_fraction,
                    cfg.max_translation_fraction,
                ),
            ),
            transforms.ColorJitter(
                brightness=cfg.brightness_factor_range,
                contrast=cfg.contrast_factor_range,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.mean, std=cfg.std),
        ]
    )
