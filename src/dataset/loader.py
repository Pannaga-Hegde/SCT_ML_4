"""PyTorch Dataset and DataLoader creation module for GestureFlow.

Implements lazy-loading GestureDataset returning (image_tensor, label_int, metadata_dict)
and central create_dataloaders factory function.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.config.config import DatasetConfig, config
from src.dataset.splitter import SubjectAwareSplitter
from src.dataset.transforms import get_train_transforms, get_val_transforms


class GestureDataset(Dataset):
    """PyTorch Dataset implementation for LeapGestRecog hand gesture dataset."""

    def __init__(
        self,
        image_relative_paths: List[str],
        class_to_idx: Dict[str, int],
        dataset_root: Path,
        transform: Optional[object] = None,
    ) -> None:
        """Initialize GestureDataset.

        Args:
            image_relative_paths: List of relative image file paths.
            class_to_idx: Mapping from string class name to integer label ID.
            dataset_root: Absolute root directory path of the dataset.
            transform: Optional torchvision transforms pipeline.
        """
        self.image_relative_paths = image_relative_paths
        self.class_to_idx = class_to_idx
        self.dataset_root = Path(dataset_root)
        self.transform = transform

    def __len__(self) -> int:
        """Return total number of dataset samples."""
        return len(self.image_relative_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict[str, str]]:
        """Fetch sample at index idx with lazy image loading.

        Args:
            idx: Integer index of the sample.

        Returns:
            Tuple of (image_tensor, label_int, metadata_dict).
        """
        rel_path = self.image_relative_paths[idx]
        abs_path = self.dataset_root / rel_path

        parts = Path(rel_path).parts
        subject_id = parts[0] if len(parts) >= 2 else "unknown"
        class_name = parts[1] if len(parts) >= 2 else "unknown"

        label = self.class_to_idx.get(class_name, -1)

        # Lazy load image as Grayscale ('L')
        with Image.open(abs_path) as img:
            image_pil = img.convert("L")

        if self.transform is not None:
            image_tensor = self.transform(image_pil)
        else:
            # Fallback if no transform passed
            image_tensor = get_val_transforms()(image_pil)

        metadata = {
            "subject": subject_id,
            "class_name": class_name,
            "filename": abs_path.name,
            "relative_path": rel_path,
        }

        return image_tensor, label, metadata


def create_dataloaders(
    cfg: DatasetConfig = config,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Factory function creating PyTorch DataLoaders for train, val, and test splits.

    Args:
        cfg: Dataset configuration dataclass.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    splitter = SubjectAwareSplitter(cfg)
    split_result = splitter.split_dataset()

    # Discover and sort class names alphabetically
    expected_classes = sorted(
        [
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
    )
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(expected_classes)}

    train_transforms = get_train_transforms(cfg)
    val_transforms = get_val_transforms(cfg)

    train_dataset = GestureDataset(
        image_relative_paths=split_result.train_files,
        class_to_idx=class_to_idx,
        dataset_root=cfg.dataset_root,
        transform=train_transforms,
    )

    val_dataset = GestureDataset(
        image_relative_paths=split_result.val_files,
        class_to_idx=class_to_idx,
        dataset_root=cfg.dataset_root,
        transform=val_transforms,
    )

    test_dataset = GestureDataset(
        image_relative_paths=split_result.test_files,
        class_to_idx=class_to_idx,
        dataset_root=cfg.dataset_root,
        transform=val_transforms,
    )

    generator = torch.Generator().manual_seed(cfg.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        generator=generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )

    return train_loader, val_loader, test_loader
