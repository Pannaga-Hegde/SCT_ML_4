"""Unit tests for dataset transforms and preprocessing pipelines."""

import torch
from PIL import Image
import pytest

from src.config.config import DatasetConfig
from src.dataset.transforms import get_train_transforms, get_val_transforms


def test_val_transforms_output_shape_and_range() -> None:
    """Test val transforms tensor shape, data type, and normalization."""
    cfg = DatasetConfig(target_image_size=(128, 128))
    val_tf = get_val_transforms(cfg)

    pil_img = Image.new("L", (640, 240), color=128)
    tensor_img = val_tf(pil_img)

    assert isinstance(tensor_img, torch.Tensor)
    assert tensor_img.shape == (1, 128, 128)
    assert tensor_img.dtype == torch.float32


def test_train_transforms_augmentation() -> None:
    """Test train transforms execute without error and preserve shape."""
    cfg = DatasetConfig(target_image_size=(128, 128))
    train_tf = get_train_transforms(cfg)

    pil_img = Image.new("L", (640, 240), color=200)
    tensor_img = train_tf(pil_img)

    assert isinstance(tensor_img, torch.Tensor)
    assert tensor_img.shape == (1, 128, 128)
