"""Unit tests for GestureCNN neural network architecture."""

import torch
import pytest

from src.models.cnn import GestureCNN


def test_gesture_cnn_instantiation() -> None:
    """Test GestureCNN instantiation with default parameters."""
    model = GestureCNN()
    assert model.in_channels == 1
    assert model.num_classes == 10
    assert isinstance(model.num_parameters, int)
    # Assert lightweight parameter budget (~450K parameters)
    assert 400_000 <= model.num_parameters <= 600_000


def test_gesture_cnn_forward_pass_shape() -> None:
    """Test forward pass output logits shape for batch size 32."""
    model = GestureCNN()
    dummy_input = torch.randn(32, 1, 128, 128)
    logits = model(dummy_input)

    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (32, 10)


def test_gesture_cnn_single_sample_forward_pass() -> None:
    """Test single sample forward pass shape for batch size 1."""
    model = GestureCNN()
    dummy_input = torch.randn(1, 1, 128, 128)
    logits = model(dummy_input)

    assert logits.shape == (1, 10)
