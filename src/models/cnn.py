"""Modular Convolutional Neural Network (GestureCNN) for Hand Gesture Recognition.

Designed for high classification accuracy on 10 gesture classes while maintaining a low parameter count
(~450K parameters) and sub-5ms CPU inference latency via Global Average Pooling.
"""

import torch
import torch.nn as nn


class GestureCNN(nn.Module):
    """Lightweight 4-Block PyTorch CNN for 10-Class Hand Gesture Recognition."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        dropout_rate: float = 0.3,
    ) -> None:
        """Initialize GestureCNN architecture.

        Args:
            in_channels: Number of input image channels (default 1 for grayscale).
            num_classes: Number of target gesture classes (default 10).
            dropout_rate: Classifier dropout rate (default 0.3).
        """
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

        # Feature Extractor: 4 Conv Blocks
        # Block 1: (1, 128, 128) -> (32, 64, 64)
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.1),
        )

        # Block 2: (32, 64, 64) -> (64, 32, 32)
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.2),
        )

        # Block 3: (64, 32, 32) -> (128, 16, 16)
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.25),
        )

        # Block 4: (128, 16, 16) -> (256, 8, 8)
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.3),
        )

        # Global Average Pooling (enforces spatial translation invariance & ultra-light params)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classification Head: Linear(256, 128) -> ReLU -> Dropout -> Linear(128, 10)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass executing feature extraction and classification.

        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            Logits tensor of shape (batch_size, num_classes).
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.global_pool(x)
        x = torch.flatten(x, 1)  # Flatten from (B, 256, 1, 1) -> (B, 256)
        logits = self.classifier(x)
        return logits

    @property
    def num_parameters(self) -> int:
        """Return total number of trainable model parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
