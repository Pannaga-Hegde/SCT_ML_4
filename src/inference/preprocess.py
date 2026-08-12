"""Hand ROI Preprocessing engine for GestureFlow.

Extracts hand Region of Interest (ROI) with configurable padding,
supports 3 live-switchable preprocessing modes (Grayscale, Histogram Equalization, CLAHE),
resizes to 128x128, normalizes, and constructs PyTorch input tensors [1, 1, 128, 128].
"""

from typing import Optional, Tuple
import cv2
import numpy as np
import torch

from src.config.inference_config import InferenceConfig, PreprocessMode, inference_config


class HandROIPreprocessor:
    """Extracts, preprocesses, and normalizes hand ROI images for GestureCNN model input."""

    def __init__(self, cfg: InferenceConfig = inference_config) -> None:
        """Initialize HandROIPreprocessor.

        Args:
            cfg: InferenceConfig instance.
        """
        self.cfg = cfg
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def extract_padded_roi(
        self,
        bgr_frame: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        padding_percent: Optional[float] = None,
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """Crop hand ROI from frame with percentage padding.

        Args:
            bgr_frame: Input BGR image array (height, width, 3).
            bbox: Bounding box tuple (x_min, y_min, x_max, y_max) or None.
            padding_percent: Percentage padding fraction (e.g. 0.20 for 20%).

        Returns:
            Tuple of (cropped_bgr_roi, padded_bbox_tuple).
        """
        h, w, c = bgr_frame.shape
        pad_pct = padding_percent if padding_percent is not None else self.cfg.roi_padding_percent

        if bbox is not None:
            x_min, y_min, x_max, y_max = bbox
            box_w = x_max - x_min
            box_h = y_max - y_min

            # First add normal padding
            pad_w = int(box_w * pad_pct)
            pad_h = int(box_h * pad_pct)
            
            x1 = x_min - pad_w
            y1 = y_min - pad_h
            x2 = x_max + pad_w
            y2 = y_max + pad_h
            
            curr_w = x2 - x1
            curr_h = y2 - y1
            
            # Force aspect ratio to match training (640:240 = 8:3)
            # If current ratio < 8/3, we need to increase width
            # If current ratio > 8/3, we need to increase height
            target_ratio = 640.0 / 240.0
            if curr_w / max(1, curr_h) < target_ratio:
                # Too tall, increase width
                new_w = int(curr_h * target_ratio)
                diff = new_w - curr_w
                x1 -= diff // 2
                x2 += diff - (diff // 2)
            else:
                # Too wide, increase height
                new_h = int(curr_w / target_ratio)
                diff = new_h - curr_h
                y1 -= diff // 2
                y2 += diff - (diff // 2)

            x_min_pad = max(0, x1)
            y_min_pad = max(0, y1)
            x_max_pad = min(w, x2)
            y_max_pad = min(h, y2)
        else:
            # Fallback: Crop center of frame if no bbox detected, keeping 8:3 ratio
            target_ratio = 640.0 / 240.0
            cx, cy = w // 2, h // 2
            # Max possible width or height
            new_w = min(w, int(h * target_ratio))
            new_h = int(new_w / target_ratio)
            x_min_pad = cx - new_w // 2
            x_max_pad = cx + new_w // 2
            y_min_pad = cy - new_h // 2
            y_max_pad = cy + new_h // 2

        roi_bgr = bgr_frame[y_min_pad:y_max_pad, x_min_pad:x_max_pad]

        # Guard against zero-sized crops
        if roi_bgr.size == 0:
            roi_bgr = bgr_frame

        return roi_bgr, (x_min_pad, y_min_pad, x_max_pad, y_max_pad)

    def preprocess_roi(
        self,
        bgr_roi: np.ndarray,
        mode: Optional[PreprocessMode] = None,
    ) -> np.ndarray:
        """Apply single-channel grayscale transformation based on selected mode.

        Modes:
            Mode 1 (GRAY): Standard BGR -> Grayscale
            Mode 2 (HIST_EQ): Grayscale + Global Histogram Equalization
            Mode 3 (CLAHE): Grayscale + Contrast Limited Adaptive Histogram Equalization

        Args:
            bgr_roi: Cropped BGR image numpy array.
            mode: PreprocessMode enum value or None (uses config default).

        Returns:
            Preprocessed 8-bit single-channel grayscale numpy array (128, 128).
        """
        active_mode = mode if mode is not None else self.cfg.preprocess_mode

        # 1. BGR to Grayscale
        gray = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2GRAY)

        # 2. Apply Mode Enhancements
        if active_mode == PreprocessMode.HIST_EQ:
            gray = cv2.equalizeHist(gray)
        elif active_mode == PreprocessMode.CLAHE:
            gray = self.clahe.apply(gray)

        # 3. Resize to Target Resolution (128x128) using INTER_AREA for area-based downsampling
        target_w, target_h = self.cfg.target_image_size
        gray_resized = cv2.resize(gray, (target_w, target_h), interpolation=cv2.INTER_AREA)

        return gray_resized

    def to_tensor(
        self,
        gray_image: np.ndarray,
        device: Optional[str] = None,
    ) -> torch.Tensor:
        """Convert 8-bit grayscale image array into normalized PyTorch input tensor [1, 1, 128, 128].

        Args:
            gray_image: 8-bit grayscale image (128, 128).
            device: PyTorch execution device ("cpu" or "cuda").

        Returns:
            PyTorch tensor of shape [1, 1, 128, 128] with values in range [-1.0, 1.0].
        """
        target_device = device if device is not None else self.cfg.device

        # Scale to [0.0, 1.0]
        img_float = gray_image.astype(np.float32) / 255.0

        # Normalize with mean=0.5, std=0.5 -> range [-1.0, 1.0]
        img_norm = (img_float - 0.5) / 0.5

        # Format shape [1, 1, 128, 128]
        tensor = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0).to(target_device)
        return tensor
