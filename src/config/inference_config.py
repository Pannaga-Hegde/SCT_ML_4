"""Inference configuration module for GestureFlow.

Centralizes runtime parameters, camera settings, preprocessing modes, confidence thresholds,
multi-stage prediction stabilization parameters, visual overlay toggles, and output directory paths.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Tuple


class PreprocessMode(Enum):
    """Supported live-switchable image preprocessing modes."""

    GRAY = "gray"  # Mode 1: Standard Grayscale (BGR -> Gray)
    HIST_EQ = "hist_eq"  # Mode 2: Grayscale + Global Histogram Equalization
    CLAHE = "clahe"  # Mode 3: Grayscale + Contrast Limited Adaptive Histogram Equalization


@dataclass
class InferenceConfig:
    """Central configuration parameters for real-time desktop webcam gesture recognition."""

    # Project Directories
    project_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent.parent
    )
    checkpoint_path: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent.parent
        / "models"
        / "checkpoints"
        / "best_model.pth"
    )
    adapted_checkpoint_path: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent.parent
        / "models"
        / "checkpoints"
        / "webcam_adapted_model.pth"
    )
    landmark_checkpoint_path: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent.parent
        / "models"
        / "checkpoints"
        / "landmark_classifier.joblib"
    )
    model_choice: str = "landmark"  # "landmark", "adapted", or "original"
    output_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent.parent
        / "outputs"
        / "inference"
    )
    debug_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent.parent
        / "outputs"
        / "inference"
        / "debug"
    )
    screenshot_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent.parent
        / "outputs"
        / "inference"
        / "screenshots"
    )

    # Camera & Frame Settings
    camera_id: int = 0
    camera_width: int = 1280
    camera_height: int = 720
    target_fps: int = 30
    mirror_camera: bool = True

    # MediaPipe Hand Detection Specs
    max_num_hands: int = 2  # Will select largest hand for ROI
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    roi_padding_percent: float = 0.20  # 20% ROI padding

    # Preprocessing & Model Resolution Specs
    target_image_size: Tuple[int, int] = (128, 128)  # (width, height)
    num_classes: int = 10
    preprocess_mode: PreprocessMode = PreprocessMode.GRAY

    # Multi-Stage Stabilization & Decision Parameters
    confidence_threshold: float = 0.70  # 70% minimum threshold
    prediction_window_size: int = 5  # Sliding window size
    min_stable_consecutive_frames: int = 3  # Must match N consecutive frames before switching
    history_queue_size: int = 10  # Last 10 predictions history

    # Visualization Toggles
    show_landmarks: bool = True
    show_skeleton: bool = True
    show_bounding_box: bool = True
    show_fps: bool = True
    show_confidence: bool = True
    show_telemetry: bool = True
    developer_mode: bool = False
    prediction_paused: bool = False

    # Execution Hardware
    device: str = "cpu"
    seed: int = 42

    def __post_init__(self) -> None:
        """Ensure runtime output directories exist upon initialization."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)


# Default global instance
inference_config = InferenceConfig()
