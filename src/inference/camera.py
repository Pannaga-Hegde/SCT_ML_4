"""Webcam capture module for GestureFlow desktop application.

Handles OpenCV VideoCapture initialization, camera resolution settings, horizontal mirroring,
frame rate timing, FPS calculation, and graceful hardware release.
"""

import time
from typing import Optional, Tuple
import cv2
import numpy as np

from src.config.inference_config import InferenceConfig, inference_config


class WebcamCapture:
    """Manages OpenCV webcam video stream and FPS telemetry."""

    def __init__(self, cfg: InferenceConfig = inference_config) -> None:
        """Initialize WebcamCapture.

        Args:
            cfg: InferenceConfig instance.
        """
        self.cfg = cfg
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_opened = False

        # FPS Telemetry
        self.frame_count = 0
        self.start_time = time.time()
        self.last_frame_time = time.time()
        self.current_fps = 0.0
        self.fps_history = []

    def open(self, camera_id: Optional[int] = None) -> bool:
        """Open camera hardware stream with configured resolution.

        Args:
            camera_id: Optional camera index override.

        Returns:
            True if camera opened successfully, False otherwise.
        """
        cid = camera_id if camera_id is not None else self.cfg.camera_id
        self.cap = cv2.VideoCapture(cid)

        if not self.cap.isOpened():
            self.is_opened = False
            return False

        # Configure hardware resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.camera_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.camera_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.cfg.target_fps)

        self.is_opened = True
        self.start_time = time.time()
        self.last_frame_time = time.time()
        return True

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read latest frame from camera and calculate real-time FPS.

        Returns:
            Tuple of (success_bool, BGR_image_array).
        """
        if not self.is_opened or self.cap is None:
            return False, None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            return False, None

        # Horizontal camera mirror
        if self.cfg.mirror_camera:
            frame = cv2.flip(frame, 1)

        # FPS Calculation
        current_time = time.time()
        dt = current_time - self.last_frame_time
        self.last_frame_time = current_time

        if dt > 0:
            instant_fps = 1.0 / dt
            # Exponential moving average for smooth FPS readout
            self.current_fps = (
                0.9 * self.current_fps + 0.1 * instant_fps
                if self.current_fps > 0
                else instant_fps
            )
            self.fps_history.append(self.current_fps)
            if len(self.fps_history) > 100:
                self.fps_history.pop(0)

        self.frame_count += 1
        return True, frame

    def get_average_fps(self) -> float:
        """Calculate average FPS over the session.

        Returns:
            Average FPS float.
        """
        elapsed = time.time() - self.start_time
        return float(self.frame_count / elapsed) if elapsed > 0 else 0.0

    def release(self) -> None:
        """Release camera hardware resource cleanly."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_opened = False
