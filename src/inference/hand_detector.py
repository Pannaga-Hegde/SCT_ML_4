"""MediaPipe Hand Detection module for GestureFlow.

Wraps MediaPipe Hands API to detect hand landmarks, extract 21 3D keypoints,
select the largest hand ROI bounding box, and measure detection latency.
"""

import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

try:
    import mediapipe as mp
    if not hasattr(mp, "solutions"):
        # mediapipe 1.0.0+ dropped the legacy solutions API — need 0.10.x
        raise ImportError(
            f"mediapipe {getattr(mp, '__version__', 'unknown')} does not expose "
            "mp.solutions.hands. Install mediapipe<0.10.14 (e.g. pip install "
            "mediapipe==0.10.13 --no-deps)."
        )
    mp_solutions = mp.solutions
    MEDIAPIPE_AVAILABLE = True
except ImportError as _mp_import_err:
    import warnings
    warnings.warn(
        f"[GestureFlow] MediaPipe unavailable — hand detection disabled. "
        f"Reason: {_mp_import_err}"
    )
    mp = None
    mp_solutions = None
    MEDIAPIPE_AVAILABLE = False

from src.config.inference_config import InferenceConfig, inference_config


class MediaPipeHandDetector:
    """Detects hands, 21 keypoints, and bounding boxes using MediaPipe Hands."""

    def __init__(self, cfg: InferenceConfig = inference_config) -> None:
        """Initialize MediaPipeHandDetector.

        Args:
            cfg: InferenceConfig instance.
        """
        self.cfg = cfg
        self.mp_available = MEDIAPIPE_AVAILABLE and mp_solutions is not None

        if self.mp_available and mp_solutions is not None:
            self.mp_hands = mp_solutions.hands
            self.mp_drawing = mp_solutions.drawing_utils
            self.mp_drawing_styles = mp_solutions.drawing_styles

            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=cfg.max_num_hands,
                min_detection_confidence=cfg.min_detection_confidence,
                min_tracking_confidence=cfg.min_tracking_confidence,
            )
        else:
            self.hands = None

    def detect(self, bgr_frame: np.ndarray) -> Dict[str, Any]:
        """Detect hands in BGR frame and return largest hand detection result.

        Args:
            bgr_frame: BGR image numpy array of shape (height, width, 3).

        Returns:
            Dictionary containing:
                - 'hand_detected': bool
                - 'landmarks': List of 21 (x, y, z) normalized tuples or None
                - 'pixel_landmarks': List of 21 (x, y) integer pixel tuples or None
                - 'handedness': str ("Right", "Left", or "Unknown")
                - 'bbox': Tuple (x_min, y_min, x_max, y_max) or None
                - 'tracking_confidence': float
                - 'latency_ms': float
                - 'raw_results': MediaPipe Hands results object
        """
        start_time = time.time()
        h, w, c = bgr_frame.shape

        default_result = {
            "hand_detected": False,
            "landmarks": None,
            "pixel_landmarks": None,
            "handedness": "None",
            "bbox": None,
            "tracking_confidence": 0.0,
            "latency_ms": 0.0,
            "raw_results": None,
        }

        if not self.mp_available or self.hands is None:
            default_result["latency_ms"] = (time.time() - start_time) * 1000.0
            return default_result

        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb_frame = np.ascontiguousarray(rgb_frame)
        
        results = self.hands.process(rgb_frame)
        latency_ms = (time.time() - start_time) * 1000.0

        if not results.multi_hand_landmarks:
            default_result["latency_ms"] = latency_ms
            default_result["raw_results"] = results
            return default_result

        # Select largest hand bounding box if multiple hands detected
        best_area = -1.0
        best_hand_idx = 0
        all_hand_data = []

        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            x_coords = [lm.x * w for lm in hand_landmarks.landmark]
            y_coords = [lm.y * h for lm in hand_landmarks.landmark]

            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))

            area = (x_max - x_min) * (y_max - y_min)
            if area > best_area:
                best_area = area
                best_hand_idx = idx

            handedness_label = "Unknown"
            confidence = 0.90
            if results.multi_handedness and idx < len(results.multi_handedness):
                h_info = results.multi_handedness[idx].classification[0]
                handedness_label = h_info.label
                confidence = float(h_info.score)

            all_hand_data.append(
                {
                    "landmarks": [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark],
                    "pixel_landmarks": [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark],
                    "handedness": handedness_label,
                    "bbox": (x_min, y_min, x_max, y_max),
                    "tracking_confidence": confidence,
                }
            )

        best_hand = all_hand_data[best_hand_idx]
        best_hand["hand_detected"] = True
        best_hand["latency_ms"] = latency_ms
        best_hand["raw_results"] = results
        return best_hand

    def close(self) -> None:
        """Close MediaPipe Hands instance."""
        if self.hands is not None:
            self.hands.close()
            self.hands = None


# Alias for backward compatibility / demo import
HandDetector = MediaPipeHandDetector

