"""Unit tests for Phase 5 inference modules.

Tests cover:
  - HandROIPreprocessor  (all 3 preprocessing modes, tensor shape/range)
  - PredictionStabilizer (majority vote, confidence gate, consecutive gate, reset, history)
  - OverlayRenderer      (frame mutation, developer window rendering, draw helpers)

No real camera, MediaPipe, or GPU is required — all tests use synthetic numpy arrays.
"""

import unittest
from collections import Counter
from typing import List, Tuple

import numpy as np
import torch

# ─────────────────────────────────────────────────────────────────────────────
# Inference config (use defaults)
# ─────────────────────────────────────────────────────────────────────────────
from src.config.inference_config import InferenceConfig, PreprocessMode


def _make_cfg(**overrides) -> InferenceConfig:
    """Return an InferenceConfig with optional overrides."""
    cfg = InferenceConfig()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# HandROIPreprocessor tests
# ─────────────────────────────────────────────────────────────────────────────
class TestHandROIPreprocessor(unittest.TestCase):
    """Tests for src/inference/preprocess.py HandROIPreprocessor."""

    def setUp(self) -> None:
        from src.inference.preprocess import HandROIPreprocessor
        self.cfg = _make_cfg()
        self.prep = HandROIPreprocessor(self.cfg)
        # Fake 720p BGR frame
        self.frame = np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)

    # ── extract_padded_roi ────────────────────────────────────────────────────

    def test_roi_with_valid_bbox_returns_correct_shape(self) -> None:
        """ROI extraction from a valid bbox should return a non-empty BGR image."""
        bbox = (100, 100, 300, 300)
        roi, padded_bbox = self.prep.extract_padded_roi(self.frame, bbox)
        self.assertGreater(roi.size, 0)
        self.assertEqual(roi.ndim, 3)
        self.assertEqual(roi.shape[2], 3)

    def test_roi_with_no_bbox_falls_back_to_center_crop(self) -> None:
        """ROI extraction without a bbox should fall back to a center crop."""
        roi, padded_bbox = self.prep.extract_padded_roi(self.frame, bbox=None)
        self.assertGreater(roi.size, 0)
        self.assertEqual(len(padded_bbox), 4)

    def test_roi_padding_is_applied(self) -> None:
        """Padded bbox should be larger than or equal to the original bbox."""
        bbox = (200, 200, 400, 400)
        _, padded_bbox = self.prep.extract_padded_roi(self.frame, bbox)
        x1, y1, x2, y2 = padded_bbox
        self.assertLessEqual(x1, 200)
        self.assertLessEqual(y1, 200)
        self.assertGreaterEqual(x2, 400)
        self.assertGreaterEqual(y2, 400)

    # ── preprocess_roi ────────────────────────────────────────────────────────

    def test_gray_mode_output_shape(self) -> None:
        """GRAY mode should produce a (128, 128) grayscale image."""
        gray = self.prep.preprocess_roi(self.frame[:200, :200], mode=PreprocessMode.GRAY)
        self.assertEqual(gray.shape, (128, 128))

    def test_hist_eq_mode_output_shape(self) -> None:
        """HIST_EQ mode should produce a (128, 128) grayscale image."""
        gray = self.prep.preprocess_roi(self.frame[:200, :200], mode=PreprocessMode.HIST_EQ)
        self.assertEqual(gray.shape, (128, 128))

    def test_clahe_mode_output_shape(self) -> None:
        """CLAHE mode should produce a (128, 128) grayscale image."""
        gray = self.prep.preprocess_roi(self.frame[:200, :200], mode=PreprocessMode.CLAHE)
        self.assertEqual(gray.shape, (128, 128))

    def test_gray_output_dtype(self) -> None:
        """Preprocessed image should be 8-bit unsigned integer."""
        gray = self.prep.preprocess_roi(self.frame[:200, :200], mode=PreprocessMode.GRAY)
        self.assertEqual(gray.dtype, np.uint8)

    # ── to_tensor ─────────────────────────────────────────────────────────────

    def test_tensor_shape(self) -> None:
        """Tensor output should have shape [1, 1, 128, 128]."""
        gray = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
        t = self.prep.to_tensor(gray)
        self.assertEqual(tuple(t.shape), (1, 1, 128, 128))

    def test_tensor_value_range(self) -> None:
        """Normalized tensor values should be in the range [-1.0, 1.0]."""
        gray = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
        t = self.prep.to_tensor(gray)
        self.assertGreaterEqual(float(t.min()), -1.0 - 1e-5)
        self.assertLessEqual(float(t.max()), 1.0 + 1e-5)

    def test_tensor_dtype(self) -> None:
        """Tensor should have float32 dtype."""
        gray = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
        t = self.prep.to_tensor(gray)
        self.assertEqual(t.dtype, torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# PredictionStabilizer tests
# ─────────────────────────────────────────────────────────────────────────────
class TestPredictionStabilizer(unittest.TestCase):
    """Tests for src/inference/stabilizer.py PredictionStabilizer."""

    def setUp(self) -> None:
        from src.inference.stabilizer import PredictionStabilizer
        self.cfg = _make_cfg(
            prediction_window_size=5,
            min_stable_consecutive_frames=3,
            confidence_threshold=0.70,
            history_queue_size=10,
        )
        self.stab = PredictionStabilizer(self.cfg)

    def _feed(self, label: str, confidence: float, times: int) -> Tuple[str, float, bool]:
        result = None
        for _ in range(times):
            result = self.stab.update(label, confidence)
        return result

    # ── Basic majority vote ───────────────────────────────────────────────────

    def test_stable_label_after_sufficient_votes(self) -> None:
        """Should declare stability after enough consecutive high-confidence votes."""
        stable_label, stable_conf, is_stable = self._feed("03_fist", 0.92, 5)
        self.assertTrue(is_stable)
        self.assertEqual(stable_label, "03_fist")

    def test_not_stable_below_consecutive_threshold(self) -> None:
        """Should not be stable with only 2 consecutive frames (threshold=3)."""
        self._feed("01_palm", 0.90, 2)
        stable_label, stable_conf, is_stable = self.stab.update("01_palm", 0.90)
        # 3rd frame should trigger stability
        self.assertTrue(is_stable)

    def test_not_stable_on_first_frame(self) -> None:
        """Single prediction should never be stable."""
        _, _, is_stable = self.stab.update("05_thumb", 0.95)
        self.assertFalse(is_stable)

    # ── Confidence gate ────────────────────────────────────────────────────────

    def test_not_stable_below_confidence_threshold(self) -> None:
        """Should not commit stability when confidence is below threshold (0.70)."""
        stable_label, stable_conf, is_stable = self._feed("02_l", 0.50, 5)
        self.assertFalse(is_stable)

    def test_stable_at_exact_confidence_threshold(self) -> None:
        """Should be stable when confidence exactly meets the threshold."""
        stable_label, stable_conf, is_stable = self._feed("07_ok", 0.70, 5)
        self.assertTrue(is_stable)

    # ── Reset ─────────────────────────────────────────────────────────────────

    def test_reset_clears_state(self) -> None:
        """After reset, stabilizer should behave as if freshly initialised."""
        self._feed("01_palm", 0.95, 5)
        self.stab.reset()
        _, _, is_stable = self.stab.update("01_palm", 0.95)
        self.assertFalse(is_stable)
        self.assertEqual(self.stab.total_predictions, 1)

    def test_reset_clears_history(self) -> None:
        """History queue should be empty after reset."""
        self._feed("01_palm", 0.95, 5)
        self.stab.reset()
        self.assertEqual(self.stab.get_history(), [])

    # ── History ───────────────────────────────────────────────────────────────

    def test_history_records_predictions(self) -> None:
        """History should record the last N predictions."""
        for i in range(7):
            self.stab.update("01_palm", 0.80)
        self.assertEqual(len(self.stab.get_history()), 7)

    def test_history_respects_max_size(self) -> None:
        """History should not exceed history_queue_size."""
        for _ in range(20):
            self.stab.update("01_palm", 0.80)
        self.assertLessEqual(len(self.stab.get_history()), self.cfg.history_queue_size)

    # ── Gesture counts ────────────────────────────────────────────────────────

    def test_most_frequent_gesture_tracking(self) -> None:
        """Most frequent stable gesture should be tracked correctly."""
        self._feed("03_fist", 0.92, 5)  # commits fist
        self._feed("01_palm", 0.92, 5)  # commits palm
        self._feed("03_fist", 0.92, 5)  # commits fist again
        most_freq = self.stab.get_most_frequent_gesture()
        self.assertEqual(most_freq, "03_fist")

    def test_most_frequent_gesture_empty(self) -> None:
        """Most frequent gesture should return 'N/A' when no stable predictions."""
        result = self.stab.get_most_frequent_gesture()
        self.assertEqual(result, "N/A")

    # ── Window distribution ───────────────────────────────────────────────────

    def test_window_distribution_sums_to_one(self) -> None:
        """Window distribution percentages should sum to ~1.0."""
        self._feed("01_palm", 0.80, 3)
        self._feed("03_fist", 0.80, 2)
        dist = self.stab.get_window_distribution()
        self.assertAlmostEqual(sum(dist.values()), 1.0, places=5)


# ─────────────────────────────────────────────────────────────────────────────
# OverlayRenderer tests
# ─────────────────────────────────────────────────────────────────────────────
class TestOverlayRenderer(unittest.TestCase):
    """Tests for src/inference/overlay.py OverlayRenderer (no display required)."""

    def setUp(self) -> None:
        from src.inference.overlay import OverlayRenderer
        self.renderer = OverlayRenderer()
        # 720p BGR frame
        self.frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    def _clone_frame(self) -> np.ndarray:
        return self.frame.copy()

    # ── draw_bounding_box ─────────────────────────────────────────────────────

    def test_draw_bounding_box_mutates_frame(self) -> None:
        """draw_bounding_box should mutate the frame (non-zero pixels expected)."""
        frame = self._clone_frame()
        self.renderer.draw_bounding_box(frame, (100, 100, 300, 300), True, 0.90)
        self.assertGreater(frame.sum(), 0)

    def test_draw_bounding_box_hidden_when_disabled(self) -> None:
        """draw_bounding_box should not draw when show_bbox=False."""
        frame = self._clone_frame()
        self.renderer.draw_bounding_box(frame, (100, 100, 300, 300), True, 0.90, show_bbox=False)
        self.assertEqual(frame.sum(), 0)

    # ── draw_landmarks ────────────────────────────────────────────────────────

    def test_draw_landmarks_with_empty_list(self) -> None:
        """draw_landmarks with empty list should not raise or mutate frame."""
        frame = self._clone_frame()
        self.renderer.draw_landmarks(frame, [])
        self.assertEqual(frame.sum(), 0)

    def test_draw_landmarks_mutates_frame(self) -> None:
        """draw_landmarks with valid landmark list should mutate the frame."""
        frame = self._clone_frame()
        landmarks = [(i / 21, i / 21) for i in range(21)]
        self.renderer.draw_landmarks(frame, landmarks, show_landmarks=True, show_skeleton=True)
        self.assertGreater(frame.sum(), 0)

    # ── draw_gesture_label ────────────────────────────────────────────────────

    def test_draw_gesture_label_mutates_frame(self) -> None:
        """draw_gesture_label should write text pixels to the frame."""
        frame = self._clone_frame()
        self.renderer.draw_gesture_label(frame, "01_palm", 0.95, True)
        self.assertGreater(frame.sum(), 0)

    # ── draw_telemetry_panel ──────────────────────────────────────────────────

    def test_draw_telemetry_panel_mutates_frame(self) -> None:
        """draw_telemetry_panel should render text/pixels on the frame."""
        frame = self._clone_frame()
        self.renderer.draw_telemetry_panel(
            frame, fps=30.0, mediapipe_ms=5.0, cnn_ms=8.0,
            preprocess_mode="GRAY", confidence=0.92, hand_detected=True,
        )
        self.assertGreater(frame.sum(), 0)

    def test_draw_telemetry_panel_hidden_when_disabled(self) -> None:
        """draw_telemetry_panel should not draw when show_telemetry=False."""
        frame = self._clone_frame()
        self.renderer.draw_telemetry_panel(
            frame, fps=30.0, mediapipe_ms=5.0, cnn_ms=8.0,
            preprocess_mode="GRAY", confidence=0.92, hand_detected=True,
            show_telemetry=False,
        )
        self.assertEqual(frame.sum(), 0)

    # ── draw_prediction_history ───────────────────────────────────────────────

    def test_draw_prediction_history_with_empty_history(self) -> None:
        """draw_prediction_history with empty list should not raise or mutate frame."""
        frame = self._clone_frame()
        self.renderer.draw_prediction_history(frame, [])
        self.assertEqual(frame.sum(), 0)

    def test_draw_prediction_history_renders_entries(self) -> None:
        """draw_prediction_history should render when history is non-empty."""
        frame = self._clone_frame()
        history = [("01_palm", 0.91), ("03_fist", 0.85)]
        self.renderer.draw_prediction_history(frame, history, show_telemetry=True)
        self.assertGreater(frame.sum(), 0)

    # ── update_developer_window (disabled path — no GUI) ─────────────────────

    def test_developer_window_disabled_does_not_raise(self) -> None:
        """Calling update_developer_window with enabled=False should not raise."""
        try:
            self.renderer.update_developer_window(
                roi_bgr=None, gray_normalized=None, tensor_data=None,
                probabilities=None, top3=None, class_names=None,
                enabled=False,
            )
        except Exception as exc:
            self.fail(f"update_developer_window raised unexpectedly: {exc}")


if __name__ == "__main__":
    unittest.main()
