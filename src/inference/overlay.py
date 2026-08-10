"""OpenCV overlay rendering engine for GestureFlow.

Renders dark-slate HUD telemetry (FPS, latency, confidence, gesture label, prediction history),
bounding box, MediaPipe landmarks/skeleton, and an optional Developer Mode debug window
(ROI, tensor heatmap, probability bar chart, Top-3 table) onto live webcam frames.

Color tokens from docs/design.md:
  CYAN    = #00f2fe  → Bounding box / stable label
  BLUE    = #4facfe  → Telemetry secondary
  EMERALD = #10b981  → High confidence  (≥ 85%)
  AMBER   = #f59e0b  → Medium confidence (60%–84%)
  ROSE    = #ef4444  → Low confidence   (< 60%)
  WHITE   = #f8fafc  → Primary text
  SLATE   = #94a3b8  → Secondary telemetry text
  BG      = #080b11  → Panel backgrounds
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ──────────────────────────────────────────
# Colour Palette (BGR tuples for OpenCV)
# ──────────────────────────────────────────
CYAN = (254, 242, 0)        # #00f2fe
BLUE = (254, 172, 79)       # #4facfe
EMERALD = (129, 185, 16)    # #10b981
AMBER = (11, 158, 245)      # #f59e0b
ROSE = (68, 68, 239)        # #ef4444
WHITE = (252, 250, 248)     # #f8fafc
SLATE = (184, 163, 148)     # #94a3b8
BG = (17, 11, 8)            # #080b11
DARK_PANEL = (42, 23, 15)   # darker translucent panel backing


def _confidence_color(confidence: float) -> Tuple[int, int, int]:
    """Return the design-spec BGR color based on confidence bracket."""
    if confidence >= 0.85:
        return EMERALD
    if confidence >= 0.60:
        return AMBER
    return ROSE


def _draw_alpha_rect(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    color: Tuple[int, int, int],
    alpha: float = 0.55,
) -> None:
    """Draw a semi-transparent filled rectangle (in-place)."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, cv2.FILLED)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def _put_text(
    img: np.ndarray,
    text: str,
    org: Tuple[int, int],
    color: Tuple[int, int, int] = WHITE,
    scale: float = 0.55,
    thickness: int = 1,
    font: int = cv2.FONT_HERSHEY_SIMPLEX,
) -> None:
    """Convenience wrapper for cv2.putText with anti-aliasing."""
    cv2.putText(img, text, org, font, scale, color, thickness, cv2.LINE_AA)


class OverlayRenderer:
    """Renders GestureFlow HUD overlay and optional Developer Mode debug window."""

    DEVWIN = "GestureFlow — Developer Mode"
    DEV_W, DEV_H = 600, 700

    def __init__(self) -> None:
        """Initialize OverlayRenderer."""
        self._dev_canvas: Optional[np.ndarray] = None

    # ──────────────────────────────────────────
    # Primary HUD
    # ──────────────────────────────────────────

    def draw_bounding_box(
        self,
        frame: np.ndarray,
        padded_bbox: Tuple[int, int, int, int],
        is_stable: bool,
        confidence: float,
        show_bbox: bool = True,
    ) -> None:
        """Draw hand ROI bounding box with confidence-gated colour."""
        if not show_bbox:
            return
        color = _confidence_color(confidence) if is_stable else CYAN
        x1, y1, x2, y2 = padded_bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

    def draw_landmarks(
        self,
        frame: np.ndarray,
        landmark_list: list,
        show_landmarks: bool = True,
        show_skeleton: bool = True,
    ) -> None:
        """Draw MediaPipe hand landmarks and skeleton connections.

        Args:
            frame: BGR frame to draw onto.
            landmark_list: List of (x, y) normalised landmark tuples.
            show_landmarks: Whether to render landmark dots.
            show_skeleton: Whether to render connecting lines.
        """
        if not landmark_list:
            return

        h, w = frame.shape[:2]
        pts = [(int(lx * w), int(ly * h)) for lx, ly in landmark_list]

        # MediaPipe hand skeleton connection pairs (21 landmarks)
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),   # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),    # Index
            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
            (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
            (5, 9), (9, 13), (13, 17),          # Palm arc
        ]

        if show_skeleton:
            for a, b in connections:
                if a < len(pts) and b < len(pts):
                    cv2.line(frame, pts[a], pts[b], SLATE, 1, cv2.LINE_AA)

        if show_landmarks:
            for px, py in pts:
                cv2.circle(frame, (px, py), 3, CYAN, cv2.FILLED, cv2.LINE_AA)
                cv2.circle(frame, (px, py), 3, WHITE, 1, cv2.LINE_AA)

    def draw_gesture_label(
        self,
        frame: np.ndarray,
        label: str,
        confidence: float,
        is_stable: bool,
        paused: bool = False,
    ) -> None:
        """Draw the main gesture label banner at top-left of frame."""
        h, w = frame.shape[:2]
        display_label = label.upper().replace("_", " ")
        if paused:
            display_label = f"[PAUSED] {display_label}"

        color = _confidence_color(confidence) if is_stable else SLATE

        # Background panel
        _draw_alpha_rect(frame, 8, 8, 340, 62, BG, alpha=0.65)
        _put_text(frame, display_label, (16, 45), color=color, scale=0.95, thickness=2)

    def draw_telemetry_panel(
        self,
        frame: np.ndarray,
        fps: float,
        mediapipe_ms: float,
        cnn_ms: float,
        preprocess_mode: str,
        confidence: float,
        hand_detected: bool,
        show_fps: bool = True,
        show_telemetry: bool = True,
    ) -> None:
        """Draw right-side dark telemetry panel with FPS, latency, confidence, mode stats.

        Args:
            frame: BGR frame to draw onto.
            fps: Current measured FPS.
            mediapipe_ms: MediaPipe detection latency in milliseconds.
            cnn_ms: CNN forward pass latency in milliseconds.
            preprocess_mode: Active preprocessing mode name string.
            confidence: Current gesture confidence (0.0 to 1.0).
            hand_detected: Whether a hand is currently visible.
            show_fps: Toggle FPS rendering.
            show_telemetry: Toggle full telemetry panel rendering.
        """
        if not show_telemetry:
            return

        h, w = frame.shape[:2]
        panel_x1 = w - 210
        panel_x2 = w - 4
        panel_y1 = 8
        panel_y2 = 195 if show_fps else 165

        _draw_alpha_rect(frame, panel_x1, panel_y1, panel_x2, panel_y2, BG, alpha=0.70)

        y = 30
        dx = panel_x1 + 10

        if show_fps:
            fps_color = EMERALD if fps >= 25 else (AMBER if fps >= 15 else ROSE)
            _put_text(frame, f"FPS: {fps:5.1f}", (dx, y), color=fps_color)
            y += 24

        _put_text(frame, f"MP:  {mediapipe_ms:5.1f} ms", (dx, y), color=SLATE)
        y += 22
        _put_text(frame, f"CNN: {cnn_ms:5.1f} ms", (dx, y), color=SLATE)
        y += 22
        _put_text(frame, f"Mode: {preprocess_mode}", (dx, y), color=BLUE)
        y += 22

        conf_color = _confidence_color(confidence)
        _put_text(frame, f"Conf: {confidence * 100:5.1f}%", (dx, y), color=conf_color)
        y += 22

        status_color = EMERALD if hand_detected else ROSE
        status_text = "Hand: YES" if hand_detected else "Hand: NO"
        _put_text(frame, status_text, (dx, y), color=status_color)

    def draw_prediction_history(
        self,
        frame: np.ndarray,
        history: List[Tuple[str, float]],
        show_telemetry: bool = True,
    ) -> None:
        """Draw last-N prediction history panel at bottom-left of frame.

        Args:
            frame: BGR frame to draw onto.
            history: List of (label, confidence) tuples (most recent last).
            show_telemetry: Toggle history panel rendering.
        """
        if not show_telemetry or not history:
            return

        h, w = frame.shape[:2]
        n = min(len(history), 10)
        recent = list(history)[-n:]

        panel_h = n * 18 + 16
        panel_y2 = h - 10
        panel_y1 = panel_y2 - panel_h

        _draw_alpha_rect(frame, 8, panel_y1, 280, panel_y2, BG, alpha=0.60)
        _put_text(frame, "RECENT PREDICTIONS", (14, panel_y1 + 14), color=SLATE, scale=0.42)

        for i, (lbl, conf) in enumerate(recent):
            row_y = panel_y1 + 28 + i * 18
            color = _confidence_color(conf)
            short = lbl.replace("_", " ").upper()
            _put_text(
                frame,
                f"{short:<20s}  {conf * 100:4.1f}%",
                (14, row_y),
                color=color,
                scale=0.42,
            )

    def draw_keyboard_help(
        self,
        frame: np.ndarray,
        show_telemetry: bool = True,
    ) -> None:
        """Draw compact keyboard shortcut legend at bottom-right of frame."""
        if not show_telemetry:
            return

        h, w = frame.shape[:2]
        shortcuts = [
            "Q=Quit  R=Reset",
            "M=Mirror  L=Landmarks",
            "B=BBox  S=Skeleton",
            "F=FPS  H=HUD  P=Pause",
            "D=DevMode  O=DebugSave",
            "1=Gray  2=HistEq  3=CLAHE",
            "C=Screenshot  SPACE=Snap",
        ]
        x = w - 200
        y_start = h - len(shortcuts) * 16 - 10

        _draw_alpha_rect(frame, x - 6, y_start - 12, w - 4, h - 4, BG, alpha=0.55)
        for i, line in enumerate(shortcuts):
            _put_text(frame, line, (x, y_start + i * 16), color=SLATE, scale=0.38, thickness=1)

    # ──────────────────────────────────────────
    # Developer Mode Debug Window
    # ──────────────────────────────────────────

    def update_developer_window(
        self,
        roi_bgr: Optional[np.ndarray],
        gray_normalized: Optional[np.ndarray],
        tensor_data: Optional[np.ndarray],
        probabilities: Optional[List[float]],
        top3: Optional[List[Tuple[str, float]]],
        class_names: Optional[List[str]],
        enabled: bool = True,
    ) -> None:
        """Render and display the Developer Mode debug window.

        Args:
            roi_bgr: Cropped BGR ROI image or None.
            gray_normalized: Preprocessed grayscale (0-255) or None.
            tensor_data: Raw tensor numpy array (values in [-1, 1]) or None.
            probabilities: List of 10 class probabilities or None.
            top3: Top-3 (label, confidence) list or None.
            class_names: All 10 class name strings or None.
            enabled: Whether to show or destroy the window.
        """
        if not enabled:
            try:
                cv2.destroyWindow(self.DEVWIN)
            except Exception:
                pass
            self._dev_canvas = None
            return

        canvas = np.full((self.DEV_H, self.DEV_W, 3), BG, dtype=np.uint8)

        # Title bar
        cv2.rectangle(canvas, (0, 0), (self.DEV_W, 28), DARK_PANEL, cv2.FILLED)
        _put_text(canvas, "GestureFlow — Developer Mode", (10, 20), color=CYAN, scale=0.62, thickness=1)

        y_cursor = 38

        # ── ROI + Gray side by side ──
        thumb_size = 128
        if roi_bgr is not None and roi_bgr.size > 0:
            roi_thumb = cv2.resize(roi_bgr, (thumb_size, thumb_size))
            canvas[y_cursor: y_cursor + thumb_size, 10: 10 + thumb_size] = roi_thumb
            _put_text(canvas, "ROI (BGR)", (10, y_cursor + thumb_size + 14), color=SLATE, scale=0.42)

        if gray_normalized is not None and gray_normalized.size > 0:
            gray_resized = cv2.resize(gray_normalized, (thumb_size, thumb_size))
            gray_bgr = cv2.cvtColor(gray_resized, cv2.COLOR_GRAY2BGR)
            canvas[y_cursor: y_cursor + thumb_size, 150: 150 + thumb_size] = gray_bgr
            _put_text(canvas, "Normalized", (150, y_cursor + thumb_size + 14), color=SLATE, scale=0.42)

        # ── Tensor heatmap ──
        if tensor_data is not None and tensor_data.size > 0:
            t = tensor_data.squeeze()
            # Scale from [-1,1] to [0,255]
            t_vis = np.clip((t + 1.0) / 2.0 * 255.0, 0, 255).astype(np.uint8)
            t_resized = cv2.resize(t_vis, (thumb_size, thumb_size))
            t_heatmap = cv2.applyColorMap(t_resized, cv2.COLORMAP_INFERNO)
            canvas[y_cursor: y_cursor + thumb_size, 290: 290 + thumb_size] = t_heatmap
            _put_text(canvas, "Tensor", (290, y_cursor + thumb_size + 14), color=SLATE, scale=0.42)

        y_cursor += thumb_size + 30

        # ── Probability bar chart ──
        if probabilities and class_names:
            bar_section_h = 180
            _put_text(canvas, "CLASS PROBABILITIES", (10, y_cursor), color=BLUE, scale=0.48, thickness=1)
            y_cursor += 18
            bar_w_max = self.DEV_W - 140
            bar_h = 14
            for i, (cls, prob) in enumerate(zip(class_names, probabilities)):
                bar_len = int(prob * bar_w_max)
                bar_color = _confidence_color(prob)
                short = cls.split("_", 1)[-1].upper()
                cv2.rectangle(
                    canvas,
                    (100, y_cursor + i * (bar_h + 4)),
                    (100 + max(bar_len, 2), y_cursor + i * (bar_h + 4) + bar_h),
                    bar_color,
                    cv2.FILLED,
                )
                _put_text(
                    canvas,
                    f"{short[:8]:<8s}",
                    (4, y_cursor + i * (bar_h + 4) + 11),
                    color=SLATE,
                    scale=0.38,
                )
                _put_text(
                    canvas,
                    f"{prob * 100:5.1f}%",
                    (100 + bar_w_max + 4, y_cursor + i * (bar_h + 4) + 11),
                    color=bar_color,
                    scale=0.38,
                )
            y_cursor += len(class_names) * (bar_h + 4) + 16

        # ── Top-3 table ──
        if top3:
            _put_text(canvas, "TOP-3 PREDICTIONS", (10, y_cursor), color=BLUE, scale=0.48, thickness=1)
            y_cursor += 18
            for rank, (lbl, conf) in enumerate(top3, 1):
                row_color = _confidence_color(conf)
                _put_text(
                    canvas,
                    f"#{rank}  {lbl.replace('_', ' ').upper():<22s}  {conf * 100:5.1f}%",
                    (10, y_cursor),
                    color=row_color,
                    scale=0.48,
                )
                y_cursor += 20

        self._dev_canvas = canvas
        cv2.imshow(self.DEVWIN, canvas)

    def destroy_developer_window(self) -> None:
        """Safely destroy the developer mode window."""
        try:
            cv2.destroyWindow(self.DEVWIN)
        except Exception:
            pass
        self._dev_canvas = None
