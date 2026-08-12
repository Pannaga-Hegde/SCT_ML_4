"""GestureFlow — Real-Time Webcam Gesture Recognition Demo.

Main application entry point. Runs complete end-to-end inference pipeline for both CNN and LANDMARK models:

  Webcam → CameraStream → HandDetector → [HandROIPreprocessor → CNN Predictor / Landmark Predictor]
        → PredictionStabilizer → OverlayRenderer → Display

Keyboard controls:
  Q  — Quit
  R  — Reset session statistics
  T  — Toggle model (LANDMARK ↔ ADAPTED CNN ↔ ORIGINAL CNN)
  M  — Toggle mirror (horizontal flip)
  L  — Toggle MediaPipe landmark rendering
  B  — Toggle bounding box
  S  — Toggle skeleton rendering
  F  — Toggle FPS display
  P  — Pause / resume prediction
  C  — Capture screenshot
  O  — Save debug bundle
  H  — Toggle telemetry HUD
  D  — Toggle Developer Mode window
  1  — Preprocessing mode: Grayscale
  2  — Preprocessing mode: Histogram Equalization
  3  — Preprocessing mode: CLAHE
"""

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.config.inference_config import InferenceConfig, PreprocessMode, inference_config
from src.inference.camera import CameraStream
from src.inference.hand_detector import HandDetector
from src.inference.overlay import OverlayRenderer
from src.inference.predictor import GesturePredictor
from src.inference.preprocess import HandROIPreprocessor
from src.inference.stabilizer import PredictionStabilizer

_MODE_MAP = {
    "gray": PreprocessMode.GRAY,
    "hist_eq": PreprocessMode.HIST_EQ,
    "clahe": PreprocessMode.CLAHE,
}

_WIN_MAIN = "GestureFlow — Real-Time Gesture Recognition"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GestureFlow — Real-time hand gesture recognition demo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera device index.")
    parser.add_argument("--no-mirror", action="store_true", help="Disable horizontal flip.")
    parser.add_argument(
        "--mode",
        choices=["gray", "hist_eq", "clahe"],
        default="gray",
        help="Initial preprocessing mode.",
    )
    parser.add_argument(
        "--model",
        choices=["landmark", "adapted", "original"],
        default="landmark",
        help="Model choice: 'landmark' (MediaPipe 3D), 'adapted' (Webcam CNN), or 'original' (LeapMotion CNN).",
    )
    parser.add_argument("--dev", action="store_true", help="Start with Developer Mode enabled.")
    parser.add_argument("--confidence", type=float, default=0.70, help="Minimum confidence threshold.")
    parser.add_argument("--window", type=int, default=5, help="Prediction stabilizer window size.")
    parser.add_argument("--mock", action="store_true", help="Run with synthetic frames for headless testing.")
    parser.add_argument("--frames", type=int, default=0, help="Max frames to run in mock mode (0 for infinite).")
    return parser.parse_args()


def _save_debug_bundle(
    cfg: InferenceConfig,
    frame: np.ndarray,
    roi_bgr: Optional[np.ndarray],
    gray_image: Optional[np.ndarray],
    prediction: Optional[dict],
    tag: str = "",
) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    bundle_dir = cfg.debug_dir / f"{ts}{('_' + tag) if tag else ''}"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(bundle_dir / "frame.jpg"), frame)

    if roi_bgr is not None and roi_bgr.size > 0:
        cv2.imwrite(str(bundle_dir / "roi.jpg"), roi_bgr)

    if gray_image is not None and gray_image.size > 0:
        cv2.imwrite(str(bundle_dir / "normalized.png"), gray_image)

    if prediction is not None:
        payload = {k: v for k, v in prediction.items() if k != "top3"}
        payload["top3"] = [
            {"label": lbl, "confidence": round(conf, 4)}
            for lbl, conf in (prediction.get("top3") or [])
        ]
        with open(bundle_dir / "prediction.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    print(f"[Debug] Bundle saved → {bundle_dir}")
    return bundle_dir


def _save_screenshot(cfg: InferenceConfig, frame: np.ndarray) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = cfg.screenshot_dir / f"screenshot_{ts}.jpg"
    cv2.imwrite(str(path), frame)
    print(f"[Screenshot] Saved → {path}")
    return path


def _append_session_log(
    log_path: Path,
    timestamp: str,
    label: str,
    confidence: float,
    is_stable: bool,
    preprocess_mode: str,
    fps: float,
) -> None:
    write_header = not log_path.exists()
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp", "label", "confidence", "is_stable",
                "preprocess_mode", "fps",
            ],
        )
        if write_header:
            writer.writeheader()
        writer.writerow({
            "timestamp": timestamp,
            "label": label,
            "confidence": round(confidence, 4),
            "is_stable": is_stable,
            "preprocess_mode": preprocess_mode,
            "fps": round(fps, 2),
        })


def run_demo(cfg: InferenceConfig, mock: bool = False, max_frames: int = 0) -> None:
    print("=" * 60)
    print("  GestureFlow — Real-Time Gesture Recognition")
    print("=" * 60)
    print(f"  Model Mode : {cfg.model_choice.upper()}")
    print(f"  Checkpoint : {cfg.checkpoint_path}")
    print(f"  Device     : {cfg.device}")
    print(f"  Preprocess : {cfg.preprocess_mode.value}")
    print(f"  Camera     : {'MOCK' if mock else cfg.camera_id}")
    print(f"  Threshold  : {cfg.confidence_threshold:.0%}")
    print("=" * 60)

    camera = CameraStream(cfg, mock=mock)
    camera.start()

    detector = HandDetector(cfg)
    preprocessor = HandROIPreprocessor(cfg)
    predictor = GesturePredictor(cfg=cfg, model_choice=cfg.model_choice)
    stabilizer = PredictionStabilizer(cfg)
    renderer = OverlayRenderer()

    session_log_path = cfg.output_dir / "session_log.csv"

    mirror = cfg.mirror_camera
    show_landmarks = cfg.show_landmarks
    show_skeleton = cfg.show_skeleton
    show_bbox = cfg.show_bounding_box
    show_fps = cfg.show_fps
    show_telemetry = cfg.show_telemetry
    paused = cfg.prediction_paused
    dev_mode = cfg.developer_mode

    current_mode = cfg.preprocess_mode

    last_prediction: Optional[dict] = None
    last_roi_bgr: Optional[np.ndarray] = None
    last_gray: Optional[np.ndarray] = None
    last_tensor: Optional[np.ndarray] = None
    last_bbox = None
    last_label = "???"
    last_confidence = 0.0
    last_is_stable = False
    last_mediapipe_ms = 0.0

    cv2.namedWindow(_WIN_MAIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(_WIN_MAIN, cfg.camera_width, cfg.camera_height)

    print("\n  [OK] All modules loaded. Window active.\n")

    frame_counter = 0
    try:
        while True:
            if max_frames > 0 and frame_counter >= max_frames:
                print(f"  Reached max frame limit ({max_frames}). Exiting loop.")
                break
            frame_counter += 1

            frame = camera.read_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            if mirror:
                frame = cv2.flip(frame, 1)

            mp_start = time.time()
            detection_result = detector.detect(frame)
            last_mediapipe_ms = (time.time() - mp_start) * 1000.0

            hand_detected = detection_result["hand_detected"]
            raw_landmarks = detection_result.get("landmarks") or []
            landmark_list = [(lm[0], lm[1]) for lm in raw_landmarks] if raw_landmarks else []
            bbox = detection_result.get("bbox")

            # Inference execution (Landmark or CNN)
            if not paused and hand_detected:
                if predictor.active_model_choice == "landmark" and raw_landmarks:
                    lms_array = np.array(raw_landmarks, dtype=np.float32)
                    prediction = predictor.predict_landmarks(lms_array)
                    roi_bgr, padded_bbox = preprocessor.extract_padded_roi(frame, bbox)
                    gray = None
                    tensor = None
                else:
                    roi_bgr, padded_bbox = preprocessor.extract_padded_roi(frame, bbox)
                    gray = preprocessor.preprocess_roi(roi_bgr, mode=current_mode)
                    tensor = preprocessor.to_tensor(gray)
                    prediction = predictor.predict(tensor)

                raw_label = prediction["predicted_label"]
                raw_conf = prediction["confidence"]

                stable_label, stable_conf, is_stable = stabilizer.update(raw_label, raw_conf)

                last_label = stable_label
                last_confidence = stable_conf
                last_is_stable = is_stable
                last_prediction = prediction
                last_roi_bgr = roi_bgr
                last_gray = gray
                last_tensor = tensor.cpu().numpy() if tensor is not None else None
                last_bbox = padded_bbox

                _append_session_log(
                    session_log_path,
                    timestamp=datetime.now().isoformat(),
                    label=stable_label,
                    confidence=stable_conf,
                    is_stable=is_stable,
                    preprocess_mode=current_mode.value,
                    fps=camera.get_fps(),
                )

            elif not hand_detected:
                stabilizer.reset()
                last_label = "---"
                last_confidence = 0.0
                last_is_stable = False
                last_prediction = None
                last_roi_bgr = None
                last_gray = None
                last_tensor = None
                last_bbox = None

            # Render HUD overlays
            if show_bbox and last_bbox is not None:
                renderer.draw_bounding_box(
                    frame, last_bbox, last_is_stable, last_confidence, show_bbox
                )

            renderer.draw_landmarks(
                frame, landmark_list, show_landmarks, show_skeleton
            )

            # Draw Gesture Label + Active Model Type indicator
            model_tag = predictor.active_model_choice.upper()
            display_label = f"[{model_tag}] {last_label}" if last_label != "---" else "GESTURE: ---"
            renderer.draw_gesture_label(
                frame, display_label, last_confidence, last_is_stable, paused
            )

            renderer.draw_telemetry_panel(
                frame,
                fps=camera.get_fps(),
                mediapipe_ms=last_mediapipe_ms,
                cnn_ms=last_prediction["latency_ms"] if last_prediction else 0.0,
                preprocess_mode=current_mode.value.upper() if predictor.active_model_choice != "landmark" else "3D_LM",
                confidence=last_confidence,
                hand_detected=hand_detected,
                show_fps=show_fps,
                show_telemetry=show_telemetry,
            )

            renderer.draw_prediction_history(
                frame, stabilizer.get_history(), show_telemetry
            )

            renderer.draw_keyboard_help(frame, show_telemetry)

            if dev_mode:
                renderer.update_developer_window(
                    roi_bgr=last_roi_bgr,
                    gray_normalized=last_gray,
                    tensor_data=last_tensor,
                    probabilities=last_prediction["probabilities"] if last_prediction else None,
                    top3=last_prediction["top3"] if last_prediction else None,
                    class_names=predictor.class_names,
                    enabled=True,
                )
            else:
                renderer.update_developer_window(
                    roi_bgr=None, gray_normalized=None, tensor_data=None,
                    probabilities=None, top3=None, class_names=None,
                    enabled=False,
                )

            cv2.imshow(_WIN_MAIN, frame)
            key = cv2.waitKey(1) & 0xFF

            if key in [ord("q"), ord("Q"), 27]:
                print("\n  [Q] Quit requested.")
                break
            elif key in [ord("r"), ord("R")]:
                stabilizer.reset()
                last_label = "???"
                last_confidence = 0.0
                print("  [R] Session statistics reset.")
            elif key in [ord("t"), ord("T")]:
                new_model_name = predictor.switch_model()
                stabilizer.reset()
                print(f"  [T] Switched active model to: {new_model_name.upper()} ({predictor.checkpoint_path.name})")
            elif key in [ord("m"), ord("M")]:
                mirror = not mirror
                print(f"  [M] Mirror: {'ON' if mirror else 'OFF'}")
            elif key in [ord("l"), ord("L")]:
                show_landmarks = not show_landmarks
                print(f"  [L] Landmarks: {'ON' if show_landmarks else 'OFF'}")
            elif key in [ord("b"), ord("B")]:
                show_bbox = not show_bbox
                print(f"  [B] Bounding box: {'ON' if show_bbox else 'OFF'}")
            elif key in [ord("s"), ord("S")]:
                show_skeleton = not show_skeleton
                print(f"  [S] Skeleton: {'ON' if show_skeleton else 'OFF'}")
            elif key in [ord("f"), ord("F")]:
                show_fps = not show_fps
                print(f"  [F] FPS display: {'ON' if show_fps else 'OFF'}")
            elif key in [ord("p"), ord("P")]:
                paused = not paused
                print(f"  [P] Prediction: {'PAUSED' if paused else 'RESUMED'}")
            elif key in [ord("h"), ord("H")]:
                show_telemetry = not show_telemetry
                print(f"  [H] Telemetry HUD: {'ON' if show_telemetry else 'OFF'}")
            elif key in [ord("d"), ord("D")]:
                dev_mode = not dev_mode
                if not dev_mode:
                    renderer.destroy_developer_window()
                print(f"  [D] Developer Mode: {'ON' if dev_mode else 'OFF'}")

    except KeyboardInterrupt:
        print("\n  Interrupted by Ctrl+C.")
    finally:
        camera.stop()
        detector.close()
        cv2.destroyAllWindows()
        print("  GestureFlow exited cleanly.")


def main() -> None:
    args = _parse_args()
    cfg = inference_config
    cfg.camera_id = args.camera
    cfg.mirror_camera = not args.no_mirror
    cfg.preprocess_mode = _MODE_MAP[args.mode]
    cfg.model_choice = args.model
    cfg.developer_mode = args.dev
    cfg.confidence_threshold = args.confidence
    cfg.prediction_window_size = args.window

    run_demo(cfg, mock=args.mock, max_frames=args.frames)


if __name__ == "__main__":
    main()
