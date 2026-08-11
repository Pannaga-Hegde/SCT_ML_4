"""Standalone MediaPipe Hand Detection Diagnostic Tool.

Opens the webcam, runs MediaPipe Hands detection on each frame,
draws detected landmarks and bounding boxes, and prints detection
status to the terminal.  Press Q or ESC to exit.

Usage:
    python tools/mp_diagnostic.py
    python tools/mp_diagnostic.py --camera 1
"""

import argparse
import sys
import time

import cv2
import numpy as np

try:
    import mediapipe as mp

    if not hasattr(mp, "solutions"):
        print(
            f"[ERROR] mediapipe {getattr(mp, '__version__', '?')} does NOT have "
            "mp.solutions.hands (requires mediapipe 0.10.x, not 1.x)."
        )
        sys.exit(1)

    print(f"[OK] MediaPipe {mp.__version__} loaded")
    mp_hands_mod = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

except ImportError as exc:
    print(f"[ERROR] MediaPipe not installed: {exc}")
    sys.exit(1)


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


def run_diagnostic(camera_id: int = 0) -> None:
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera index {camera_id}.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print(f"[OK] Webcam {camera_id} opened")

    hands = mp_hands_mod.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    print("[OK] mp.solutions.hands.Hands() initialized")
    print("[RUNNING] Press Q or ESC to quit")

    frame_count = 0
    detection_count = 0
    start_time = time.time()
    WIN = "GestureFlow - MediaPipe Diagnostic"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        frame = cv2.flip(frame, 1)
        frame_count += 1
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        t0 = time.perf_counter()
        results = hands.process(rgb)
        mp_ms = (time.perf_counter() - t0) * 1000.0
        rgb.flags.writeable = True

        hand_detected = False
        if results.multi_hand_landmarks:
            hand_detected = True
            detection_count += 1
            for hand_lms in results.multi_hand_landmarks:
                xs = [lm.x * w for lm in hand_lms.landmark]
                ys = [lm.y * h for lm in hand_lms.landmark]
                x1, y1 = int(min(xs)), int(min(ys))
                x2, y2 = int(max(xs)), int(max(ys))
                pad_x, pad_y = int((x2-x1)*0.2), int((y2-y1)*0.2)
                cv2.rectangle(frame, (max(0,x1-pad_x), max(0,y1-pad_y)),
                              (min(w,x2+pad_x), min(h,y2+pad_y)), (0,255,255), 2)
                pts = [(int(lm.x*w), int(lm.y*h)) for lm in hand_lms.landmark]
                for a, b in HAND_CONNECTIONS:
                    if a < len(pts) and b < len(pts):
                        cv2.line(frame, pts[a], pts[b], (148,163,184), 1, cv2.LINE_AA)
                for px, py in pts:
                    cv2.circle(frame, (px, py), 4, (0,255,255), -1, cv2.LINE_AA)

        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0.0
        detect_rate = detection_count / frame_count * 100 if frame_count > 0 else 0.0

        status = "Hand: YES" if hand_detected else "Hand: NO "
        col = (0, 255, 128) if hand_detected else (0, 80, 240)
        cv2.rectangle(frame, (8,8), (360,120), (8,11,17), -1)
        cv2.putText(frame, status, (16,45), cv2.FONT_HERSHEY_SIMPLEX, 0.95, col, 2, cv2.LINE_AA)
        cv2.putText(frame, f"FPS:{fps:.1f}  MP:{mp_ms:.1f}ms  Det:{detect_rate:.1f}%",
                    (16,75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (184,163,148), 1, cv2.LINE_AA)

        cv2.imshow(WIN, frame)

        if frame_count % 30 == 0:
            print(f"  Frame {frame_count:5d} | FPS {fps:5.1f} | MP {mp_ms:5.1f}ms | Hand: {'YES' if hand_detected else 'NO '} | DetectRate: {detect_rate:.1f}%")

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break

    cap.release()
    hands.close()
    cv2.destroyAllWindows()
    print(f"[DONE] Frames:{frame_count}  Detected:{detection_count}  Rate:{detect_rate:.1f}%  MP:{mp.__version__}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    run_diagnostic(args.camera)
