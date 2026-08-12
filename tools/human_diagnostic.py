import cv2
import json
import time
from pathlib import Path
from datetime import datetime
import numpy as np

from src.config.inference_config import inference_config, PreprocessMode
from src.inference.camera import CameraStream
from src.inference.hand_detector import HandDetector
from src.inference.preprocess import HandROIPreprocessor
from src.inference.predictor import GesturePredictor

def main():
    cfg = inference_config
    cfg.camera_id = 0
    
    out_dir = Path("outputs/inference/diagnostic")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    camera = CameraStream(cfg, mock=False)
    if not camera.start():
        print("Failed to open camera.")
        return
    
    detector = HandDetector(cfg)
    preprocessor = HandROIPreprocessor(cfg)
    predictor = GesturePredictor(cfg=cfg)
    
    class_names = predictor.class_names
    expected_class_idx = 0
    
    print("="*60)
    print(" HUMAN-IN-THE-LOOP DIAGNOSTIC")
    print("="*60)
    print(" Controls:")
    print(" [0-9] : Select expected gesture")
    print(" [SPACE] : Capture sample")
    print(" [Q] : Quit")
    print("-" * 60)
    for i, name in enumerate(class_names):
        print(f"  {i}: {name}")
    print("="*60)
    
    cv2.namedWindow("Diagnostic", cv2.WINDOW_NORMAL)
    
    try:
        while True:
            frame = camera.read_frame()
            if frame is None:
                time.sleep(0.01)
                continue
                
            frame = cv2.flip(frame, 1)
            display = frame.copy()
            
            res = detector.detect(frame)
            hand_detected = res["hand_detected"]
            bbox = res.get("bbox")
            
            curr_class_name = class_names[expected_class_idx]
            cv2.putText(display, f"Expected: [{expected_class_idx}] {curr_class_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            roi_bgr = None
            pred_gray = None
            if hand_detected and bbox is not None:
                roi_bgr, padded_bbox = preprocessor.extract_padded_roi(frame, bbox)
                
                if roi_bgr.size > 0:
                    # Default display is GRAY prediction
                    gray = preprocessor.preprocess_roi(roi_bgr, mode=PreprocessMode.GRAY)
                    tensor = preprocessor.to_tensor(gray)
                    pred_gray = predictor.predict(tensor)
                    
                    x1, y1, x2, y2 = padded_bbox
                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    pred_label = pred_gray['predicted_label']
                    conf = pred_gray['confidence']
                    cv2.putText(display, f"Pred (GRAY): {pred_label} ({conf:.2f})", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(display, "Hand: NO", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
            cv2.imshow("Diagnostic", display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("Quitting diagnostic...")
                break
            elif ord('0') <= key <= ord('9'):
                idx = key - ord('0')
                if idx < len(class_names):
                    expected_class_idx = idx
                    print(f"Selected expected gesture: {class_names[expected_class_idx]}")
            elif key == ord(' '):
                if not hand_detected or roi_bgr is None or roi_bgr.size == 0:
                    print("Cannot capture: No hand detected.")
                    continue
                
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                sample_dir = out_dir / f"sample_{ts}_{curr_class_name}"
                sample_dir.mkdir(parents=True, exist_ok=True)
                
                # Save frame and ROI
                cv2.imwrite(str(sample_dir / "frame.jpg"), frame)
                cv2.imwrite(str(sample_dir / "roi_bgr.jpg"), roi_bgr)
                
                # Test with GRAY, HIST_EQ, CLAHE
                results_payload = {
                    "expected_class": curr_class_name,
                    "roi_dimensions": {"width": roi_bgr.shape[1], "height": roi_bgr.shape[0]},
                    "predictions": {}
                }
                
                for mode in [PreprocessMode.GRAY, PreprocessMode.HIST_EQ, PreprocessMode.CLAHE]:
                    processed = preprocessor.preprocess_roi(roi_bgr, mode=mode)
                    tensor = preprocessor.to_tensor(processed)
                    pred = predictor.predict(tensor)
                    
                    mode_str = mode.value
                    cv2.imwrite(str(sample_dir / f"input_{mode_str}.png"), processed)
                    
                    results_payload["predictions"][mode_str] = {
                        "predicted_label": pred["predicted_label"],
                        "confidence": float(pred["confidence"]),
                        "probabilities": {class_names[i]: float(pred["probabilities"][i]) for i in range(len(class_names))},
                        "latency_ms": pred["latency_ms"]
                    }
                    
                with open(sample_dir / "data.json", "w") as f:
                    json.dump(results_payload, f, indent=2)
                    
                print(f"Captured sample -> {sample_dir}")
                
    finally:
        camera.stop()
        detector.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
