"""PyTorch GestureCNN & MediaPipe Landmark predictor module for GestureFlow.

Supports both CNN image-based prediction and MediaPipe 3D Landmark Geometric feature prediction.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import joblib

from src.config.inference_config import InferenceConfig, inference_config
from src.models.cnn import GestureCNN
from src.inference.feature_engineering import extract_rich_geometric_features


class GesturePredictor:
    """Performs real-time gesture classification using CNN or MediaPipe Landmark Model."""

    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        cfg: InferenceConfig = inference_config,
        model_choice: Optional[str] = None,
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.active_model_choice = model_choice if model_choice else cfg.model_choice

        self.class_names = [
            "01_palm", "02_l", "03_fist", "04_fist_moved", "05_thumb",
            "06_index", "07_ok", "08_palm_moved", "09_c", "10_down",
        ]

        self.model = GestureCNN(
            num_classes=cfg.num_classes, dropout_rate=0.0
        ).to(self.device)
        
        self.landmark_model = None
        self.load_model(self.active_model_choice, checkpoint_path=checkpoint_path)

    def load_model(self, choice: str, checkpoint_path: Optional[Path] = None) -> str:
        """Load model weights based on choice ('landmark', 'adapted', or 'original')."""
        if choice == "landmark":
            lm_v2 = Path("models/checkpoints/landmark_classifier_v2.joblib")
            lm_v1 = Path("models/checkpoints/landmark_classifier.joblib")
            lm_path = checkpoint_path if checkpoint_path else (lm_v2 if lm_v2.exists() else lm_v1)
            
            if lm_path.exists():
                self.landmark_model = joblib.load(lm_path)
                self.checkpoint_path = lm_path
                self.active_model_choice = "landmark"
                return "landmark"
            else:
                choice = "adapted"

        if checkpoint_path:
            target_path = Path(checkpoint_path)
        elif choice == "adapted" and self.cfg.adapted_checkpoint_path.exists():
            target_path = self.cfg.adapted_checkpoint_path
        else:
            target_path = self.cfg.checkpoint_path

        if not target_path.exists():
            target_path = self.cfg.checkpoint_path

        checkpoint = torch.load(target_path, map_location=self.device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.eval()
        self.checkpoint_path = target_path
        self.active_model_choice = "adapted" if target_path == self.cfg.adapted_checkpoint_path else "original"
        return self.active_model_choice

    def switch_model(self) -> str:
        """Toggle active model between 'landmark', 'adapted', and 'original'."""
        if self.active_model_choice == "landmark":
            new_choice = "adapted"
        elif self.active_model_choice == "adapted":
            new_choice = "original"
        else:
            new_choice = "landmark"
        return self.load_model(new_choice)

    def predict(self, input_tensor: torch.Tensor) -> Dict[str, Tuple]:
        """Perform CNN forward pass prediction on image tensor [1, 1, 128, 128]."""
        start_time = time.time()

        with torch.no_grad():
            input_tensor = input_tensor.to(self.device)
            outputs = self.model(input_tensor)
            probs = torch.softmax(outputs, dim=1).squeeze(0).cpu().numpy()

        latency_ms = (time.time() - start_time) * 1000.0

        top_idx = int(np.argmax(probs))
        top_confidence = float(probs[top_idx])

        top3_indices = np.argsort(probs)[::-1][:3]
        top3 = [(self.class_names[idx], float(probs[idx])) for idx in top3_indices]

        return {
            "predicted_idx": top_idx,
            "predicted_label": self.class_names[top_idx],
            "confidence": top_confidence,
            "probabilities": [float(p) for p in probs],
            "top3": top3,
            "latency_ms": latency_ms,
            "model_type": f"CNN ({self.active_model_choice.upper()})"
        }

    def predict_landmarks(self, landmarks_21x3: np.ndarray) -> Dict[str, Tuple]:
        """Perform 3D landmark geometric feature classification on 21x3 landmark array."""
        start_time = time.time()
        
        if self.landmark_model is None:
            self.load_model("landmark")

        # Extract 86 rich geometric features or 63 raw normalized coordinates depending on model expectation
        try:
            features = extract_rich_geometric_features(landmarks_21x3).reshape(1, -1)
            if hasattr(self.landmark_model, "n_features_in_") and self.landmark_model.n_features_in_ == 63:
                features = features[:, :63]
        except Exception:
            features = landmarks_21x3.flatten().reshape(1, -1)
            
        if hasattr(self.landmark_model, "predict_proba"):
            probs = self.landmark_model.predict_proba(features)[0]
        else:
            pred_idx = int(self.landmark_model.predict(features)[0])
            probs = np.zeros(10, dtype=np.float32)
            probs[pred_idx] = 1.0

        latency_ms = (time.time() - start_time) * 1000.0

        top_idx = int(np.argmax(probs))
        top_confidence = float(probs[top_idx])

        top3_indices = np.argsort(probs)[::-1][:3]
        top3 = [(self.class_names[idx], float(probs[idx])) for idx in top3_indices]

        return {
            "predicted_idx": top_idx,
            "predicted_label": self.class_names[top_idx],
            "confidence": top_confidence,
            "probabilities": [float(p) for p in probs],
            "top3": top3,
            "latency_ms": latency_ms,
            "model_type": "LANDMARK_V2"
        }
