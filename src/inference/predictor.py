"""PyTorch GestureCNN predictor module for GestureFlow.

Loads trained model weights checkpoint (best_model.pth), executes model forward pass,
applies Softmax probability calculation, extracts Top-3 prediction distributions,
and measures CPU inference latency in milliseconds.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch

from src.config.inference_config import InferenceConfig, inference_config
from src.models.cnn import GestureCNN


class GesturePredictor:
    """Performs real-time gesture classification using PyTorch GestureCNN."""

    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        cfg: InferenceConfig = inference_config,
    ) -> None:
        """Initialize GesturePredictor.

        Args:
            checkpoint_path: Path to best_model.pth artifact or None.
            cfg: InferenceConfig instance.
        """
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.checkpoint_path = (
            Path(checkpoint_path) if checkpoint_path else cfg.checkpoint_path
        )

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"GestureCNN checkpoint artifact missing at: {self.checkpoint_path}"
            )

        # Load gesture class catalog from classes.json
        classes_path = Path("outputs/dataset/classes.json")
        if classes_path.exists():
            with open(classes_path, "r", encoding="utf-8") as f:
                raw_classes = json.load(f)
                self.class_names = [raw_classes[str(i)] for i in range(len(raw_classes))]
        else:
            self.class_names = [
                "01_palm",
                "02_l",
                "03_fist",
                "04_fist_moved",
                "05_thumb",
                "06_index",
                "07_ok",
                "08_palm_moved",
                "09_c",
                "10_down",
            ]

        # Instantiate GestureCNN Architecture and load checkpoint state_dict
        self.model = GestureCNN(
            num_classes=cfg.num_classes, dropout_rate=0.0  # Dropout off during inference
        ).to(self.device)

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.eval()

    def predict(self, input_tensor: torch.Tensor) -> Dict[str, Tuple]:
        """Perform forward pass prediction on normalized image tensor [1, 1, 128, 128].

        Args:
            input_tensor: PyTorch tensor shape [1, 1, 128, 128].

        Returns:
            Dictionary containing:
                - 'predicted_idx': int
                - 'predicted_label': str
                - 'confidence': float (0.0 to 1.0)
                - 'probabilities': List[float] (10 floats)
                - 'top3': List[Tuple[str, float]]
                - 'latency_ms': float
        """
        start_time = time.time()

        with torch.no_grad():
            input_tensor = input_tensor.to(self.device)
            outputs = self.model(input_tensor)
            probs = torch.softmax(outputs, dim=1).squeeze(0).cpu().numpy()

        latency_ms = (time.time() - start_time) * 1000.0

        top_idx = int(np.argmax(probs))
        top_confidence = float(probs[top_idx])

        # Top-3 predictions list
        top3_indices = np.argsort(probs)[::-1][:3]
        top3 = [(self.class_names[idx], float(probs[idx])) for idx in top3_indices]

        return {
            "predicted_idx": top_idx,
            "predicted_label": self.class_names[top_idx],
            "confidence": top_confidence,
            "probabilities": [float(p) for p in probs],
            "top3": top3,
            "latency_ms": latency_ms,
        }
