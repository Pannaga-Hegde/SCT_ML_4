"""Multi-stage prediction stabilizer for GestureFlow.

Implements a sliding window majority vote + confidence averaging + consecutive stable frame
counting stabilization strategy to eliminate rapid gesture label flickering during live webcam inference.
"""

from collections import Counter, deque
from typing import Deque, Dict, List, Optional, Tuple

from src.config.inference_config import InferenceConfig, inference_config


class PredictionStabilizer:
    """Multi-stage gesture prediction stabilizer using window voting + confidence gating."""

    def __init__(self, cfg: InferenceConfig = inference_config) -> None:
        """Initialize PredictionStabilizer.

        Args:
            cfg: InferenceConfig instance.
        """
        self.cfg = cfg
        self.window_size = cfg.prediction_window_size
        self.min_consecutive = cfg.min_stable_consecutive_frames
        self.confidence_threshold = cfg.confidence_threshold
        self.history_size = cfg.history_queue_size

        # Sliding window prediction queue
        self._window: Deque[str] = deque(maxlen=self.window_size)
        self._confidence_window: Deque[float] = deque(maxlen=self.window_size)

        # Consecutive stable frame counter
        self._current_stable_label: Optional[str] = None
        self._consecutive_count: int = 0
        self._committed_label: Optional[str] = None

        # Last 10 prediction history for display
        self._history: Deque[Tuple[str, float]] = deque(maxlen=self.history_size)

        # Session statistics
        self.total_predictions = 0
        self.gesture_counts: Counter = Counter()

    def update(self, label: str, confidence: float) -> Tuple[str, float, bool]:
        """Update stabilizer with a new raw prediction.

        Stability gates:
        1. Majority vote within the sliding window must agree on a single label.
        2. Average confidence in the window must exceed the confidence threshold.
        3. The winning label must appear in at least `min_stable_consecutive_frames`
           of the most recent frames.

        Args:
            label: Raw predicted class label string.
            confidence: Raw predicted confidence (0.0 to 1.0).

        Returns:
            Tuple of (stable_label, stable_confidence, is_stable_bool).
        """
        self._window.append(label)
        self._confidence_window.append(confidence)
        self._history.append((label, confidence))
        self.total_predictions += 1

        # Majority vote from window
        vote_counts = Counter(self._window)
        majority_label, majority_votes = vote_counts.most_common(1)[0]

        # Average confidence across window entries matching the majority label
        matching_confidences = [
            c for lbl, c in zip(self._window, self._confidence_window) if lbl == majority_label
        ]
        avg_confidence = sum(matching_confidences) / len(matching_confidences)

        # Confidence gate
        if avg_confidence < self.confidence_threshold:
            return (
                self._committed_label or majority_label,
                avg_confidence,
                False,
            )

        # Consecutive stable frame gate
        if label == majority_label:
            if self._current_stable_label == majority_label:
                self._consecutive_count += 1
            else:
                self._current_stable_label = majority_label
                self._consecutive_count = 1
        else:
            self._consecutive_count = 0
            self._current_stable_label = majority_label

        is_stable = self._consecutive_count >= self.min_consecutive

        if is_stable:
            self._committed_label = majority_label
            self.gesture_counts[majority_label] += 1

        final_label = self._committed_label if self._committed_label else majority_label
        return final_label, avg_confidence, is_stable

    def reset(self) -> None:
        """Reset all stabilizer state and session statistics."""
        self._window.clear()
        self._confidence_window.clear()
        self._current_stable_label = None
        self._consecutive_count = 0
        self._committed_label = None
        self._history.clear()
        self.total_predictions = 0
        self.gesture_counts = Counter()

    def get_history(self) -> List[Tuple[str, float]]:
        """Return the last N prediction history entries.

        Returns:
            List of (label, confidence) tuples in chronological order.
        """
        return list(self._history)

    def get_most_frequent_gesture(self) -> str:
        """Return the most frequently predicted stable gesture label.

        Returns:
            Most common gesture label string or 'N/A' if no predictions made.
        """
        if not self.gesture_counts:
            return "N/A"
        return self.gesture_counts.most_common(1)[0][0]

    def get_window_distribution(self) -> Dict[str, float]:
        """Return current window prediction label distribution as percentage.

        Returns:
            Dictionary of {label: percentage_float}.
        """
        if not self._window:
            return {}
        counts = Counter(self._window)
        total = len(self._window)
        return {lbl: count / total for lbl, count in counts.items()}
