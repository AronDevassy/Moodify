"""
Real-Time Inference and Prediction Engine.

Loads the trained model artifact bundle (Random Forest or SVM), scales incoming
feature vectors, applies temporal probability smoothing (moving average or EMA),
enforces confidence thresholds, and computes both categorical emotion and 4D continuous mood.
"""

import os
from collections import deque, Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import joblib
import numpy as np

from src.facial_features import extract_facial_features, FEATURE_DIMENSION, FEATURE_NAMES
from src.mood_mapping import compute_mood_vector_approach_a

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "face_mood_model.joblib"
)

CONFIDENCE_THRESHOLD = 0.55
SMOOTHING_WINDOW = 10


@dataclass
class PredictionResult:
    """Structured container for facial mood inference outputs."""
    emotion: str                        # Display emotion (e.g. 'HAPPY' or 'UNCERTAIN')
    raw_emotion: str                    # Argmax emotion before confidence thresholding
    confidence: float                   # Top class confidence [0.0, 1.0]
    probabilities: Dict[str, float]     # Full probability distribution for all classes
    mood_vector: Dict[str, float]       # 4D mood vector: valence, energy, calmness, darkness
    is_confident: bool                  # True if confidence >= CONFIDENCE_THRESHOLD
    smoothed_probabilities: Dict[str, float]


class TemporalSmoother:
    """
    Temporal Smoother to mitigate frame-to-frame classification jitter.
    Supports Exponential Moving Average (EMA) and Rolling Moving Average.
    """

    def __init__(self, window_size: int = SMOOTHING_WINDOW, method: str = "ema", ema_alpha: float = 0.25):
        self.window_size = max(window_size, 1)
        self.method = method.lower()
        self.ema_alpha = ema_alpha
        self.history: deque = deque(maxlen=self.window_size)
        self.class_history: deque = deque(maxlen=self.window_size)
        self.current_ema: Optional[np.ndarray] = None

    def update(self, raw_probs: np.ndarray, classes: List[str]) -> Tuple[np.ndarray, str]:
        """
        Update smoother state with new frame probabilities and return smoothed distribution.
        """
        self.history.append(raw_probs)
        raw_top_idx = int(np.argmax(raw_probs))
        self.class_history.append(classes[raw_top_idx])

        if self.method == "ema":
            if self.current_ema is None:
                self.current_ema = np.copy(raw_probs)
            else:
                self.current_ema = (self.ema_alpha * raw_probs) + ((1.0 - self.ema_alpha) * self.current_ema)
            smoothed_probs = np.copy(self.current_ema)
        elif self.method == "majority_vote":
            # Moving average of probabilities with majority class
            smoothed_probs = np.mean(self.history, axis=0)
            majority_class = Counter(self.class_history).most_common(1)[0][0]
            # Ensure probability aligns
            return smoothed_probs, majority_class
        else:
            # Default: Simple Moving Average over window
            smoothed_probs = np.mean(self.history, axis=0)

        # Normalize to ensure sum to 1.0
        prob_sum = np.sum(smoothed_probs)
        if prob_sum > 1e-6:
            smoothed_probs /= prob_sum

        top_idx = int(np.argmax(smoothed_probs))
        top_class = classes[top_idx]
        return smoothed_probs, top_class

    def reset(self):
        """Reset smoother history (e.g. when face tracking is lost)."""
        self.history.clear()
        self.class_history.clear()
        self.current_ema = None


class FaceMoodPredictor:
    """
    End-to-end inference predictor for facial emotion and continuous mood.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        smoothing_window: int = SMOOTHING_WINDOW,
        smoothing_method: str = "ema"
    ):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.smoothing_window = smoothing_window

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model bundle not found at:\n  {self.model_path}\n"
                f"Please run 'python train_face_model.py' to train and save the model first."
            )

        # Load serialized bundle
        bundle = joblib.load(self.model_path)
        self.model = bundle["model"]
        self.model_name = bundle.get("model_name", "Classifier")
        self.scaler = bundle["scaler"]
        self.classes = list(bundle["classes"])
        self.feature_names = bundle.get("feature_names", FEATURE_NAMES)

        # Initialize smoother
        self.smoother = TemporalSmoother(window_size=smoothing_window, method=smoothing_method)

    def predict_vector(self, feature_vector: np.ndarray) -> PredictionResult:
        """
        Predict emotion and mood from a raw extracted feature vector.
        """
        # Ensure 2D shape and sanitize NaNs
        vector_clean = np.nan_to_num(feature_vector.reshape(1, -1), nan=0.0, posinf=0.0, neginf=0.0)

        # Scale features using pre-fitted scaler
        scaled_vector = self.scaler.transform(vector_clean)

        # Predict probability distribution
        if hasattr(self.model, "predict_proba"):
            raw_probs = self.model.predict_proba(scaled_vector)[0]
        else:
            # Fallback for models without direct predict_proba
            pred_class = self.model.predict(scaled_vector)[0]
            raw_probs = np.zeros(len(self.classes))
            raw_probs[self.classes.index(pred_class)] = 1.0

        # Apply temporal smoothing
        smoothed_probs, top_class = self.smoother.update(raw_probs, self.classes)

        # Build probabilities dictionary
        prob_dict = {cls_name: float(smoothed_probs[i]) for i, cls_name in enumerate(self.classes)}
        raw_prob_dict = {cls_name: float(raw_probs[i]) for i, cls_name in enumerate(self.classes)}

        top_confidence = float(np.max(smoothed_probs))
        is_confident = top_confidence >= self.confidence_threshold

        # Format display label
        if is_confident:
            display_emotion = top_class.upper()
        else:
            display_emotion = "UNCERTAIN"

        # Compute continuous 4D mood vector
        mood_vec = compute_mood_vector_approach_a(prob_dict)

        return PredictionResult(
            emotion=display_emotion,
            raw_emotion=top_class.upper(),
            confidence=top_confidence,
            probabilities=raw_prob_dict,
            smoothed_probabilities=prob_dict,
            mood_vector=mood_vec,
            is_confident=is_confident
        )

    def predict_from_landmarks(
        self,
        landmarks: Optional[List[Any]],
        blendshapes: Optional[Dict[str, float]] = None
    ) -> Optional[PredictionResult]:
        """
        Convenience wrapper extracting features and running prediction.
        Returns None if no landmarks are provided.
        """
        if landmarks is None:
            self.smoother.reset()
            return None

        vector, _ = extract_facial_features(landmarks, blendshapes)
        return self.predict_vector(vector)
