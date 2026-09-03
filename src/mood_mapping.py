"""
Mood Vector Mapping Module.

Translates discrete emotion probability distributions into continuous 4D
mood representations:
  - valence:   [0.0, 1.0] (pleasantness / positive affect)
  - energy:    [0.0, 1.0] (arousal / physiological activation)
  - calmness:  [0.0, 1.0] (tranquility / low distress)
  - darkness:  [0.0, 1.0] (somberness / gloom / negative tension)

Theoretical Basis:
Based on Russell's Circumplex Model of Affect, Thayer's Bi-dimensional
Activation Model, and affective computing literature.
"""

from typing import Dict, Optional, List
import numpy as np

# Grounded 4D coordinate mapping for each discrete emotion
# Coordinates are normalized to [0.0, 1.0]
EMOTION_TO_MOOD_MAP: Dict[str, Dict[str, float]] = {
    "happy": {
        "valence": 0.85,
        "energy": 0.70,
        "calmness": 0.60,
        "darkness": 0.05
    },
    "sad": {
        "valence": 0.15,
        "energy": 0.20,
        "calmness": 0.35,
        "darkness": 0.85
    },
    "angry": {
        "valence": 0.15,
        "energy": 0.85,
        "calmness": 0.10,
        "darkness": 0.80
    },
    "surprised": {
        "valence": 0.65,
        "energy": 0.85,
        "calmness": 0.25,
        "darkness": 0.15
    },
    "neutral": {
        "valence": 0.50,
        "energy": 0.40,
        "calmness": 0.80,
        "darkness": 0.20
    },
    "fear": {
        "valence": 0.15,
        "energy": 0.80,
        "calmness": 0.10,
        "darkness": 0.85
    },
    "disgust": {
        "valence": 0.20,
        "energy": 0.60,
        "calmness": 0.20,
        "darkness": 0.70
    }
}


def compute_mood_vector_approach_a(probabilities: Dict[str, float]) -> Dict[str, float]:
    """
    Approach A: Classification-derived mood mapping.
    Calculates the expected continuous mood vector by taking the probability-weighted
    sum of canonical mood coordinates across all emotion classes.

    Parameters:
        probabilities: Dictionary mapping emotion names (e.g. 'happy', 'sad') to float probabilities.

    Returns:
        mood_vector: Dictionary with 'valence', 'energy', 'calmness', 'darkness' in [0.0, 1.0].
    """
    valence = 0.0
    energy = 0.0
    calmness = 0.0
    darkness = 0.0
    total_weight = 0.0

    for emotion, prob in probabilities.items():
        clean_emotion = emotion.lower().strip()
        if clean_emotion in EMOTION_TO_MOOD_MAP:
            coords = EMOTION_TO_MOOD_MAP[clean_emotion]
            weight = max(float(prob), 0.0)
            valence += weight * coords["valence"]
            energy += weight * coords["energy"]
            calmness += weight * coords["calmness"]
            darkness += weight * coords["darkness"]
            total_weight += weight

    if total_weight > 1e-6:
        valence /= total_weight
        energy /= total_weight
        calmness /= total_weight
        darkness /= total_weight
    else:
        # Default neutral fallback if probabilities are all zero
        valence = 0.50
        energy = 0.40
        calmness = 0.80
        darkness = 0.20

    return {
        "valence": float(np.clip(valence, 0.0, 1.0)),
        "energy": float(np.clip(energy, 0.0, 1.0)),
        "calmness": float(np.clip(calmness, 0.0, 1.0)),
        "darkness": float(np.clip(darkness, 0.0, 1.0)),
    }


class DirectMoodRegressor:
    """
    Approach B: Direct regression model stub for when human-annotated continuous
    mood ratings are available in the training data.
    """

    def __init__(self, regressor_model=None):
        self.regressor = regressor_model

    def predict(self, feature_vector: np.ndarray) -> Dict[str, float]:
        if self.regressor is None:
            raise NotImplementedError(
                "Direct continuous regressor not trained. Use compute_mood_vector_approach_a."
            )
        preds = self.regressor.predict(feature_vector.reshape(1, -1))[0]
        return {
            "valence": float(np.clip(preds[0], 0.0, 1.0)),
            "energy": float(np.clip(preds[1], 0.0, 1.0)),
            "calmness": float(np.clip(preds[2], 0.0, 1.0)),
            "darkness": float(np.clip(preds[3], 0.0, 1.0)),
        }
