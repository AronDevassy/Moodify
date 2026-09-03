"""
Unit and Integration Test Suite for Facial Expression and Mood Prediction Pipeline.

Tests:
1. Mock synthetic landmark generation
2. No face detected handling
3. One face detected feature extraction
4. Invalid / degenerate landmarks handling
5. Feature vector dimension and name alignment
6. Scale and translation invariance (normalization)
7. NaN and Inf sanitization
8. Model bundle persistence and loading
9. End-to-end prediction output structure and mood vector range [0.0, 1.0]
10. Low confidence thresholding producing 'UNCERTAIN'
11. Temporal smoother behavior and jitter dampening
"""

import os
import unittest
from types import SimpleNamespace
import numpy as np

from src.facial_features import (
    extract_facial_features,
    FEATURE_NAMES,
    FEATURE_DIMENSION,
    GEOMETRIC_FEATURE_NAMES,
    SELECTED_BLENDSHAPES
)
from src.mood_mapping import compute_mood_vector_approach_a, EMOTION_TO_MOOD_MAP
from src.predict_model import (
    FaceMoodPredictor,
    TemporalSmoother,
    CONFIDENCE_THRESHOLD,
    SMOOTHING_WINDOW
)
from src.preprocessing import generate_synthetic_dataset, load_dataset


def create_mock_face(scale: float = 1.0, offset_x: float = 0.0, offset_y: float = 0.0):
    """Generate 478 mock 3D normalized landmarks with standard facial topology."""
    landmarks = []
    for i in range(478):
        x = (0.5 + ((i % 25) - 12) * 0.012) * scale + offset_x
        y = (0.5 + ((i // 25) - 10) * 0.015) * scale + offset_y
        z = 0.0
        landmarks.append(SimpleNamespace(x=x, y=y, z=z))

    # Fix key landmarks with anatomically sensible relative positions
    # Eyes
    landmarks[159] = SimpleNamespace(x=0.40 * scale + offset_x, y=0.38 * scale + offset_y, z=0.0)  # Left eye center
    landmarks[386] = SimpleNamespace(x=0.60 * scale + offset_x, y=0.38 * scale + offset_y, z=0.0)  # Right eye center
    landmarks[33] = SimpleNamespace(x=0.35 * scale + offset_x, y=0.38 * scale + offset_y, z=0.0)   # Left outer
    landmarks[133] = SimpleNamespace(x=0.45 * scale + offset_x, y=0.38 * scale + offset_y, z=0.0)  # Left inner
    landmarks[160] = SimpleNamespace(x=0.40 * scale + offset_x, y=0.36 * scale + offset_y, z=0.0)  # Left top1
    landmarks[144] = SimpleNamespace(x=0.40 * scale + offset_x, y=0.40 * scale + offset_y, z=0.0)  # Left bot1
    landmarks[158] = SimpleNamespace(x=0.41 * scale + offset_x, y=0.36 * scale + offset_y, z=0.0)  # Left top2
    landmarks[153] = SimpleNamespace(x=0.41 * scale + offset_x, y=0.40 * scale + offset_y, z=0.0)  # Left bot2

    landmarks[263] = SimpleNamespace(x=0.65 * scale + offset_x, y=0.38 * scale + offset_y, z=0.0)  # Right outer
    landmarks[362] = SimpleNamespace(x=0.55 * scale + offset_x, y=0.38 * scale + offset_y, z=0.0)  # Right inner
    landmarks[385] = SimpleNamespace(x=0.60 * scale + offset_x, y=0.36 * scale + offset_y, z=0.0)  # Right top1
    landmarks[380] = SimpleNamespace(x=0.60 * scale + offset_x, y=0.40 * scale + offset_y, z=0.0)  # Right bot1
    landmarks[387] = SimpleNamespace(x=0.59 * scale + offset_x, y=0.36 * scale + offset_y, z=0.0)  # Right top2
    landmarks[373] = SimpleNamespace(x=0.59 * scale + offset_x, y=0.40 * scale + offset_y, z=0.0)  # Right bot2

    # Eyebrows
    landmarks[105] = SimpleNamespace(x=0.40 * scale + offset_x, y=0.32 * scale + offset_y, z=0.0)  # Left brow center
    landmarks[70] = SimpleNamespace(x=0.44 * scale + offset_x, y=0.33 * scale + offset_y, z=0.0)   # Left brow inner
    landmarks[334] = SimpleNamespace(x=0.60 * scale + offset_x, y=0.32 * scale + offset_y, z=0.0)  # Right brow center
    landmarks[300] = SimpleNamespace(x=0.56 * scale + offset_x, y=0.33 * scale + offset_y, z=0.0)  # Right brow inner

    # Face contour anchors
    landmarks[10] = SimpleNamespace(x=0.50 * scale + offset_x, y=0.20 * scale + offset_y, z=0.0)   # Forehead
    landmarks[152] = SimpleNamespace(x=0.50 * scale + offset_x, y=0.80 * scale + offset_y, z=0.0)  # Chin
    landmarks[1] = SimpleNamespace(x=0.50 * scale + offset_x, y=0.50 * scale + offset_y, z=0.0)    # Nose tip
    landmarks[168] = SimpleNamespace(x=0.50 * scale + offset_x, y=0.35 * scale + offset_y, z=0.0)  # Nose bridge
    landmarks[234] = SimpleNamespace(x=0.28 * scale + offset_x, y=0.48 * scale + offset_y, z=0.0)  # Temple left
    landmarks[454] = SimpleNamespace(x=0.72 * scale + offset_x, y=0.48 * scale + offset_y, z=0.0)  # Temple right

    # Mouth
    landmarks[61] = SimpleNamespace(x=0.42 * scale + offset_x, y=0.65 * scale + offset_y, z=0.0)   # Left corner
    landmarks[291] = SimpleNamespace(x=0.58 * scale + offset_x, y=0.65 * scale + offset_y, z=0.0)  # Right corner
    landmarks[0] = SimpleNamespace(x=0.50 * scale + offset_x, y=0.62 * scale + offset_y, z=0.0)    # Upper lip top
    landmarks[17] = SimpleNamespace(x=0.50 * scale + offset_x, y=0.68 * scale + offset_y, z=0.0)   # Lower lip bot
    landmarks[13] = SimpleNamespace(x=0.50 * scale + offset_x, y=0.63 * scale + offset_y, z=0.0)   # Upper lip inner
    landmarks[14] = SimpleNamespace(x=0.50 * scale + offset_x, y=0.67 * scale + offset_y, z=0.0)   # Lower lip inner

    return landmarks


class TestFacialPipeline(unittest.TestCase):

    def test_no_face_detected(self):
        """Feature extraction with None input should return zero vector of correct shape."""
        vec, d = extract_facial_features(None)
        self.assertIsInstance(vec, np.ndarray)
        self.assertEqual(vec.shape, (FEATURE_DIMENSION,))
        self.assertTrue(np.all(vec == 0.0))
        self.assertEqual(len(d), FEATURE_DIMENSION)

    def test_insufficient_landmarks(self):
        """Feature extraction with truncated landmark list (<468) should return zeros."""
        truncated_lms = [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(50)]
        vec, d = extract_facial_features(truncated_lms)
        self.assertEqual(vec.shape, (FEATURE_DIMENSION,))
        self.assertTrue(np.all(vec == 0.0))

    def test_feature_vector_dimension_and_names(self):
        """Extracted vector length must strictly match FEATURE_DIMENSION and FEATURE_NAMES."""
        lms = create_mock_face()
        vec, d = extract_facial_features(lms)
        self.assertEqual(len(vec), FEATURE_DIMENSION)
        self.assertEqual(len(d), FEATURE_DIMENSION)
        self.assertEqual(list(d.keys()), FEATURE_NAMES)

    def test_nan_and_inf_prevention(self):
        """Feature extraction must sanitize NaNs and Infs."""
        lms = create_mock_face()
        # Inject extreme or degenerate coordinates
        lms[159] = SimpleNamespace(x=float("nan"), y=0.4, z=0.0)
        vec, d = extract_facial_features(lms)
        self.assertFalse(np.isnan(vec).any(), "Found NaN in feature vector!")
        self.assertFalse(np.isinf(vec).any(), "Found Inf in feature vector!")

    def test_scale_and_translation_invariance(self):
        """Geometric features must remain invariant under facial scale and position shifts."""
        face_standard = create_mock_face(scale=1.0, offset_x=0.0, offset_y=0.0)
        face_scaled_shifted = create_mock_face(scale=1.8, offset_x=0.15, offset_y=-0.08)

        vec1, _ = extract_facial_features(face_standard)
        vec2, _ = extract_facial_features(face_scaled_shifted)

        # Compare geometric distance features (indices 0 to 20)
        max_diff = np.max(np.abs(vec1[:21] - vec2[:21]))
        self.assertLess(max_diff, 1e-4, f"Scale invariance violation: max diff = {max_diff}")

    def test_blendshapes_integration(self):
        """Blendshapes dictionary values must correctly populate into the feature vector."""
        lms = create_mock_face()
        test_blendshapes = {
            "mouthSmileLeft": 0.85,
            "mouthSmileRight": 0.82,
            "jawOpen": 0.40,
            "browInnerUp": 0.65
        }
        vec, d = extract_facial_features(lms, blendshapes=test_blendshapes)
        self.assertAlmostEqual(d["mouthSmileLeft"], 0.85, places=4)
        self.assertAlmostEqual(d["mouthSmileRight"], 0.82, places=4)
        self.assertAlmostEqual(d["jawOpen"], 0.40, places=4)
        self.assertAlmostEqual(d["browInnerUp"], 0.65, places=4)

    def test_continuous_mood_mapping(self):
        """Continuous mood mapping must produce valid [0.0, 1.0] coordinates for all dimensions."""
        happy_probs = {"happy": 0.90, "neutral": 0.10}
        mood = compute_mood_vector_approach_a(happy_probs)
        for key in ["valence", "energy", "calmness", "darkness"]:
            self.assertIn(key, mood)
            self.assertGreaterEqual(mood[key], 0.0)
            self.assertLessEqual(mood[key], 1.0)
        # Happy must have high valence and low darkness
        self.assertGreater(mood["valence"], 0.70)
        self.assertLess(mood["darkness"], 0.20)

        sad_probs = {"sad": 0.95, "neutral": 0.05}
        mood_sad = compute_mood_vector_approach_a(sad_probs)
        self.assertLess(mood_sad["valence"], 0.30)
        self.assertGreater(mood_sad["darkness"], 0.70)

    def test_predictor_model_loading_and_inference(self):
        """FaceMoodPredictor must load models/face_mood_model.joblib and predict accurately."""
        predictor = FaceMoodPredictor(confidence_threshold=0.55)
        self.assertIsNotNone(predictor.model)
        self.assertEqual(len(predictor.classes), 7)

        # Test prediction on mock face
        lms = create_mock_face()
        res = predictor.predict_from_landmarks(lms)
        self.assertIsNotNone(res)
        self.assertIn(res.emotion, [c.upper() for c in predictor.classes] + ["UNCERTAIN"])
        self.assertGreaterEqual(res.confidence, 0.0)
        self.assertLessEqual(res.confidence, 1.0)
        self.assertAlmostEqual(sum(res.smoothed_probabilities.values()), 1.0, places=4)

    def test_confidence_threshold_uncertain(self):
        """If highest class probability is below CONFIDENCE_THRESHOLD, output must be UNCERTAIN."""
        predictor = FaceMoodPredictor(confidence_threshold=0.80)
        # Create an ambiguous uniform feature vector
        ambiguous_vec = np.ones(FEATURE_DIMENSION, dtype=np.float32) * 0.1
        res = predictor.predict_vector(ambiguous_vec)
        if res.confidence < 0.80:
            self.assertEqual(res.emotion, "UNCERTAIN")
            self.assertFalse(res.is_confident)

    def test_temporal_smoother_dampens_jitter(self):
        """Temporal smoother must dampen rapid single-frame class flips."""
        smoother = TemporalSmoother(window_size=10, method="ema", ema_alpha=0.20)
        classes = ["happy", "neutral", "sad"]

        p_happy = np.array([0.9, 0.1, 0.0])
        p_sad_glitch = np.array([0.0, 0.1, 0.9])

        # Feed 5 happy frames
        for _ in range(5):
            smoothed, top = smoother.update(p_happy, classes)
        self.assertEqual(top, "happy")

        # Feed 1 single glitch frame
        smoothed, top = smoother.update(p_sad_glitch, classes)
        # Smoothed top class should remain 'happy' despite 1 frame glitch!
        self.assertEqual(top, "happy", "Temporal smoother failed to prevent single-frame glitch flip!")


if __name__ == "__main__":
    unittest.main()
