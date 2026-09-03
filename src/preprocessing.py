"""
Data Preprocessing, Validation, and Dataset Utilities.

Handles dataset loading, feature alignment with FEATURE_NAMES, NaN/Inf handling,
class imbalance checking, feature scaling, and synthetic baseline generation.
"""

import os
from typing import Tuple, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.facial_features import FEATURE_NAMES, FEATURE_DIMENSION

EMOTION_LABELS = [
    "happy",
    "sad",
    "angry",
    "surprised",
    "neutral",
    "fear",
    "disgust"
]

DEFAULT_DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "facial_dataset.csv"
)


def validate_dataset_format(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate that the DataFrame has a 'label' column and all required FEATURE_NAMES.
    """
    if "label" not in df.columns:
        return False, "Missing 'label' column in dataset."

    missing_cols = [f for f in FEATURE_NAMES if f not in df.columns]
    if missing_cols:
        return False, f"Missing {len(missing_cols)} feature columns: {missing_cols[:5]}..."

    return True, "Dataset format is valid."


def get_dataset_distribution(df: pd.DataFrame) -> pd.Series:
    """Return counts of each emotion class in the dataset."""
    return df["label"].value_counts()


def print_distribution_report(df: pd.DataFrame) -> None:
    """
    Print the dataset distribution in the required format and warn if imbalanced.
    """
    counts = get_dataset_distribution(df)
    print("\nDataset distribution\n")
    for emotion in EMOTION_LABELS:
        count = counts.get(emotion, 0)
        print(f"{emotion:<11} {count:>5}")

    # Check for unlisted classes
    for emotion, count in counts.items():
        if emotion not in EMOTION_LABELS:
            print(f"{emotion:<11} {count:>5} (non-standard)")

    print()
    if len(counts) > 0:
        max_count = counts.max()
        min_count = counts.min()
        ratio = max_count / max(min_count, 1)
        if ratio > 2.0:
            print(f"[WARNING] Dataset is imbalanced! Imbalance ratio: {ratio:.1f}x "
                  f"(max: {max_count}, min: {min_count}). Using class weighting.\n")


def load_dataset(csv_path: str = DEFAULT_DATASET_PATH) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Load dataset from CSV, sanitize NaN/Infs, and return (df, X, y).
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. Run collect_data.py to collect samples "
            f"or run with --generate-baseline to create a seed dataset."
        )

    df = pd.read_csv(csv_path)
    is_valid, msg = validate_dataset_format(df)
    if not is_valid:
        raise ValueError(f"Dataset validation failed: {msg}")

    # Filter for known emotions
    df = df[df["label"].isin(EMOTION_LABELS)].copy()
    if len(df) == 0:
        raise ValueError("Dataset contains zero valid emotion samples.")

    # Extract ordered feature matrix
    X = df[FEATURE_NAMES].values.astype(np.float32)

    # Sanitize NaNs/Infs
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = df["label"].values

    return df, X, y


def generate_synthetic_dataset(
    num_samples_per_class: int = 150,
    output_path: str = DEFAULT_DATASET_PATH,
    random_state: int = 42
) -> str:
    """
    Generate a biologically realistic synthetic feature dataset for all 7 emotions.
    Grounds features in FACS (Facial Action Coding System) and MediaPipe blendshapes.
    """
    rng = np.random.default_rng(random_state)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    rows = []

    # Characteristic profiles for each emotion (baseline deviations)
    profiles: Dict[str, Dict[str, Tuple[float, float]]] = {
        "happy": {
            "mouthSmileLeft": (0.75, 0.12),
            "mouthSmileRight": (0.75, 0.12),
            "mouth_corner_elevation": (0.28, 0.05),
            "smile_curvature": (0.75, 0.10),
            "mouth_width": (1.30, 0.08),
            "eyeSquintLeft": (0.45, 0.10),
            "eyeSquintRight": (0.45, 0.10),
            "cheekPuff": (0.35, 0.08),
            "left_eye_aspect_ratio": (0.26, 0.03),
            "right_eye_aspect_ratio": (0.26, 0.03),
        },
        "sad": {
            "mouthFrownLeft": (0.65, 0.12),
            "mouthFrownRight": (0.65, 0.12),
            "mouth_corner_elevation": (-0.18, 0.04),
            "smile_curvature": (-0.25, 0.06),
            "browInnerUp": (0.70, 0.12),
            "browDownLeft": (0.40, 0.10),
            "browDownRight": (0.40, 0.10),
            "left_brow_to_eye_dist": (0.42, 0.04),
            "right_brow_to_eye_dist": (0.42, 0.04),
            "eye_closure_indicator": (0.30, 0.08),
        },
        "angry": {
            "browDownLeft": (0.80, 0.10),
            "browDownRight": (0.80, 0.10),
            "brow_inner_distance": (0.22, 0.03),
            "eyebrow_elevation": (-0.12, 0.04),
            "eyeSquintLeft": (0.60, 0.10),
            "eyeSquintRight": (0.60, 0.10),
            "mouthPucker": (0.40, 0.10),
            "noseSneerLeft": (0.45, 0.10),
            "noseSneerRight": (0.45, 0.10),
            "mouth_corner_elevation": (-0.08, 0.03),
        },
        "surprised": {
            "jawOpen": (0.85, 0.10),
            "mouth_openness": (0.65, 0.10),
            "mouth_aspect_ratio": (0.75, 0.12),
            "mouth_height": (0.80, 0.10),
            "browOuterUpLeft": (0.80, 0.10),
            "browOuterUpRight": (0.80, 0.10),
            "browInnerUp": (0.75, 0.10),
            "eyebrow_elevation": (0.30, 0.05),
            "eyeWideLeft": (0.80, 0.10),
            "eyeWideRight": (0.80, 0.10),
            "left_eye_aspect_ratio": (0.38, 0.04),
            "right_eye_aspect_ratio": (0.38, 0.04),
        },
        "neutral": {
            "mouth_corner_elevation": (0.02, 0.03),
            "smile_curvature": (0.10, 0.04),
            "mouth_width": (0.95, 0.05),
            "mouth_height": (0.18, 0.03),
            "mouth_aspect_ratio": (0.19, 0.03),
            "left_eye_aspect_ratio": (0.30, 0.03),
            "right_eye_aspect_ratio": (0.30, 0.03),
            "eyebrow_elevation": (0.05, 0.03),
            "eyebrow_asymmetry": (0.02, 0.01),
        },
        "fear": {
            "eyeWideLeft": (0.85, 0.10),
            "eyeWideRight": (0.85, 0.10),
            "browInnerUp": (0.80, 0.10),
            "mouthStretchLeft": (0.65, 0.10),
            "mouthStretchRight": (0.65, 0.10),
            "jawOpen": (0.45, 0.10),
            "mouth_openness": (0.38, 0.08),
            "left_eye_aspect_ratio": (0.36, 0.03),
            "right_eye_aspect_ratio": (0.36, 0.03),
            "mouth_corner_elevation": (-0.05, 0.04),
        },
        "disgust": {
            "noseSneerLeft": (0.85, 0.10),
            "noseSneerRight": (0.85, 0.10),
            "mouthUpperUpLeft": (0.75, 0.10),
            "mouthUpperUpRight": (0.75, 0.10),
            "browDownLeft": (0.55, 0.10),
            "browDownRight": (0.55, 0.10),
            "eyeSquintLeft": (0.50, 0.10),
            "eyeSquintRight": (0.50, 0.10),
            "mouth_corner_elevation": (-0.12, 0.04),
            "mouth_width": (1.05, 0.06),
        }
    }

    # Baseline default values for neutral face
    neutral_baselines: Dict[str, float] = {
        "left_eye_openness": 0.28,
        "right_eye_openness": 0.28,
        "average_eye_openness": 0.28,
        "left_eye_aspect_ratio": 0.30,
        "right_eye_aspect_ratio": 0.30,
        "eye_closure_indicator": 0.10,
        "left_brow_to_eye_dist": 0.45,
        "right_brow_to_eye_dist": 0.45,
        "eyebrow_elevation": 0.05,
        "eyebrow_asymmetry": 0.02,
        "brow_inner_distance": 0.40,
        "mouth_width": 0.95,
        "mouth_height": 0.18,
        "mouth_aspect_ratio": 0.19,
        "mouth_openness": 0.05,
        "mouth_corner_elevation": 0.02,
        "smile_curvature": 0.10,
        "lip_separation": 0.05,
        "face_width": 2.10,
        "face_height": 2.45,
        "face_aspect_ratio": 1.16,
        "head_roll": 0.0,
        "head_yaw": 0.0,
        "head_pitch": 0.0,
    }
    # All blendshapes default to ~0.02 for neutral
    for feat in FEATURE_NAMES:
        if feat not in neutral_baselines:
            neutral_baselines[feat] = 0.02

    for emotion, num_samples in [
        ("happy", num_samples_per_class),
        ("sad", int(num_samples_per_class * 0.9)),
        ("angry", int(num_samples_per_class * 0.85)),
        ("surprised", int(num_samples_per_class * 0.75)),
        ("neutral", int(num_samples_per_class * 1.1)),
        ("fear", int(num_samples_per_class * 0.65)),
        ("disgust", int(num_samples_per_class * 0.60)),
    ]:
        prof = profiles[emotion]
        for _ in range(num_samples):
            row = {"label": emotion}
            # Add moderate head motion variance
            curr_roll = float(rng.normal(0.0, 0.08))
            curr_yaw = float(rng.normal(0.0, 0.09))
            curr_pitch = float(rng.normal(0.0, 0.08))

            for feat in FEATURE_NAMES:
                if feat == "head_roll":
                    val = curr_roll
                elif feat == "head_yaw":
                    val = curr_yaw
                elif feat == "head_pitch":
                    val = curr_pitch
                elif feat in prof:
                    mean, std = prof[feat]
                    val = float(rng.normal(mean, std))
                else:
                    base = neutral_baselines[feat]
                    val = float(rng.normal(base, base * 0.15 + 0.01))

                # Clamp blendshapes to [0, 1]
                if feat in profiles["happy"] or "mouth" in feat or "brow" in feat or "eye" in feat or "jaw" in feat or "cheek" in feat or "nose" in feat:
                    if not feat.startswith("head_") and feat != "eyebrow_elevation" and feat != "mouth_corner_elevation" and feat != "smile_curvature":
                        val = float(np.clip(val, 0.0, 1.0))

                row[feat] = val
            rows.append(row)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(output_path, index=False)
    print(f"[SUCCESS] Generated synthetic dataset with {len(df_out)} samples at: {output_path}")
    return output_path
