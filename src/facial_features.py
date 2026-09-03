"""
Facial Feature Extraction Module.

Extracts normalized geometric features from 478 MediaPipe 3D face landmarks
and combines them with MediaPipe facial blendshapes to create a robust,
scale-invariant, translation-invariant feature vector for mood prediction.
"""

import math
from typing import List, Dict, Tuple, Optional, Any, Union
import numpy as np

# Landmark Indices
LEFT_EYE_OUTER = 33
LEFT_EYE_INNER = 133
LEFT_EYE_TOP1 = 160
LEFT_EYE_TOP2 = 158
LEFT_EYE_BOT1 = 144
LEFT_EYE_BOT2 = 153
LEFT_EYE_CENTER = 159

RIGHT_EYE_OUTER = 263
RIGHT_EYE_INNER = 362
RIGHT_EYE_TOP1 = 385
RIGHT_EYE_TOP2 = 387
RIGHT_EYE_BOT1 = 380
RIGHT_EYE_BOT2 = 373
RIGHT_EYE_CENTER = 386

LEFT_BROW_CENTER = 105
LEFT_BROW_INNER = 70
LEFT_BROW_OUTER = 107
RIGHT_BROW_CENTER = 334
RIGHT_BROW_INNER = 300
RIGHT_BROW_OUTER = 336

NOSE_TIP = 1
NOSE_BRIDGE = 168
CHIN = 152
FOREHEAD = 10
TEMPLE_LEFT = 234
TEMPLE_RIGHT = 454
JAW_LEFT = 58
JAW_RIGHT = 288

MOUTH_CORNER_LEFT = 61
MOUTH_CORNER_RIGHT = 291
UPPER_LIP_TOP = 0
LOWER_LIP_BOTTOM = 17
UPPER_LIP_INNER = 13
LOWER_LIP_INNER = 14

# Selected high-value blendshape keys from MediaPipe
SELECTED_BLENDSHAPES = [
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "mouthPucker",
    "mouthStretchLeft",
    "mouthStretchRight",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "jawOpen",
    "browDownLeft",
    "browDownRight",
    "browInnerUp",
    "browOuterUpLeft",
    "browOuterUpRight",
    "eyeBlinkLeft",
    "eyeBlinkRight",
    "eyeWideLeft",
    "eyeWideRight",
    "eyeSquintLeft",
    "eyeSquintRight",
    "cheekPuff",
    "noseSneerLeft",
    "noseSneerRight",
]

# Canonical ordered list of all features in the final vector
GEOMETRIC_FEATURE_NAMES = [
    # Eyes
    "left_eye_openness",
    "right_eye_openness",
    "average_eye_openness",
    "left_eye_aspect_ratio",
    "right_eye_aspect_ratio",
    "eye_closure_indicator",
    # Eyebrows
    "left_brow_to_eye_dist",
    "right_brow_to_eye_dist",
    "eyebrow_elevation",
    "eyebrow_asymmetry",
    "brow_inner_distance",
    # Mouth
    "mouth_width",
    "mouth_height",
    "mouth_aspect_ratio",
    "mouth_openness",
    "mouth_corner_elevation",
    "smile_curvature",
    "lip_separation",
    # Face Geometry & Pose
    "face_width",
    "face_height",
    "face_aspect_ratio",
    "head_roll",
    "head_yaw",
    "head_pitch",
]

FEATURE_NAMES = GEOMETRIC_FEATURE_NAMES + SELECTED_BLENDSHAPES
FEATURE_DIMENSION = len(FEATURE_NAMES)


def _euclidean_dist_2d(p1: Any, p2: Any) -> float:
    """Calculate 2D Euclidean distance between two landmark objects or tuples."""
    x1, y1 = (p1.x, p1.y) if hasattr(p1, "x") else (p1[0], p1[1])
    x2, y2 = (p2.x, p2.y) if hasattr(p2, "x") else (p2[0], p2[1])
    return math.hypot(x1 - x2, y1 - y2)


def _euclidean_dist_3d(p1: Any, p2: Any) -> float:
    """Calculate 3D Euclidean distance between two landmark objects or tuples."""
    x1, y1, z1 = (p1.x, p1.y, getattr(p1, "z", 0.0)) if hasattr(p1, "x") else (p1[0], p1[1], p1[2] if len(p1) > 2 else 0.0)
    x2, y2, z2 = (p2.x, p2.y, getattr(p2, "z", 0.0)) if hasattr(p2, "x") else (p2[0], p2[1], p2[2] if len(p2) > 2 else 0.0)
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)


def _calculate_ear(outer: Any, inner: Any, top1: Any, bot1: Any, top2: Any, bot2: Any) -> float:
    """
    Calculate Eye Aspect Ratio (EAR) based on Soukupová & Čech (2016).
    """
    v1 = _euclidean_dist_2d(top1, bot1)
    v2 = _euclidean_dist_2d(top2, bot2)
    h = _euclidean_dist_2d(outer, inner)
    if h < 1e-6:
        return 0.0
    return (v1 + v2) / (2.0 * h)


def extract_facial_features(
    landmarks: Optional[List[Any]],
    blendshapes: Optional[Dict[str, float]] = None,
    image_shape: Optional[Tuple[int, int]] = None
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Extract normalized geometric features and blendshape features.

    Parameters:
        landmarks: List of 478 normalized 3D landmarks (with .x, .y, .z) or None
        blendshapes: Dictionary mapping blendshape name to float score [0.0, 1.0] or None
        image_shape: Optional (height, width) for coordinate scaling if needed

    Returns:
        feature_vector: Fixed-dimension 1D numpy array of shape (FEATURE_DIMENSION,)
        feature_dict: Dictionary mapping feature names to their float values
    """
    feature_dict: Dict[str, float] = {}

    # Handle missing or degenerate landmarks
    if landmarks is None or len(landmarks) < 468:
        # Return all zeros if no landmarks detected
        zeros = np.zeros(FEATURE_DIMENSION, dtype=np.float32)
        zero_dict = {name: 0.0 for name in FEATURE_NAMES}
        return zeros, zero_dict

    try:
        # Scale normalization baseline: Inter-Ocular Distance (IOD)
        # Distance between eye centers (or pupils)
        p_left_eye = landmarks[LEFT_EYE_CENTER]
        p_right_eye = landmarks[RIGHT_EYE_CENTER]
        iod = _euclidean_dist_2d(p_left_eye, p_right_eye)
        if iod < 1e-5:
            iod = 1e-5  # Prevent division by zero

        # Vertical face size baseline: Forehead to Chin
        p_forehead = landmarks[FOREHEAD]
        p_chin = landmarks[CHIN]
        face_len = _euclidean_dist_2d(p_forehead, p_chin)
        if face_len < 1e-5:
            face_len = 1e-5

        # -------------------------------------------------------------
        # 1. Eyes
        # -------------------------------------------------------------
        left_ear = _calculate_ear(
            landmarks[LEFT_EYE_OUTER],
            landmarks[LEFT_EYE_INNER],
            landmarks[LEFT_EYE_TOP1],
            landmarks[LEFT_EYE_BOT1],
            landmarks[LEFT_EYE_TOP2],
            landmarks[LEFT_EYE_BOT2],
        )
        right_ear = _calculate_ear(
            landmarks[RIGHT_EYE_OUTER],
            landmarks[RIGHT_EYE_INNER],
            landmarks[RIGHT_EYE_TOP1],
            landmarks[RIGHT_EYE_BOT1],
            landmarks[RIGHT_EYE_TOP2],
            landmarks[RIGHT_EYE_BOT2],
        )
        left_eye_openness = _euclidean_dist_2d(landmarks[LEFT_EYE_TOP1], landmarks[LEFT_EYE_BOT1]) / iod
        right_eye_openness = _euclidean_dist_2d(landmarks[RIGHT_EYE_TOP1], landmarks[RIGHT_EYE_BOT1]) / iod
        avg_eye_openness = (left_eye_openness + right_eye_openness) / 2.0
        avg_ear = (left_ear + right_ear) / 2.0
        # Closure indicator: 1.0 when completely closed, 0.0 when fully open
        eye_closure = float(np.clip(1.0 - (avg_ear / 0.35), 0.0, 1.0))

        feature_dict["left_eye_openness"] = float(left_eye_openness)
        feature_dict["right_eye_openness"] = float(right_eye_openness)
        feature_dict["average_eye_openness"] = float(avg_eye_openness)
        feature_dict["left_eye_aspect_ratio"] = float(left_ear)
        feature_dict["right_eye_aspect_ratio"] = float(right_ear)
        feature_dict["eye_closure_indicator"] = float(eye_closure)

        # -------------------------------------------------------------
        # 2. Eyebrows
        # -------------------------------------------------------------
        left_brow_to_eye = _euclidean_dist_2d(landmarks[LEFT_BROW_CENTER], landmarks[LEFT_EYE_CENTER]) / iod
        right_brow_to_eye = _euclidean_dist_2d(landmarks[RIGHT_BROW_CENTER], landmarks[RIGHT_EYE_CENTER]) / iod
        
        # Eyebrow elevation relative to nose bridge
        p_nose_bridge = landmarks[NOSE_BRIDGE]
        left_brow_elev = (p_nose_bridge.y - landmarks[LEFT_BROW_CENTER].y) / face_len
        right_brow_elev = (p_nose_bridge.y - landmarks[RIGHT_BROW_CENTER].y) / face_len
        eyebrow_elevation = (left_brow_elev + right_brow_elev) / 2.0
        eyebrow_asymmetry = abs(left_brow_elev - right_brow_elev)
        brow_inner_dist = _euclidean_dist_2d(landmarks[LEFT_BROW_INNER], landmarks[RIGHT_BROW_INNER]) / iod

        feature_dict["left_brow_to_eye_dist"] = float(left_brow_to_eye)
        feature_dict["right_brow_to_eye_dist"] = float(right_brow_to_eye)
        feature_dict["eyebrow_elevation"] = float(eyebrow_elevation)
        feature_dict["eyebrow_asymmetry"] = float(eyebrow_asymmetry)
        feature_dict["brow_inner_distance"] = float(brow_inner_dist)

        # -------------------------------------------------------------
        # 3. Mouth
        # -------------------------------------------------------------
        p_mouth_left = landmarks[MOUTH_CORNER_LEFT]
        p_mouth_right = landmarks[MOUTH_CORNER_RIGHT]
        mouth_w = _euclidean_dist_2d(p_mouth_left, p_mouth_right)
        p_lip_top = landmarks[UPPER_LIP_TOP]
        p_lip_bot = landmarks[LOWER_LIP_BOTTOM]
        mouth_h = _euclidean_dist_2d(p_lip_top, p_lip_bot)

        mar = mouth_h / (mouth_w + 1e-6)
        normalized_mouth_w = mouth_w / iod
        normalized_mouth_h = mouth_h / iod

        p_inner_top = landmarks[UPPER_LIP_INNER]
        p_inner_bot = landmarks[LOWER_LIP_INNER]
        inner_lip_sep = _euclidean_dist_2d(p_inner_top, p_inner_bot) / iod

        # Lip corner elevation / smile curvature
        # In image coords, y grows downwards; so higher corner => smaller y => positive elevation
        mouth_mid_y = (p_lip_top.y + p_lip_bot.y) / 2.0
        left_corner_elev = (mouth_mid_y - p_mouth_left.y) / iod
        right_corner_elev = (mouth_mid_y - p_mouth_right.y) / iod
        mouth_corner_elevation = (left_corner_elev + right_corner_elev) / 2.0

        # Smile curvature: distance between mouth corners and nose tip vs mouth center
        smile_curvature = (normalized_mouth_w * 0.5) + mouth_corner_elevation

        feature_dict["mouth_width"] = float(normalized_mouth_w)
        feature_dict["mouth_height"] = float(normalized_mouth_h)
        feature_dict["mouth_aspect_ratio"] = float(mar)
        feature_dict["mouth_openness"] = float(inner_lip_sep)
        feature_dict["mouth_corner_elevation"] = float(mouth_corner_elevation)
        feature_dict["smile_curvature"] = float(smile_curvature)
        feature_dict["lip_separation"] = float(inner_lip_sep)

        # -------------------------------------------------------------
        # 4. Face Geometry & Approximate Head Pose
        # -------------------------------------------------------------
        face_w = _euclidean_dist_2d(landmarks[TEMPLE_LEFT], landmarks[TEMPLE_RIGHT]) / iod
        normalized_face_h = face_len / iod
        face_aspect_ratio = face_len / (face_w * iod + 1e-6)

        # Approximate Head Roll (angle of eyes relative to horizontal)
        dx = p_right_eye.x - p_left_eye.x
        dy = p_right_eye.y - p_left_eye.y
        head_roll = math.atan2(dy, dx)  # In radians

        # Approximate Head Yaw (asymmetry of nose tip to temple/jaw)
        p_nose = landmarks[NOSE_TIP]
        dist_nose_left = _euclidean_dist_2d(p_nose, landmarks[TEMPLE_LEFT])
        dist_nose_right = _euclidean_dist_2d(p_nose, landmarks[TEMPLE_RIGHT])
        head_yaw = (dist_nose_right - dist_nose_left) / (dist_nose_right + dist_nose_left + 1e-6)

        # Approximate Head Pitch (ratio of nose-forehead vs nose-chin)
        dist_nose_forehead = _euclidean_dist_2d(p_nose, p_forehead)
        dist_nose_chin = _euclidean_dist_2d(p_nose, p_chin)
        head_pitch = (dist_nose_chin - dist_nose_forehead) / (dist_nose_chin + dist_nose_forehead + 1e-6)

        feature_dict["face_width"] = float(face_w)
        feature_dict["face_height"] = float(normalized_face_h)
        feature_dict["face_aspect_ratio"] = float(face_aspect_ratio)
        feature_dict["head_roll"] = float(head_roll)
        feature_dict["head_yaw"] = float(head_yaw)
        feature_dict["head_pitch"] = float(head_pitch)

    except Exception as e:
        # Safe fallback on unexpected math error
        for name in GEOMETRIC_FEATURE_NAMES:
            if name not in feature_dict:
                feature_dict[name] = 0.0

    # -------------------------------------------------------------
    # 5. Blendshapes Integration
    # -------------------------------------------------------------
    for bs_name in SELECTED_BLENDSHAPES:
        if blendshapes is not None and bs_name in blendshapes:
            feature_dict[bs_name] = float(np.clip(blendshapes[bs_name], 0.0, 1.0))
        else:
            feature_dict[bs_name] = 0.0

    # Assemble fixed-dimension numpy vector in guaranteed deterministic order
    vector = np.empty(FEATURE_DIMENSION, dtype=np.float32)
    for i, name in enumerate(FEATURE_NAMES):
        val = feature_dict.get(name, 0.0)
        # Sanitize NaN / Inf
        if math.isnan(val) or math.isinf(val):
            val = 0.0
        vector[i] = val
        feature_dict[name] = float(val)

    return vector, feature_dict
