"""
Face Detection and Landmark Tracking Module using MediaPipe Face Landmarker.

Uses the modern MediaPipe Tasks Vision API (FaceLandmarker) with the
face_landmarker.task model. Extracts 478 3D normalized landmarks and 52 facial
blendshape scores.
"""

import os
import warnings
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import cv2

# Suppress protobuf symbol_database deprecation warning emitted by MediaPipe internals
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")
warnings.filterwarnings("ignore", message=".*SymbolDatabase.GetPrototype.*")

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
except ImportError:
    mp = None
    python = None
    vision = None


DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "face_landmarker.task"
)

# Key landmark indices in MediaPipe Face Mesh / Face Landmarker (478 points):
# Eyes
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]  # [outer, top1, top2, inner, bot1, bot2]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
LEFT_PUPIL = 468   # Iris center if available, fallback 159
RIGHT_PUPIL = 473  # Iris center if available, fallback 386

# Eyebrows
LEFT_EYEBROW = [70, 63, 105, 66, 107]
RIGHT_EYEBROW = [300, 293, 334, 296, 336]

# Lips / Mouth
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88]
LIPS_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
MOUTH_CORNER_LEFT = 61
MOUTH_CORNER_RIGHT = 291
UPPER_LIP_TOP = 0
LOWER_LIP_BOTTOM = 17
UPPER_LIP_INNER = 13
LOWER_LIP_INNER = 14

# Face geometry anchors
CHIN = 152
FOREHEAD = 10
NOSE_TIP = 1
NOSE_BRIDGE = 168
TEMPLE_LEFT = 234
TEMPLE_RIGHT = 454
JAW_LEFT = 58
JAW_RIGHT = 288


class FaceDetector:
    """
    Wrapper around MediaPipe Face Landmarker task.
    """

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, num_faces: int = 1):
        self.model_path = model_path
        self.num_faces = num_faces
        self.detector = None

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"MediaPipe face landmarker model not found at:\n  {self.model_path}\n"
                f"Please run 'python download_model.py' to download the required model."
            )

        self._init_detector()

    def _init_detector(self):
        """Initialize the MediaPipe Face Landmarker."""
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=self.num_faces,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def detect(self, frame_bgr: np.ndarray) -> Tuple[Optional[List[Any]], Optional[Dict[str, float]]]:
        """
        Process a BGR image frame and return (landmarks, blendshapes).

        Returns:
            landmarks: List of normalized landmark objects (with .x, .y, .z) or None
            blendshapes: Dictionary mapping blendshape name to float score [0.0, 1.0] or None
        """
        if self.detector is None or frame_bgr is None or frame_bgr.size == 0:
            return None, None

        # MediaPipe requires RGB format
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        try:
            result = self.detector.detect(mp_image)
        except Exception as e:
            print(f"[Warning] Detection error: {e}")
            return None, None

        if not result.face_landmarks or len(result.face_landmarks) == 0:
            return None, None

        # Return the first detected face's landmarks
        face_landmarks = result.face_landmarks[0]

        # Extract blendshapes dictionary if available
        blendshapes_dict = None
        if result.face_blendshapes and len(result.face_blendshapes) > 0:
            blendshapes_dict = {
                category.category_name: float(category.score)
                for category in result.face_blendshapes[0]
            }

        return face_landmarks, blendshapes_dict

    def draw_landmarks(
        self,
        frame_bgr: np.ndarray,
        landmarks: Optional[List[Any]],
        draw_contours: bool = True,
        draw_points: bool = True,
        color: Tuple[int, int, int] = (255, 0, 0)  # Pure Blue in BGR
    ) -> np.ndarray:
        """
        Draw facial landmark contours and key points onto the given frame in blue only.
        """
        if landmarks is None:
            return frame_bgr

        h, w, _ = frame_bgr.shape

        def to_pixel(lm):
            return int(lm.x * w), int(lm.y * h)

        # Draw contour connections in blue
        contour_groups = [
            (LEFT_EYE_INDICES, color, True),
            (RIGHT_EYE_INDICES, color, True),
            (LEFT_EYEBROW, color, False),
            (RIGHT_EYEBROW, color, False),
            (LIPS_OUTER, color, True),
            (LIPS_INNER, color, True),
        ]

        if draw_contours:
            for indices, c, closed in contour_groups:
                pts = []
                for idx in indices:
                    if idx < len(landmarks):
                        pts.append(to_pixel(landmarks[idx]))
                if len(pts) > 1:
                    pts_arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame_bgr, [pts_arr], closed, c, 1, cv2.LINE_AA)

        if draw_points:
            # Highlight key anchor points in blue
            key_points = [
                (NOSE_TIP, color, 3),
                (CHIN, color, 2),
                (FOREHEAD, color, 2),
                (MOUTH_CORNER_LEFT, color, 2),
                (MOUTH_CORNER_RIGHT, color, 2),
            ]
            for idx, c, radius in key_points:
                if idx < len(landmarks):
                    px, py = to_pixel(landmarks[idx])
                    cv2.circle(frame_bgr, (px, py), radius, c, -1, cv2.LINE_AA)

        return frame_bgr

    def close(self):
        """Release MediaPipe resources."""
        if self.detector is not None:
            self.detector.close()
            self.detector = None
