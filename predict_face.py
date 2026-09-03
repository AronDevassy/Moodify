"""
Real-Time Facial Expression and Mood Analyzer.

Captures webcam frames, extracts normalized facial geometry and blendshapes
using MediaPipe Face Landmarker, applies temporal probability smoothing,
and renders real-time predictions for 7 emotion classes and continuous 4D mood.

Usage:
  python predict_face.py
  python predict_face.py --camera 0 --confidence 0.55 --smoothing 10
"""

import os
import sys
import time
import argparse
import warnings
from typing import Optional
import cv2
import numpy as np

# Suppress protobuf symbol_database deprecation warning emitted by MediaPipe internals
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")
warnings.filterwarnings("ignore", message=".*SymbolDatabase.GetPrototype.*")

from src.face_detector import FaceDetector
from src.predict_model import (
    FaceMoodPredictor,
    CONFIDENCE_THRESHOLD,
    SMOOTHING_WINDOW,
    PredictionResult
)

# Visual color mapping for emotions (BGR format for OpenCV)
EMOTION_COLORS = {
    "HAPPY": (40, 200, 40),       # Green
    "SAD": (220, 120, 30),        # Blue
    "ANGRY": (30, 30, 220),       # Red
    "SURPRISED": (0, 215, 255),   # Gold / Yellow
    "NEUTRAL": (180, 180, 180),   # Silver Gray
    "FEAR": (180, 50, 180),       # Purple / Violet
    "DISGUST": (30, 140, 110),    # Olive / Rust
    "UNCERTAIN": (0, 140, 255)    # Orange
}


def print_console_report(res: PredictionResult):
    """Print clean terminal report formatted exactly as specified."""
    print("\n-----------------------------------------")
    print("FACIAL MOOD ANALYZER")
    print("-----------------------------------------")
    print(f"\nEmotion:\n{res.emotion}\n")

    print("Probabilities:")
    sorted_probs = sorted(res.smoothed_probabilities.items(), key=lambda x: x[1], reverse=True)
    for emo, p in sorted_probs:
        print(f"{emo.capitalize():<10} {int(p * 100):>3}%")

    print("\nMood Vector:")
    mv = res.mood_vector
    print(f"Valence    {mv['valence']:>5.2f}")
    print(f"Energy     {mv['energy']:>5.2f}")
    print(f"Calmness   {mv['calmness']:>5.2f}")
    print(f"Darkness   {mv['darkness']:>5.2f}")
    print("-----------------------------------------")


def draw_hud(
    frame: np.ndarray,
    res: Optional[PredictionResult],
    face_detected: bool,
    fps: float,
    confidence_thresh: float
) -> np.ndarray:
    """Render rich on-screen HUD overlay with emotion, probability bars, and 4D mood gauges."""
    h, w, _ = frame.shape
    overlay = frame.copy()

    # Left sidebar panel for statistics
    sidebar_w = 280
    cv2.rectangle(overlay, (0, 0), (sidebar_w, h), (18, 18, 18), -1)

    # Top status banner
    cv2.rectangle(overlay, (sidebar_w, 0), (w, 55), (25, 25, 25), -1)
    # Bottom warning banner
    cv2.rectangle(overlay, (sidebar_w, h - 35), (w, h), (25, 25, 25), -1)

    # Alpha blend overlay with background
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Status header
    face_str = "Face: detected" if face_detected else "Face: NO FACE"
    face_color = (0, 255, 0) if face_detected else (0, 0, 255)
    cv2.putText(frame, face_str, (sidebar_w + 20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, face_color, 2)
    cv2.putText(frame, f"FPS: {int(fps)}", (sidebar_w + 220, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2)
    cv2.putText(frame, "Press 'Q' to quit", (w - 180, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1)

    # Bottom scientific affect warning
    warning_text = "Scientific notice: Visible facial expression estimate - not direct internal emotion"
    cv2.putText(frame, warning_text, (sidebar_w + 15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1)

    # Left Sidebar content: Title
    cv2.putText(frame, "MOOD ANALYZER", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 215, 255), 2)
    cv2.line(frame, (15, 45), (sidebar_w - 15, 45), (60, 60, 60), 1)

    if res is not None and face_detected:
        color = EMOTION_COLORS.get(res.emotion, (200, 200, 200))

        # Big Emotion Badge
        cv2.putText(frame, "PREDICTED EMOTION", (15, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        cv2.putText(frame, res.emotion, (15, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.95, color, 2)

        conf_str = f"Confidence: {int(res.confidence * 100)}%"
        cv2.putText(frame, conf_str, (15, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)

        # -------------------------------------------------------------
        # Emotion Probabilities section
        # -------------------------------------------------------------
        cv2.line(frame, (15, 142), (sidebar_w - 15, 142), (50, 50, 50), 1)
        cv2.putText(frame, "PROBABILITIES", (15, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        sorted_probs = sorted(res.smoothed_probabilities.items(), key=lambda x: x[1], reverse=True)
        bar_y = 185
        max_bar_w = 120

        for emo, prob in sorted_probs:
            emo_cap = emo.capitalize()
            # Emotion name
            cv2.putText(frame, f"{emo_cap[:7]:<7}", (15, bar_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 210, 210), 1)
            # Bar background
            cv2.rectangle(frame, (85, bar_y - 9), (85 + max_bar_w, bar_y + 2), (40, 40, 40), -1)
            # Bar fill
            fill_w = int(max_bar_w * prob)
            bar_color = EMOTION_COLORS.get(emo.upper(), (200, 200, 200))
            if fill_w > 0:
                cv2.rectangle(frame, (85, bar_y - 9), (85 + fill_w, bar_y + 2), bar_color, -1)
            # Percent text
            cv2.putText(frame, f"{int(prob * 100)}%", (215, bar_y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1)
            bar_y += 24

        # -------------------------------------------------------------
        # Continuous Mood Vector Section
        # -------------------------------------------------------------
        cv2.line(frame, (15, bar_y + 5), (sidebar_w - 15, bar_y + 5), (50, 50, 50), 1)
        bar_y += 25
        cv2.putText(frame, "CONTINUOUS MOOD (0.0-1.0)", (15, bar_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        bar_y += 25

        mood_items = [
            ("Valence", res.mood_vector["valence"], (0, 220, 100)),
            ("Energy", res.mood_vector["energy"], (0, 150, 255)),
            ("Calmness", res.mood_vector["calmness"], (255, 180, 50)),
            ("Darkness", res.mood_vector["darkness"], (80, 80, 220)),
        ]

        for mood_name, val, m_color in mood_items:
            cv2.putText(frame, f"{mood_name:<8}", (15, bar_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 210, 210), 1)
            cv2.rectangle(frame, (85, bar_y - 9), (85 + max_bar_w, bar_y + 2), (40, 40, 40), -1)
            m_fill_w = int(max_bar_w * val)
            if m_fill_w > 0:
                cv2.rectangle(frame, (85, bar_y - 9), (85 + m_fill_w, bar_y + 2), m_color, -1)
            cv2.putText(frame, f"{val:.2f}", (215, bar_y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1)
            bar_y += 24

    else:
        cv2.putText(frame, "NO FACE DETECTED", (15, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 220), 2)
        cv2.putText(frame, "Position your face", (15, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1)
        cv2.putText(frame, "in front of the camera", (15, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1)

    return frame


def run_predictor(
    camera_id: int = 0,
    confidence_thresh: float = CONFIDENCE_THRESHOLD,
    smoothing_window: int = SMOOTHING_WINDOW,
    smoothing_method: str = "ema"
):
    """Run real-time webcam facial mood predictor."""
    print("=" * 60)
    print("INITIALIZING REAL-TIME FACIAL MOOD ANALYZER")
    print("=" * 60)

    # Initialize Face Landmarker detector
    try:
        detector = FaceDetector()
    except Exception as e:
        print(f"[Error] Failed to initialize FaceDetector: {e}")
        return

    # Initialize ML Inference Predictor
    try:
        predictor = FaceMoodPredictor(
            confidence_threshold=confidence_thresh,
            smoothing_window=smoothing_window,
            smoothing_method=smoothing_method
        )
        print(f"[Ready] Loaded Model: {predictor.model_name}")
        print(f"[Config] Confidence Threshold: {confidence_thresh:.2f}")
        print(f"[Config] Temporal Smoothing: {smoothing_method.upper()} (window: {smoothing_window})")
    except Exception as e:
        print(f"[Error] Failed to initialize FaceMoodPredictor: {e}")
        detector.close()
        return

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"[Error] Could not open webcam device {camera_id}.")
        detector.close()
        return

    prev_time = time.time()
    last_print_time = time.time()
    fps = 30.0

    print("\nLive webcam feed started. Press 'Q' in window to exit.\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[Warning] Webcam read error. Retrying...")
                time.sleep(0.05)
                continue

            # Calculate FPS
            curr_time = time.time()
            dt = curr_time - prev_time
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)
            prev_time = curr_time

            # Flip horizontally for natural mirror view
            frame = cv2.flip(frame, 1)

            # Detect face & landmarks
            landmarks, blendshapes = detector.detect(frame)
            face_detected = landmarks is not None

            pred_res: Optional[PredictionResult] = None
            if face_detected:
                # Draw facial mesh contours
                frame = detector.draw_landmarks(frame, landmarks)
                # Run prediction with temporal smoothing
                pred_res = predictor.predict_from_landmarks(landmarks, blendshapes)
            else:
                predictor.smoother.reset()

            # Render rich on-screen HUD
            frame = draw_hud(frame, pred_res, face_detected, fps, confidence_thresh)
            cv2.imshow("Facial Expression & Mood Analyzer", frame)

            # Periodic terminal output (every 1.5 seconds)
            if pred_res is not None and (curr_time - last_print_time) >= 1.5:
                print_console_report(pred_res)
                last_print_time = curr_time

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q') or key == 27:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        print("\n[Exited] Facial Mood Analyzer stopped cleanly.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time Facial Expression and Mood Analyzer")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--confidence", type=float, default=CONFIDENCE_THRESHOLD, help="Confidence threshold")
    parser.add_argument("--smoothing", type=int, default=SMOOTHING_WINDOW, help="Temporal smoothing window size")
    parser.add_argument("--method", choices=["ema", "moving_average", "majority_vote"], default="ema", help="Smoothing method")
    args = parser.parse_args()

    run_predictor(
        camera_id=args.camera,
        confidence_thresh=args.confidence,
        smoothing_window=args.smoothing,
        smoothing_method=args.method
    )
