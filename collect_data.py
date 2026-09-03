"""
Interactive Facial Dataset Collection Utility.

Uses OpenCV webcam stream + MediaPipe Face Landmarker to extract normalized
geometric and blendshape features in real time, appending them to data/facial_dataset.csv.

Shortcuts:
  1 = happy
  2 = sad
  3 = angry
  4 = surprised
  5 = neutral
  6 = fear
  7 = disgust
  SPACE = Start / Pause continuous recording
  R     = Record a single frame sample
  C     = Show sample distribution breakdown in console
  Q     = Save and Quit
"""

import os
import sys
import time
import argparse
import csv
import warnings
import cv2
import numpy as np

# Suppress protobuf symbol_database deprecation warning emitted by MediaPipe internals
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")
warnings.filterwarnings("ignore", message=".*SymbolDatabase.GetPrototype.*")

from src.face_detector import FaceDetector
from src.facial_features import extract_facial_features, FEATURE_NAMES, FEATURE_DIMENSION
from src.preprocessing import (
    DEFAULT_DATASET_PATH,
    EMOTION_LABELS,
    generate_synthetic_dataset
)

KEY_TO_EMOTION = {
    ord('1'): "happy",
    ord('2'): "sad",
    ord('3'): "angry",
    ord('4'): "surprised",
    ord('5'): "neutral",
    ord('6'): "fear",
    ord('7'): "disgust",
}


def draw_hud(
    frame: np.ndarray,
    current_label: str,
    recording: bool,
    sample_counts: dict,
    face_detected: bool,
    fps: float
) -> np.ndarray:
    """Draw status HUD overlay on the webcam frame."""
    h, w, _ = frame.shape
    overlay = frame.copy()

    # Semi-transparent top bar
    cv2.rectangle(overlay, (0, 0), (w, 100), (20, 20, 20), -1)
    # Semi-transparent bottom bar
    cv2.rectangle(overlay, (0, h - 45), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Face status & FPS
    face_str = "Face: detected" if face_detected else "Face: NO FACE"
    face_color = (0, 255, 0) if face_detected else (0, 0, 255)
    cv2.putText(frame, face_str, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, face_color, 2)
    cv2.putText(frame, f"FPS: {int(fps)}", (220, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)

    # Recording Status
    rec_str = "[* RECORDING]" if recording else "[PAUSED - Press SPACE]"
    rec_color = (0, 0, 255) if recording else (180, 180, 180)
    cv2.putText(frame, rec_str, (330, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, rec_color, 2)

    # Current label & samples
    current_count = sample_counts.get(current_label, 0)
    total_samples = sum(sample_counts.values())
    cv2.putText(
        frame,
        f"Current label: {current_label.upper()}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )
    cv2.putText(
        frame,
        f"Samples: {current_count} (Total: {total_samples})",
        (350, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    # Hotkey instructions at the bottom
    help_text = "1:Hap 2:Sad 3:Ang 4:Sur 5:Neu 6:Fea 7:Dis | SPACE:Record | R:Single | Q:Save & Quit"
    cv2.putText(frame, help_text, (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

    return frame


def run_collector(output_path: str = DEFAULT_DATASET_PATH, camera_id: int = 0):
    """Run interactive webcam data collection mode."""
    print("=" * 60)
    print("FACIAL DATASET COLLECTION MODE")
    print("=" * 60)
    print(f"Facial feature vector: {FEATURE_DIMENSION} dimensions")
    print(f"Saving dataset to: {output_path}")
    print("Keyboard shortcuts:")
    print("  1 = happy | 2 = sad | 3 = angry | 4 = surprised")
    print("  5 = neutral | 6 = fear | 7 = disgust")
    print("  SPACE = Toggle continuous recording")
    print("  R     = Record single sample")
    print("  C     = Show sample distribution")
    print("  Q     = Quit and save\n")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Initialize CSV if it doesn't exist
    file_exists = os.path.exists(output_path)
    sample_counts = {emotion: 0 for emotion in EMOTION_LABELS}

    if file_exists:
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                for row in reader:
                    if row and row[0] in sample_counts:
                        sample_counts[row[0]] += 1
            print(f"Loaded existing dataset with {sum(sample_counts.values())} total samples:")
            for emo, cnt in sample_counts.items():
                print(f"  {emo:<11}: {cnt}")
            print()
        except Exception as e:
            print(f"Note: Could not read existing dataset: {e}")

    # Initialize detector
    try:
        detector = FaceDetector()
    except Exception as e:
        print(f"[Error] Failed to initialize FaceDetector: {e}")
        return

    # Open Webcam
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"[Error] Could not open webcam with ID {camera_id}.")
        print("Ensure your camera is connected and not in use by another program.")
        detector.close()
        return

    current_label = "neutral"
    recording = False
    csv_file = open(output_path, "a", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)

    if not file_exists or os.path.getsize(output_path) == 0:
        csv_writer.writerow(["label"] + FEATURE_NAMES)
        csv_file.flush()

    prev_time = time.time()
    fps = 30.0
    recorded_this_session = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[Warning] Failed to read frame from webcam. Retrying...")
                time.sleep(0.1)
                continue

            # Calculate FPS
            curr_time = time.time()
            time_diff = curr_time - prev_time
            if time_diff > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / time_diff)
            prev_time = curr_time

            # Flip frame horizontally for intuitive mirror view
            frame = cv2.flip(frame, 1)

            # Detect face
            landmarks, blendshapes = detector.detect(frame)
            face_detected = landmarks is not None

            # Visualize landmarks
            if face_detected:
                frame = detector.draw_landmarks(frame, landmarks)

            # Record sample if enabled and face is present
            if recording and face_detected:
                vector, _ = extract_facial_features(landmarks, blendshapes)
                csv_writer.writerow([current_label] + [f"{v:.5f}" for v in vector])
                csv_file.flush()
                sample_counts[current_label] += 1
                recorded_this_session += 1

            # Render HUD
            frame = draw_hud(frame, current_label, recording, sample_counts, face_detected, fps)
            cv2.imshow("Facial Dataset Collection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q') or key == 27:
                break
            elif key in KEY_TO_EMOTION:
                current_label = KEY_TO_EMOTION[key]
                print(f"\n[Label Changed] Current label: {current_label.upper()} | Samples recorded: {sample_counts[current_label]}")
            elif key == ord(' '):
                recording = not recording
                status = "STARTED" if recording else "PAUSED"
                print(f"[Recording {status}] Current label: {current_label.upper()} | Samples recorded: {sample_counts[current_label]}")
            elif key == ord('r') or key == ord('R'):
                if face_detected:
                    vector, _ = extract_facial_features(landmarks, blendshapes)
                    csv_writer.writerow([current_label] + [f"{v:.5f}" for v in vector])
                    csv_file.flush()
                    sample_counts[current_label] += 1
                    recorded_this_session += 1
                    print(f"[Single Sample Recorded] {current_label.upper()} -> {sample_counts[current_label]}")
                else:
                    print("[Warning] Cannot record: No face detected.")
            elif key == ord('c') or key == ord('C'):
                print("\n--- Current Dataset Distribution ---")
                for emo in EMOTION_LABELS:
                    print(f"  {emo:<11}: {sample_counts.get(emo, 0)}")
                print("------------------------------------\n")

    finally:
        csv_file.close()
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        print(f"\n[Finished] Session complete. {recorded_this_session} samples recorded in this session.")
        print(f"Dataset successfully saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect Facial Mood Training Data")
    parser.add_argument("--output", default=DEFAULT_DATASET_PATH, help="Path to output CSV")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument("--generate-synthetic", action="store_true", help="Generate synthetic baseline dataset")
    parser.add_argument("--samples", type=int, default=150, help="Samples per class for synthetic generation")
    args = parser.parse_args()

    if args.generate_synthetic:
        generate_synthetic_dataset(num_samples_per_class=args.samples, output_path=args.output)
    else:
        run_collector(output_path=args.output, camera_id=args.camera)
