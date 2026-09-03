"""
Model Download Script for MediaPipe Face Landmarker.

This script explicitly downloads the official MediaPipe Face Landmarker task model
(face_landmarker.task) into the models/ directory.

MediaPipe Tasks require a pre-trained .task model containing:
1. Face detector model
2. 478 3D landmark regressor model
3. 52 facial blendshape coefficients model

Model source:
Google MediaPipe Models Repository
URL: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
"""

import os
import sys
import urllib.request
import shutil

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODELS_DIR, "face_landmarker.task")


def download_face_landmarker(force: bool = False) -> str:
    """Download the face_landmarker.task file if not already present."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    if os.path.exists(MODEL_PATH) and not force:
        size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        print(f"[OK] Model already exists at: {MODEL_PATH} ({size_mb:.2f} MB)")
        return MODEL_PATH

    print(f"Downloading MediaPipe Face Landmarker model...")
    print(f"Source: {MODEL_URL}")
    print(f"Destination: {MODEL_PATH}")

    # Use curl if available (robust on Windows/Linux) or fallback to urllib
    try:
        import subprocess
        result = subprocess.run(
            ["curl.exe", "-fSL", "--progress-bar", "-o", MODEL_PATH, MODEL_URL],
            check=True
        )
        if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1000000:
            size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
            print(f"[SUCCESS] Download completed ({size_mb:.2f} MB).")
            return MODEL_PATH
    except Exception as e:
        print(f"curl download attempt failed ({e}), falling back to urllib...")

    req = urllib.request.Request(
        MODEL_URL,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    )
    with urllib.request.urlopen(req) as response, open(MODEL_PATH, "wb") as out_file:
        shutil.copyfileobj(response, out_file)

    size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"[SUCCESS] Download completed ({size_mb:.2f} MB).")
    return MODEL_PATH


if __name__ == "__main__":
    force_flag = "--force" in sys.argv
    download_face_landmarker(force=force_flag)
