"""
CLI Training Script for Facial Mood Prediction.

Compares Random Forest and Support Vector Machine (SVM) on facial feature data.
Evaluates using 5-fold Stratified Cross-Validation, Macro F1, Precision, Recall,
and Confusion Matrix.

Usage:
  python train_face_model.py
  python train_face_model.py --generate-baseline
  python train_face_model.py --data data/facial_dataset.csv --model models/face_mood_model.joblib
"""

import os
import sys
import argparse

from src.preprocessing import DEFAULT_DATASET_PATH, generate_synthetic_dataset
from src.train_model import train_and_compare_models, DEFAULT_MODEL_SAVE_PATH


def main():
    parser = argparse.ArgumentParser(description="Train Facial Mood ML Classifiers")
    parser.add_argument("--data", default=DEFAULT_DATASET_PATH, help="Path to facial dataset CSV")
    parser.add_argument("--model", default=DEFAULT_MODEL_SAVE_PATH, help="Output path for trained model bundle")
    parser.add_argument("--generate-baseline", action="store_true", help="Generate synthetic baseline dataset if missing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        if args.generate_baseline:
            print(f"Dataset not found at {args.data}. Generating synthetic baseline dataset...")
            generate_synthetic_dataset(num_samples_per_class=150, output_path=args.data, random_state=args.seed)
        else:
            print(f"[Error] Dataset not found at '{args.data}'.")
            print("To generate an initial baseline dataset, run:")
            print("  python train_face_model.py --generate-baseline")
            print("Or run collect_data.py to record from your webcam.")
            sys.exit(1)

    train_and_compare_models(
        dataset_path=args.data,
        model_save_path=args.model,
        random_state=args.seed
    )


if __name__ == "__main__":
    main()
