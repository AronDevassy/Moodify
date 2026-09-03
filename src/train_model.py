"""
Model Training and Evaluation Pipeline.

Trains and evaluates Random Forest and Support Vector Machine (SVM) classifiers
using stratified train/test split, cross-validation, precision, recall, F1 score,
and confusion matrix. Handles class imbalance with class_weight='balanced'.
Saves the top-performing model and preprocessing scaler to models/face_mood_model.joblib.
"""

import os
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)

from src.facial_features import FEATURE_NAMES, FEATURE_DIMENSION
from src.preprocessing import (
    load_dataset,
    print_distribution_report,
    DEFAULT_DATASET_PATH,
    EMOTION_LABELS
)

DEFAULT_MODEL_SAVE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "face_mood_model.joblib"
)


def evaluate_classifier(
    name: str,
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    classes: list,
    cv_folds: int = 5
) -> Dict[str, Any]:
    """
    Perform Stratified K-Fold cross validation and test set evaluation.
    """
    print("=" * 60)
    print(f"EVALUATING MODEL: {name.upper()}")
    print("=" * 60)

    # 1. Stratified Cross-Validation on training set
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scoring = {
        "accuracy": "accuracy",
        "macro_f1": "f1_macro",
        "weighted_f1": "f1_weighted",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro"
    }
    cv_results = cross_validate(model, X_train, y_train, cv=skf, scoring=scoring, n_jobs=-1)

    cv_macro_f1 = float(np.mean(cv_results["test_macro_f1"]))
    cv_acc = float(np.mean(cv_results["test_accuracy"]))
    print(f"5-Fold Cross Validation:")
    print(f"  CV Macro F1 : {cv_macro_f1:.4f} (+/- {np.std(cv_results['test_macro_f1']):.4f})")
    print(f"  CV Accuracy : {cv_acc:.4f} (+/- {np.std(cv_results['test_accuracy']):.4f})")

    # 2. Fit on full training set and evaluate on test set
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    test_acc = float(accuracy_score(y_test, y_pred))
    test_macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    test_weighted_f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
    test_macro_prec = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    test_macro_rec = float(recall_score(y_test, y_pred, average="macro", zero_division=0))

    print(f"\nHoldout Test Set Performance:")
    print(f"  Test Accuracy   : {test_acc:.4f}")
    print(f"  Macro Precision : {test_macro_prec:.4f}")
    print(f"  Macro Recall    : {test_macro_rec:.4f}")
    print(f"  Macro F1-Score  : {test_macro_f1:.4f}")
    print(f"  Weighted F1     : {test_weighted_f1:.4f}\n")

    print("Detailed Classification Report:")
    print(classification_report(y_test, y_pred, labels=classes, zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    print("Confusion Matrix:")
    header = "          " + " ".join([f"{c[:3]:>5}" for c in classes])
    print(header)
    for i, row in enumerate(cm):
        row_str = f"{classes[i][:8]:<10}" + " ".join([f"{val:>5}" for val in row])
        print(row_str)
    print()

    return {
        "name": name,
        "model": model,
        "cv_macro_f1": cv_macro_f1,
        "cv_accuracy": cv_acc,
        "test_macro_f1": test_macro_f1,
        "test_weighted_f1": test_weighted_f1,
        "test_accuracy": test_acc,
        "test_precision": test_macro_prec,
        "test_recall": test_macro_rec,
        "confusion_matrix": cm,
    }


def train_and_compare_models(
    dataset_path: str = DEFAULT_DATASET_PATH,
    model_save_path: str = DEFAULT_MODEL_SAVE_PATH,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Train Random Forest and SVM classifiers, compare metrics, and persist the winner.
    """
    df, X, y = load_dataset(dataset_path)
    print_distribution_report(df)

    # Stratified Train/Test split (80/20)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        stratify=y,
        random_state=random_state
    )

    # Feature Scaling: fit scaler ONLY on train to avoid data leakage
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    unique_classes = sorted(list(set(y)))

    # 1. Model 1: Random Forest Classifier
    # Uses class_weight='balanced' to handle imbalance
    rf_model = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_split=4,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1
    )
    rf_results = evaluate_classifier(
        "Random Forest Classifier",
        rf_model,
        X_train,
        y_train,
        X_test,
        y_test,
        unique_classes
    )

    # 2. Model 2: Support Vector Machine (SVM) Classifier
    # Uses RBF kernel, probability=True for confidence scoring and class_weight='balanced'
    svm_model = SVC(
        C=2.0,
        kernel="rbf",
        gamma="scale",
        probability=True,
        class_weight="balanced",
        random_state=random_state
    )
    svm_results = evaluate_classifier(
        "Support Vector Classifier (SVM)",
        svm_model,
        X_train,
        y_train,
        X_test,
        y_test,
        unique_classes
    )

    # Compare and select winning model based on Macro F1 Score
    print("=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Model':<32} {'CV Macro F1':<14} {'Test Macro F1':<15} {'Test Accuracy':<15}")
    print("-" * 75)
    print(f"{rf_results['name']:<32} {rf_results['cv_macro_f1']:<14.4f} {rf_results['test_macro_f1']:<15.4f} {rf_results['test_accuracy']:<15.4f}")
    print(f"{svm_results['name']:<32} {svm_results['cv_macro_f1']:<14.4f} {svm_results['test_macro_f1']:<15.4f} {svm_results['test_accuracy']:<15.4f}")
    print("-" * 75)

    if rf_results["test_macro_f1"] >= svm_results["test_macro_f1"]:
        winner = rf_results
        runner_up = svm_results
    else:
        winner = svm_results
        runner_up = rf_results

    print(f"\n[WINNER SELECTED]: {winner['name']} (Test Macro F1: {winner['test_macro_f1']:.4f})\n")

    # Save artifact bundle: model, scaler, feature names, classes
    os.makedirs(os.path.dirname(os.path.abspath(model_save_path)), exist_ok=True)
    bundle = {
        "model": winner["model"],
        "model_name": winner["name"],
        "scaler": scaler,
        "feature_names": FEATURE_NAMES,
        "feature_dimension": FEATURE_DIMENSION,
        "classes": list(winner["model"].classes_),
        "metrics": {
            "test_macro_f1": winner["test_macro_f1"],
            "test_accuracy": winner["test_accuracy"],
            "test_precision": winner["test_precision"],
            "test_recall": winner["test_recall"]
        }
    }
    joblib.dump(bundle, model_save_path)
    print(f"[SAVED] Exported model bundle to: {model_save_path}")

    return {
        "winner": winner,
        "rf_results": rf_results,
        "svm_results": svm_results,
        "model_path": model_save_path
    }
