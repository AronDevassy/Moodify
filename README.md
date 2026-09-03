# Computer Vision — Facial Expression & Affect / Mood Prediction

An end-to-end computer-vision and machine-learning system that analyzes webcam input, tracks 3D facial landmarks and blendshapes with MediaPipe Face Landmarker, extracts normalized scale/rotation-invariant geometric features, and predicts discrete emotional expressions alongside continuous 4D affect mood representations (`valence`, `energy`, `calmness`, `darkness`).

---

## ⚠️ Important Scientific Warning & Ethical Disclosure

> [!IMPORTANT]
> **Visible Facial Expression $\neq$ True Internal Emotional State**
> 
> This system performs **facial expression classification / affect estimation**, **NOT** mind reading or emotional ground-truth detection.
> 
> Biopsychosocial and affective science research (e.g., Barrett et al., *Psychological Science in the Public Interest*, 2019) conclusively shows that:
> - A **smile** does not guarantee internal happiness (smiles can indicate politeness, nervousness, compliance, or masking).
> - A **neutral face** does not indicate neutral internal emotions (intense internal feelings frequently occur without external motor expression).
> - A **furrowed brow** often indicates concentration, cognitive load, physical strain, or squinting against bright sunlight rather than anger.
> 
> **Key Limitations & Sources of Error:**
> 1. **Morphological Variations**: Baseline bone structure, eye shape, lip curvature, and age-related tissue laxity differ naturally across individuals.
> 2. **Cultural & Social Display Rules**: Context dictates when individuals express, suppress, or amplify visible expressions.
> 3. **Environmental Factors**: Lighting conditions, shadows, camera sensor noise, and low frame rates introduce variance in landmark tracking.
> 4. **Head Pose & Occlusion**: Out-of-plane head rotations ($> 45^\circ$), hand-to-face contact, eyeglasses, or facial hair can distort landmark detection.
> 5. **Dataset Bias**: Models trained on posed or non-diverse datasets reflect the biases and stereotypes present in their training distributions.

---

## System Architecture

```text
               Webcam Video Stream (OpenCV)
                           │
                           ▼
          Face Detection & Tracking (MediaPipe)
           (models/face_landmarker.task model)
                           │
       ┌───────────────────┴───────────────────┐
       ▼                                       ▼
478 3D Normalized Landmarks              52 ARKit Blendshapes
       │                                       │
       └───────────────────┬───────────────────┘
                           ▼
         Facial Feature Engineering (50-D Vector)
       - Scale-normalized distances (IOD baseline)
       - Eye Aspect Ratios (EAR) & Eye Openness
       - Eyebrow elevation & asymmetry
       - Mouth Aspect Ratio (MAR) & Smile curvature
       - Approximate Head Pose (Yaw, Pitch, Roll)
       - Selected Expression Blendshapes
                           │
                           ▼
       StandardScaler Feature Preprocessing & Z-Score
                           │
                           ▼
         ML Classifier (Random Forest / SVM)
        (Trained with class_weight='balanced')
                           │
                           ▼
              Raw Probability Distribution
                           │
                           ▼
         Temporal Probability Smoothing (EMA)
                           │
                           ▼
         Confidence Threshold (e.g. 0.55)
              │                          │
        (Below Thresh)             (Above Thresh)
              ▼                          ▼
      Emotion: UNCERTAIN          Emotion: [HAPPY, SAD, etc.]
              │                          │
              └────────────┬─────────────┘
                           ▼
          Continuous 4D Mood Vector Generator
        (Valence, Energy, Calmness, Darkness)
```

---

## Output Representations

### 1. Discrete Emotion Classification (7 Classes)
- `happy`
- `sad`
- `angry`
- `surprised`
- `neutral`
- `fear`
- `disgust`
- *(Special status: `UNCERTAIN` when top prediction confidence is below `CONFIDENCE_THRESHOLD = 0.55`)*

### 2. Continuous 4D Affect / Mood Representation
Mapped to $[0.0, 1.0]$ based on Russell's Circumplex Model of Affect:
- **Valence** ($0.0 - 1.0$): Degree of pleasantness / positivity (e.g., Happy: $0.85$, Sad: $0.15$).
- **Energy** ($0.0 - 1.0$): Degree of physiological activation / arousal (e.g., Angry/Surprised: $0.85$, Sad: $0.20$).
- **Calmness** ($0.0 - 1.0$): Degree of relaxation and tranquility (e.g., Neutral: $0.80$, Fear/Angry: $0.10$).
- **Darkness** ($0.0 - 1.0$): Degree of somber tension or gloom (e.g., Sad/Fear: $0.85$, Happy: $0.05$).

---

## Directory Structure

```text
moodpredictor/
│
├── models/
│   ├── face_landmarker.task     # Google MediaPipe Face Landmarker task model
│   └── face_mood_model.joblib   # Trained ML model artifact bundle (scaler + classifier)
│
├── data/
│   └── facial_dataset.csv       # Training dataset of numerical feature vectors
│
├── src/
│   ├── __init__.py
│   ├── face_detector.py         # MediaPipe Face Landmarker wrapper & contour renderer
│   ├── facial_features.py       # Scale-invariant geometric feature engineering & blendshapes
│   ├── preprocessing.py         # Dataset validation, class imbalance check, synthetic generator
│   ├── mood_mapping.py          # Continuous 4D mood coordinate mapping (Approach A & B)
│   ├── train_model.py           # Random Forest vs SVM training, StratifiedKFold, and metrics
│   └── predict_model.py         # Inference engine with temporal smoothing & confidence
│
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py         # Comprehensive unit tests with synthetic landmark mocks
│
├── download_model.py            # Explicit model downloader (no silent runtime download)
├── collect_data.py              # Interactive webcam data collection mode
├── train_face_model.py          # CLI model training & evaluation script
├── predict_face.py              # Real-time webcam facial mood analyzer HUD
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation & ML guide
```

---

## Installation & Setup

### 1. Requirements
Ensure Python 3.10+ is installed. Install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Download MediaPipe Model
The system uses the modern MediaPipe Face Landmarker Task API. To avoid silent background downloads during execution, explicitly fetch the model once:

```bash
python download_model.py
```
This places `face_landmarker.task` (~3.58 MB) into `models/`.

---

## Quickstart & Usage

### 1. Real-Time Mood Analyzer (Webcam)
Run the live facial mood analyzer:

```bash
python predict_face.py
```
*Controls:*
- Position your face in front of the camera.
- The on-screen HUD displays detected face contours, predicted emotion badge, probability bars, and 4 continuous mood gauges.
- Terminal outputs periodically:
```text
-----------------------------------------
FACIAL MOOD ANALYZER
-----------------------------------------

Emotion:
HAPPY

Probabilities:
Happy      82%
Neutral    10%
Surprised   4%
Sad         2%
Angry       2%

Mood Vector:
Valence     0.84
Energy      0.67
Calmness    0.43
Darkness    0.12
-----------------------------------------
```
- Press **Q** to exit.

*Configurable Flags:*
```bash
python predict_face.py --camera 0 --confidence 0.55 --smoothing 10 --method ema
```

---

### 2. Dataset Collection Mode
Record custom personal training samples across the 7 emotion classes:

```bash
python collect_data.py
```

*Keyboard Shortcuts:*
- **1** = happy
- **2** = sad
- **3** = angry
- **4** = surprised
- **5** = neutral
- **6** = fear
- **7** = disgust
- **SPACE** = Start / Pause continuous recording
- **R** = Record a single frame sample
- **C** = Print current sample distribution in console
- **Q** = Save to `data/facial_dataset.csv` and quit

*Synthetic Baseline Generation:*
To generate an immediate 800+ sample baseline dataset without recording manually:
```bash
python collect_data.py --generate-synthetic
```

---

### 3. Model Training & Comparison
Train and compare Random Forest and Support Vector Machine (SVM) classifiers:

```bash
python train_face_model.py
```

*Output includes:*
- Class distribution and imbalance warnings
- 5-Fold Stratified Cross-Validation scores
- Test set Accuracy, Macro Precision, Macro Recall, Macro F1, Weighted F1
- Multi-class Confusion Matrix
- Automated selection and export of the winning model to `models/face_mood_model.joblib`

---

### 4. Running Automated Tests
Run the mock-based unit and integration test suite:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```
Verifies:
- Graceful handling of missing faces (`None`)
- Invalid / degenerate landmark tolerance
- Exact feature dimension matching (`50` features)
- Scale & translation invariance (2x scale shift test)
- NaN / Inf sanitization
- Model serialization & inference
- Confidence thresholding (`UNCERTAIN`)
- Temporal smoother jitter dampening

---

## Teach Me the ML: Comprehensive Pipeline Explanation

### 1. Why MediaPipe?
Direct deep-learning models on raw camera video (e.g. CNNs or Vision Transformers) require substantial GPU acceleration and are vulnerable to background clutter, lighting changes, skin tone artifacts, and camera distance shifts.

MediaPipe separates **face perception** from **emotion classification**:
- MediaPipe's lightweight TFLite model detects face bounding boxes and localizes **478 3D landmarks** plus **52 blendshape coefficients** in real-time on CPU at 30+ FPS.
- It compresses millions of raw RGB pixels into structured, semantic geometric spatial points.

### 2. What are Landmarks vs. Blendshapes?
- **3D Landmarks**: Normalized Cartesian coordinates $(x, y, z)$ describing anatomical face contours (eyelids, pupils, lips, eyebrows, jawline, forehead, nose).
- **Blendshapes**: Facial Action Coding System (FACS) inspired scalar coefficients in $[0.0, 1.0]$. They describe muscular action units such as `mouthSmileLeft`, `browDownRight`, `jawOpen`, and `eyeBlink`. Blendshapes isolate expressive muscle activation independent of head pose.

### 3. Why Raw Coordinates are Problematic
Passing raw $(x, y)$ coordinates directly to a machine learning classifier fails in practice because:
1. **Translation sensitivity**: Moving left or right changes all raw coordinates, causing false predictions.
2. **Scale sensitivity**: Sitting closer or further from the webcam changes coordinate spans.
3. **Curse of Dimensionality**: 478 points $\times$ 3 coordinates = 1,434 numbers. Classifiers overfit quickly on non-expressive head locations rather than facial muscle movement.

### 4. What Feature Engineering Does
Feature engineering translates raw points into **invariant anatomical ratios and curvatures**:
- **Eye Aspect Ratio (EAR)**: $\frac{\|p_{top1} - p_{bot1}\| + \|p_{top2} - p_{bot2}\|}{2 \times \|p_{inner} - p_{outer}\|}$. Invariant to scale and rotation.
- **Inter-Ocular Normalization (IOD)**: All distance features (mouth width, brow elevation, lip separation) are divided by the distance between pupils ($\|p_{left\_eye} - p_{right\_eye}\|$). If you move closer to the camera, both your mouth width and your eye distance scale by the same factor, leaving the ratio invariant!
- **Smile Curvature**: Compares lip corner elevation against mouth center height relative to IOD.
- **Head Pose Estimation**: Derives yaw, pitch, and roll to account for head tilt without mistaking a tilted head for a facial expression.

### 5. How Random Forest Works
- An ensemble of decision trees (e.g., 150 trees) trained on random subsets of the data and features (bagging).
- Each tree splits features at specific thresholds (e.g., `if smile_curvature > 0.42 and eye_aspect_ratio < 0.28`).
- The ensemble averages predictions across all trees to produce well-calibrated class probabilities ($P(\text{happy})$, etc.).
- Robust to non-linear feature interactions and outliers.

### 6. How Support Vector Machine (SVM) Works
- Constructs maximum-margin hyperplanes separating classes in high-dimensional space.
- Uses the **Radial Basis Function (RBF) kernel** $K(x, x') = \exp(-\gamma \|x - x'\|^2)$ to map non-linearly separable facial geometric features into higher dimensions where clear linear boundaries exist.
- Calibrated using Platt scaling (`probability=True`) to yield smooth probabilities.

### 7. Why Training Data Matters & Class Imbalance
If an emotion dataset has 800 `neutral` faces and only 50 `disgust` faces, a naive model can achieve $94\%$ accuracy simply by guessing `neutral` on every frame!
- We use **Stratified Splits** so each train/test partition maintains exact class proportions.
- We use **`class_weight='balanced'`** in both Random Forest and SVM. This penalizes mistakes on minority classes inversely proportional to their frequencies:
$$w_j = \frac{N}{K \cdot n_j}$$
- We evaluate using **Macro F1-Score** (which treats all classes equally) rather than accuracy alone.

### 8. What Overfitting Means
Overfitting occurs when an ML model memorizes superficial patterns of the training data (e.g., a specific person's eyebrow shape or background lighting) rather than true generalizable expression signals.
- **Remedies Applied**:
  - Distance normalization by face size.
  - Z-score normalization with `StandardScaler` fitted strictly on training data.
  - Cross-validation with 5 folds.
  - Tree depth constraints in Random Forest.

### 9. Why Temporal Smoothing Helps
Classifying video frames independently produces high-frequency visual flicker (e.g. a momentary blink or lighting artifact causes a sudden frame flip `happy` $\to$ `neutral` $\to$ `happy`).
- We implement **Exponential Moving Average (EMA)** on the probability vectors:
$$P_{smoothed}^{(t)} = \alpha \cdot P_{raw}^{(t)} + (1 - \alpha) \cdot P_{smoothed}^{(t-1)}$$
- This creates smooth, natural transitions while preserving immediate responsiveness.

### 10. Why Emotion Prediction from Faces is Inherently Uncertain
Human faces are communication devices, not transparent windows into internal mental states. A person can smile sarcastically, scowl in deep concentration, or mask anxiety behind a calm exterior. By pairing discrete predictions with **confidence thresholds** (`UNCERTAIN`) and **continuous multidimensional mood vectors** (`valence`, `energy`, `calmness`, `darkness`), the system models affective tendency rather than making brittle, deterministic claims.
