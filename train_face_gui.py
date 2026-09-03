"""
Facial Expression Dataset Collector GUI.

A standalone Tkinter GUI application for collecting, labeling, and saving
facial expression training datasets for ML model training.

Workflow:
  LIVE Mode -> START RECORDING -> RECORDING Mode -> STOP RECORDING ->
  LABELING Mode (Video Frozen) -> Select Expression Label -> SAVE / DISCARD -> LIVE Mode

Usage:
  python train_face_gui.py
"""

import os
import sys
import time
import uuid
import csv
import warnings
from enum import Enum, auto
from typing import Optional, List, Dict, Tuple, Any

import cv2
import numpy as np

# Suppress protobuf symbol_database deprecation warning emitted by MediaPipe internals
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")
warnings.filterwarnings("ignore", message=".*SymbolDatabase.GetPrototype.*")

import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

from src.face_detector import FaceDetector
from src.facial_features import extract_facial_features, FEATURE_NAMES, FEATURE_DIMENSION
from src.preprocessing import DEFAULT_DATASET_PATH, EMOTION_LABELS


# =============================================================================
# Configuration & Constants
# =============================================================================

SAMPLE_INTERVAL = 5  # Capture one feature vector every 5 frames during RECORDING
DEFAULT_CAMERA_ID = 0

# Colors for dark mode UI
COLOR_BG = "#1E1E1E"          # Main background
COLOR_CARD = "#252526"        # Panel background
COLOR_CARD_BORDER = "#3E3E42" # Panel border
COLOR_TEXT = "#CCCCCC"        # Primary text
COLOR_TEXT_MUTED = "#888888"  # Secondary text
COLOR_WHITE = "#FFFFFF"

COLOR_READY = "#007ACC"       # Ready badge color (Blue)
COLOR_RECORDING = "#D32F2F"   # Recording badge color (Red)
COLOR_STOPPED = "#FF9800"     # Stopped badge color (Amber)
COLOR_SUCCESS = "#388E3C"     # Save / Success green
COLOR_FACE_OK = "#4CAF50"     # Face detected green
COLOR_FACE_NO = "#F44336"     # No face red


class AppState(Enum):
    """Explicit state management for the data collector GUI."""
    LIVE = auto()
    RECORDING = auto()
    LABELING = auto()


class FacialDatasetGUI:
    """Tkinter-based GUI Application for Facial Expression Dataset Collection."""

    def __init__(self, root: tk.Tk, dataset_path: str = DEFAULT_DATASET_PATH, camera_id: int = DEFAULT_CAMERA_ID):
        self.root = root
        self.dataset_path = dataset_path
        self.camera_id = camera_id

        # Window configuration
        self.root.title("Facial Expression Dataset Collector")
        self.root.geometry("1120x740")
        self.root.minsize(1000, 680)
        self.root.configure(bg=COLOR_BG)

        # Application state
        self.state = AppState.LIVE
        self.selected_label: Optional[str] = None
        self.session_samples: List[np.ndarray] = []
        self.frame_counter: int = 0
        self.current_frozen_frame: Optional[np.ndarray] = None
        self.latest_frame: Optional[np.ndarray] = None

        # FPS calculation
        self.prev_time = time.time()
        self.fps = 30.0

        # Dataset directory & stats
        os.makedirs(os.path.dirname(os.path.abspath(self.dataset_path)), exist_ok=True)
        self.dataset_counts: Dict[str, int] = {emo: 0 for emo in EMOTION_LABELS}

        # Initialize detector & camera safely
        self.detector: Optional[FaceDetector] = None
        self.cap: Optional[cv2.VideoCapture] = None

        # Build UI layout
        self._init_styles()
        self._build_ui()

        # Keyboard bindings
        self.root.bind("<Key-r>", lambda e: self._on_toggle_recording())
        self.root.bind("<Key-R>", lambda e: self._on_toggle_recording())
        self.root.bind("<Key-d>", lambda e: self._on_discard())
        self.root.bind("<Key-D>", lambda e: self._on_discard())
        self.root.bind("<Key-q>", lambda e: self._on_quit())
        self.root.bind("<Key-Q>", lambda e: self._on_quit())
        self.root.protocol("WM_DELETE_WINDOW", self._on_quit)

        # Deferred startup for camera & detector initialization
        self.root.after(100, self._async_init_hardware)

    def _init_styles(self):
        """Configure ttk styles for dark mode aesthetics."""
        style = ttk.Style()
        style.theme_use("clam")

        # General dark frame
        style.configure("TFrame", background=COLOR_BG)
        style.configure("Card.TFrame", background=COLOR_CARD, relief="solid", borderwidth=1)

        # Labels
        style.configure("TLabel", background=COLOR_BG, foreground=COLOR_TEXT, font=("Segoe UI", 10))
        style.configure("Header.TLabel", background=COLOR_BG, foreground=COLOR_WHITE, font=("Segoe UI", 16, "bold"))
        style.configure("SubHeader.TLabel", background=COLOR_CARD, foreground=COLOR_WHITE, font=("Segoe UI", 11, "bold"))
        style.configure("Card.TLabel", background=COLOR_CARD, foreground=COLOR_TEXT, font=("Segoe UI", 10))
        style.configure("Muted.TLabel", background=COLOR_CARD, foreground=COLOR_TEXT_MUTED, font=("Segoe UI", 9))

        # Radio Buttons for emotion selection
        style.configure(
            "Emotion.TRadiobutton",
            background=COLOR_CARD,
            foreground=COLOR_TEXT,
            font=("Segoe UI", 10, "bold"),
            padding=6
        )
        style.map(
            "Emotion.TRadiobutton",
            background=[("selected", "#007ACC"), ("active", "#3E3E42")],
            foreground=[("selected", COLOR_WHITE), ("active", COLOR_WHITE)]
        )

    def _build_ui(self):
        """Construct main Tkinter application layout."""
        # Top Title Bar
        title_frame = ttk.Frame(self.root, padding=(15, 10))
        title_frame.pack(fill=tk.X)

        lbl_title = ttk.Label(title_frame, text="FACIAL EXPRESSION DATASET COLLECTOR", style="Header.TLabel")
        lbl_title.pack(side=tk.LEFT)

        lbl_subtitle = ttk.Label(
            title_frame,
            text="Clean GUI for capturing & labeling MediaPipe facial landmark features",
            style="Muted.TLabel"
        )
        lbl_subtitle.pack(side=tk.LEFT, padx=(15, 0), pady=(4, 0))

        # Main Layout Container (Left: Camera, Right: Control & Stats)
        main_container = ttk.Frame(self.root, padding=10)
        main_container.pack(fill=tk.BOTH, expand=True)

        # -------------------------------------------------------------
        # LEFT PANE: Camera Viewport & Live Status
        # -------------------------------------------------------------
        left_pane = ttk.Frame(main_container)
        left_pane.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Camera Display Label
        self.cam_frame_card = ttk.Frame(left_pane, style="Card.TFrame")
        self.cam_frame_card.pack(fill=tk.BOTH, expand=True)

        self.cam_canvas = tk.Label(self.cam_frame_card, bg="#000000")
        self.cam_canvas.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        # Camera Status Info Strip below viewport
        cam_info_strip = ttk.Frame(left_pane, padding=(5, 8))
        cam_info_strip.pack(fill=tk.X)

        self.lbl_face_status = tk.Label(
            cam_info_strip,
            text="Face: Detecting...",
            font=("Segoe UI", 11, "bold"),
            bg=COLOR_BG,
            fg=COLOR_FACE_OK
        )
        self.lbl_face_status.pack(side=tk.LEFT, padx=5)

        self.lbl_fps = tk.Label(
            cam_info_strip,
            text="FPS: 0",
            font=("Segoe UI", 10),
            bg=COLOR_BG,
            fg=COLOR_TEXT_MUTED
        )
        self.lbl_fps.pack(side=tk.LEFT, padx=15)

        self.lbl_interval = tk.Label(
            cam_info_strip,
            text=f"Sampling: Every {SAMPLE_INTERVAL} frames",
            font=("Segoe UI", 10),
            bg=COLOR_BG,
            fg=COLOR_TEXT_MUTED
        )
        self.lbl_interval.pack(side=tk.RIGHT, padx=5)

        # -------------------------------------------------------------
        # RIGHT PANE: Control Cards, Labeling Panel & Dataset Stats
        # -------------------------------------------------------------
        right_pane = ttk.Frame(main_container, width=380)
        right_pane.pack(side=tk.RIGHT, fill=tk.Y)
        right_pane.pack_propagate(False)

        # --- 1. Status & Recording Control Card ---
        ctrl_card = ttk.Frame(right_pane, style="Card.TFrame", padding=15)
        ctrl_card.pack(fill=tk.X, pady=(0, 10))

        lbl_ctrl_title = ttk.Label(ctrl_card, text="RECORDING CONTROLS", style="SubHeader.TLabel")
        lbl_ctrl_title.pack(anchor=tk.W, pady=(0, 8))

        # Main Recording Badge (READY / RECORDING / STOPPED)
        self.lbl_badge = tk.Label(
            ctrl_card,
            text="READY",
            font=("Segoe UI", 13, "bold"),
            bg=COLOR_READY,
            fg=COLOR_WHITE,
            padx=12,
            pady=6
        )
        self.lbl_badge.pack(fill=tk.X, pady=(0, 10))

        # Counter display
        self.lbl_sample_count = tk.Label(
            ctrl_card,
            text="Samples captured: 0",
            font=("Segoe UI", 11, "bold"),
            bg=COLOR_CARD,
            fg=COLOR_WHITE
        )
        self.lbl_sample_count.pack(anchor=tk.W, pady=(0, 10))

        # Big Toggle Button (START RECORDING / STOP RECORDING)
        self.btn_toggle_rec = tk.Button(
            ctrl_card,
            text="START RECORDING",
            font=("Segoe UI", 12, "bold"),
            bg=COLOR_READY,
            fg=COLOR_WHITE,
            activebackground="#005A9E",
            activeforeground=COLOR_WHITE,
            relief=tk.RAISED,
            bd=2,
            padx=10,
            pady=8,
            cursor="hand2",
            command=self._on_toggle_recording
        )
        self.btn_toggle_rec.pack(fill=tk.X)

        # --- 2. Labeling Panel Card ---
        self.label_card = ttk.Frame(right_pane, style="Card.TFrame", padding=15)
        self.label_card.pack(fill=tk.X, pady=(0, 10))

        lbl_label_title = ttk.Label(self.label_card, text="EXPRESSION LABELING", style="SubHeader.TLabel")
        lbl_label_title.pack(anchor=tk.W, pady=(0, 8))

        self.lbl_select_instruct = ttk.Label(
            self.label_card,
            text="Select expression for recorded session:",
            style="Muted.TLabel"
        )
        self.lbl_select_instruct.pack(anchor=tk.W, pady=(0, 8))

        # Radio Buttons grid for Emotions
        self.emotion_var = tk.StringVar(value="")
        emotion_grid_frame = ttk.Frame(self.label_card, style="Card.TFrame")
        emotion_grid_frame.pack(fill=tk.X, pady=(0, 12))

        self.emotion_radios: Dict[str, tk.Radiobutton] = {}
        for i, emo in enumerate(EMOTION_LABELS):
            row, col = divmod(i, 2)
            rbtn = tk.Radiobutton(
                emotion_grid_frame,
                text=emo.upper(),
                value=emo,
                variable=self.emotion_var,
                font=("Segoe UI", 9, "bold"),
                bg=COLOR_CARD,
                fg=COLOR_TEXT,
                selectcolor="#007ACC",
                activebackground=COLOR_CARD,
                activeforeground=COLOR_WHITE,
                indicatoron=0,
                bd=1,
                relief=tk.RAISED,
                padx=8,
                pady=6,
                cursor="hand2",
                command=self._on_emotion_selected
            )
            rbtn.grid(row=row, column=col, sticky="ew", padx=3, pady=3)
            emotion_grid_frame.columnconfigure(col, weight=1)
            self.emotion_radios[emo] = rbtn

        # Action Buttons: SAVE and DISCARD
        btn_action_frame = ttk.Frame(self.label_card, style="Card.TFrame")
        btn_action_frame.pack(fill=tk.X)

        self.btn_save = tk.Button(
            btn_action_frame,
            text="SAVE",
            font=("Segoe UI", 11, "bold"),
            bg=COLOR_SUCCESS,
            fg=COLOR_WHITE,
            activebackground="#1B5E20",
            activeforeground=COLOR_WHITE,
            relief=tk.RAISED,
            bd=2,
            pady=6,
            cursor="hand2",
            command=self._on_save
        )
        self.btn_save.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        self.btn_discard = tk.Button(
            btn_action_frame,
            text="DISCARD",
            font=("Segoe UI", 11, "bold"),
            bg=COLOR_RECORDING,
            fg=COLOR_WHITE,
            activebackground="#B71C1C",
            activeforeground=COLOR_WHITE,
            relief=tk.RAISED,
            bd=2,
            pady=6,
            cursor="hand2",
            command=self._on_discard
        )
        self.btn_discard.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(4, 0))

        # --- 3. Dataset Statistics Card ---
        stats_card = ttk.Frame(right_pane, style="Card.TFrame", padding=15)
        stats_card.pack(fill=tk.BOTH, expand=True)

        lbl_stats_title = ttk.Label(stats_card, text="DATASET STATISTICS", style="SubHeader.TLabel")
        lbl_stats_title.pack(anchor=tk.W, pady=(0, 6))

        self.lbl_total_samples = ttk.Label(stats_card, text="Total samples: 0", style="Card.TLabel")
        self.lbl_total_samples.pack(anchor=tk.W, pady=(0, 8))

        # Stats Table Frame
        self.stats_table_frame = ttk.Frame(stats_card, style="Card.TFrame")
        self.stats_table_frame.pack(fill=tk.BOTH, expand=True)

        self.stat_labels: Dict[str, ttk.Label] = {}
        for emo in EMOTION_LABELS:
            row_frame = ttk.Frame(self.stats_table_frame, style="Card.TFrame")
            row_frame.pack(fill=tk.X, pady=1)

            lbl_name = ttk.Label(row_frame, text=f"{emo.capitalize()}:", style="Card.TLabel", width=12)
            lbl_name.pack(side=tk.LEFT)

            lbl_val = ttk.Label(row_frame, text="0", style="Card.TLabel", font=("Segoe UI", 10, "bold"))
            lbl_val.pack(side=tk.RIGHT)
            self.stat_labels[emo] = lbl_val

        # --- Bottom Hotkey Status Bar ---
        footer_frame = ttk.Frame(self.root, padding=(15, 6))
        footer_frame.pack(fill=tk.X, side=tk.BOTTOM)

        lbl_hotkeys = ttk.Label(
            footer_frame,
            text="Shortcuts:  [ R ] Start/Stop Recording  |  [ D ] Discard Session  |  [ Q ] Quit Application",
            style="Muted.TLabel"
        )
        lbl_hotkeys.pack(side=tk.LEFT)

        # Set initial UI state
        self._update_ui_state()

    def _async_init_hardware(self):
        """Initialize FaceDetector and OpenCV VideoCapture."""
        # 1. Initialize FaceDetector
        try:
            self.detector = FaceDetector()
        except FileNotFoundError as fnf_err:
            messagebox.showerror("MediaPipe Model Error", str(fnf_err))
            return
        except Exception as e:
            messagebox.showerror("Face Detector Error", f"Failed to initialize FaceDetector:\n{e}")
            return

        # 2. Initialize Camera
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            messagebox.showerror(
                "Camera Error",
                f"Could not open webcam with ID {self.camera_id}.\n"
                "Please verify your camera is connected and not in use by another app."
            )
            return

        # 3. Load Dataset Distribution Statistics
        self._load_dataset_stats()

        # Start main camera loop
        self._update_camera()

    def _load_dataset_stats(self):
        """Load and compute sample distribution from data/facial_dataset.csv."""
        self.dataset_counts = {emo: 0 for emo in EMOTION_LABELS}

        if not os.path.exists(self.dataset_path) or os.path.getsize(self.dataset_path) == 0:
            self._refresh_stats_ui()
            return

        try:
            with open(self.dataset_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if not header:
                    self._refresh_stats_ui()
                    return

                # Find index of 'label' column
                label_idx = -1
                if "label" in header:
                    label_idx = header.index("label")

                if label_idx == -1:
                    # Fallback to column index 0 if header format is standard
                    label_idx = 0

                for row in reader:
                    if row and len(row) > label_idx:
                        lbl = row[label_idx].strip().lower()
                        if lbl in self.dataset_counts:
                            self.dataset_counts[lbl] += 1
        except Exception as e:
            print(f"[Warning] Error loading dataset stats: {e}")

        self._refresh_stats_ui()

    def _refresh_stats_ui(self):
        """Update dataset statistics numbers in GUI."""
        total = sum(self.dataset_counts.values())
        self.lbl_total_samples.config(text=f"Total dataset samples: {total:,}")

        for emo, count in self.dataset_counts.items():
            if emo in self.stat_labels:
                self.stat_labels[emo].config(text=f"{count:,}")

    def _update_ui_state(self):
        """Update widget states, labels, badges, and colors based on AppState."""
        if self.state == AppState.LIVE:
            self.lbl_badge.config(text="READY", bg=COLOR_READY)
            self.lbl_sample_count.config(text="Samples captured: 0")
            self.btn_toggle_rec.config(
                text="START RECORDING",
                bg=COLOR_READY,
                activebackground="#005A9E",
                state=tk.NORMAL
            )

            # Disable labeling controls
            self.lbl_select_instruct.config(foreground=COLOR_TEXT_MUTED)
            for rbtn in self.emotion_radios.values():
                rbtn.config(state=tk.DISABLED)
            self.btn_save.config(state=tk.DISABLED, bg="#2E4F2F")
            self.btn_discard.config(state=tk.DISABLED, bg="#5C2626")

        elif self.state == AppState.RECORDING:
            self.lbl_badge.config(text="🔴 RECORDING", bg=COLOR_RECORDING)
            self.lbl_sample_count.config(text=f"Samples captured: {len(self.session_samples)}")
            self.btn_toggle_rec.config(
                text="STOP RECORDING",
                bg=COLOR_RECORDING,
                activebackground="#B71C1C",
                state=tk.NORMAL
            )

            # Keep labeling controls disabled while recording
            self.lbl_select_instruct.config(foreground=COLOR_TEXT_MUTED)
            for rbtn in self.emotion_radios.values():
                rbtn.config(state=tk.DISABLED)
            self.btn_save.config(state=tk.DISABLED, bg="#2E4F2F")
            self.btn_discard.config(state=tk.DISABLED, bg="#5C2626")

        elif self.state == AppState.LABELING:
            self.lbl_badge.config(text="⏸ RECORDING STOPPED", bg=COLOR_STOPPED)
            self.lbl_sample_count.config(text=f"Samples captured: {len(self.session_samples)}")
            self.btn_toggle_rec.config(
                text="RECORDING FINISHED",
                bg="#555555",
                activebackground="#555555",
                state=tk.DISABLED
            )

            # Enable labeling controls
            self.lbl_select_instruct.config(foreground=COLOR_WHITE)
            for rbtn in self.emotion_radios.values():
                rbtn.config(state=tk.NORMAL)

            self.btn_save.config(state=tk.NORMAL, bg=COLOR_SUCCESS)
            self.btn_discard.config(state=tk.NORMAL, bg=COLOR_RECORDING)

    def _update_camera(self):
        """Main camera acquisition and rendering loop (runs via root.after)."""
        if self.cap is None or not self.cap.isOpened():
            return

        # If in LABELING state, video is frozen on current_frozen_frame
        if self.state == AppState.LABELING:
            if self.current_frozen_frame is not None:
                self._render_frame_to_canvas(self.current_frozen_frame)
            # Re-schedule frame loop to maintain GUI responsiveness without updating camera
            self.root.after(30, self._update_camera)
            return

        # Acquire frame from webcam
        ret, frame_bgr = self.cap.read()
        if not ret or frame_bgr is None:
            self.lbl_face_status.config(text="Face: Camera Error", fg=COLOR_FACE_NO)
            self.root.after(30, self._update_camera)
            return

        # FPS calculation
        curr_time = time.time()
        time_diff = curr_time - self.prev_time
        if time_diff > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / time_diff)
        self.prev_time = curr_time
        self.lbl_fps.config(text=f"FPS: {int(self.fps)}")

        # Mirror frame horizontally for user comfort
        frame_bgr = cv2.flip(frame_bgr, 1)

        # Detect face & landmarks
        landmarks, blendshapes = None, None
        if self.detector is not None:
            landmarks, blendshapes = self.detector.detect(frame_bgr)

        face_detected = landmarks is not None

        # Update Face Status overlay label
        if face_detected:
            self.lbl_face_status.config(text="Face: Detected", fg=COLOR_FACE_OK)
            # Draw blue landmark visualization
            frame_bgr = self.detector.draw_landmarks(frame_bgr, landmarks)
        else:
            self.lbl_face_status.config(
                text="⚠ No face detected" if self.state == AppState.RECORDING else "Face: Not detected",
                fg=COLOR_FACE_NO
            )

        # Handle sample collection during RECORDING state
        if self.state == AppState.RECORDING:
            self.frame_counter += 1
            if face_detected and (self.frame_counter % SAMPLE_INTERVAL == 0):
                feature_vector, _ = extract_facial_features(landmarks, blendshapes)
                # Quality check: prevent empty or zero-filled vectors
                if feature_vector is not None and np.any(feature_vector != 0):
                    self.session_samples.append(feature_vector)
                    self.lbl_sample_count.config(text=f"Samples captured: {len(self.session_samples)}")

        # Store latest frame
        self.latest_frame = frame_bgr.copy()

        # Render frame onto Tkinter Canvas
        self._render_frame_to_canvas(frame_bgr)

        # Schedule next update ~30 FPS
        self.root.after(15, self._update_camera)

    def _render_frame_to_canvas(self, frame_bgr: np.ndarray):
        """Convert BGR OpenCV frame to PIL ImageTk image and display on Canvas."""
        h, w, _ = frame_bgr.shape
        canvas_w = self.cam_canvas.winfo_width()
        canvas_h = self.cam_canvas.winfo_height()

        if canvas_w < 50 or canvas_h < 50:
            canvas_w, canvas_h = 640, 480

        # Calculate fit aspect ratio
        scale = min(canvas_w / w, canvas_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)

        if new_w > 0 and new_h > 0:
            pil_img = pil_img.resize((new_w, new_h), Image.Resampling.BILINEAR)

        img_tk = ImageTk.PhotoImage(image=pil_img)
        self.cam_canvas.img_tk = img_tk  # Keep reference to prevent GC
        self.cam_canvas.config(image=img_tk)

    # =========================================================================
    # Event Handlers & State Transitions
    # =========================================================================

    def _on_toggle_recording(self):
        """Toggle recording: LIVE -> RECORDING or RECORDING -> LABELING."""
        if self.state == AppState.LIVE:
            # Start Recording
            self.state = AppState.RECORDING
            self.session_samples.clear()
            self.frame_counter = 0
            self.selected_label = None
            self.emotion_var.set("")
            self._update_ui_state()

        elif self.state == AppState.RECORDING:
            # Stop Recording & Freeze Frame -> Enter LABELING state
            self.state = AppState.LABELING
            if self.latest_frame is not None:
                self.current_frozen_frame = self.latest_frame.copy()
            self._update_ui_state()

    def _on_emotion_selected(self):
        """Callback when user selects an emotion radio button."""
        self.selected_label = self.emotion_var.get()

    def _on_save(self):
        """Save recorded session samples to CSV dataset."""
        if self.state != AppState.LABELING:
            return

        if not self.session_samples:
            messagebox.showwarning("Empty Session", "No facial samples were captured during this session.")
            self._reset_to_live()
            return

        label = self.emotion_var.get()
        if not label:
            messagebox.showwarning("Missing Label", "Please select an expression label before saving!")
            return

        # Generate session ID and timestamp
        session_id = f"sess_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        # Check existing CSV header structure for compatibility
        header_has_session_info = True
        file_exists = os.path.exists(self.dataset_path) and os.path.getsize(self.dataset_path) > 0

        if file_exists:
            try:
                with open(self.dataset_path, "r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                    if header and "session_id" not in header:
                        header_has_session_info = False
            except Exception as e:
                print(f"[Warning] Header check error: {e}")

        # If existing CSV has legacy header ('label', 'feature1'...), migrate it
        if file_exists and not header_has_session_info:
            self._migrate_existing_csv_header()

        # Append new samples to CSV dataset
        try:
            new_file = not os.path.exists(self.dataset_path) or os.path.getsize(self.dataset_path) == 0
            with open(self.dataset_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if new_file:
                    writer.writerow(["session_id", "timestamp", "label"] + FEATURE_NAMES)

                for sample in self.session_samples:
                    row = [session_id, timestamp, label] + sample.tolist()
                    writer.writerow(row)

            # Update dataset counts
            self.dataset_counts[label] = self.dataset_counts.get(label, 0) + len(self.session_samples)
            self._refresh_stats_ui()

        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save samples to CSV dataset:\n{e}")
            return

        # Return to LIVE state
        self._reset_to_live()

    def _migrate_existing_csv_header(self):
        """Upgrade legacy CSV header to include session_id and timestamp columns."""
        try:
            rows = []
            with open(self.dataset_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                for r in reader:
                    if r:
                        # Prepend default session_id and timestamp
                        rows.append(["legacy_session", "", r[0]] + r[1:])

            with open(self.dataset_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["session_id", "timestamp", "label"] + FEATURE_NAMES)
                writer.writerows(rows)
        except Exception as e:
            print(f"[Warning] Failed CSV header migration: {e}")

    def _on_discard(self):
        """Discard current recording session samples without saving."""
        if self.state != AppState.LABELING:
            return

        self.session_samples.clear()
        self._reset_to_live()

    def _reset_to_live(self):
        """Reset session buffers and return application to LIVE mode."""
        self.session_samples.clear()
        self.selected_label = None
        self.emotion_var.set("")
        self.current_frozen_frame = None
        self.state = AppState.LIVE
        self._update_ui_state()

    def _on_quit(self):
        """Cleanly release resources and close application window."""
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            self.cap = None

        if self.detector is not None:
            self.detector.close()
            self.detector = None

        self.root.destroy()


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    root = tk.Tk()
    app = FacialDatasetGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
