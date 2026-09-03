"""
BGM MASTER V1 - Offline Music Player & Moodify AI Engine.

Complete offline desktop music player application featuring:
  - Folder & File MP3/Audio Importing with Track Metadata
  - Dynamic Mood Playlists (Romantic, Happy, Sad, Lonely, Chill, Excited + Custom Moods)
  - Profile Management & Automated Multi-Frame Face Registration with Live Validation
  - Real-Time Facial Expression Recognition & Automatic Mood Playlist Auto-Play (Moodify)

Usage:
  python bgm_master.py
"""

import os
import sys
import time
import json
import math
import uuid
import ctypes
import warnings
from typing import Optional, List, Dict, Tuple, Any

import cv2
import numpy as np

# Suppress protobuf symbol_database deprecation warning emitted by MediaPipe internals
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")
warnings.filterwarnings("ignore", message=".*SymbolDatabase.GetPrototype.*")

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from PIL import Image, ImageTk

# Attempt optional PyGame import for audio playback
try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False

from src.face_detector import FaceDetector
from src.facial_features import extract_facial_features, FEATURE_NAMES, FEATURE_DIMENSION
from src.predict_model import FaceMoodPredictor, CONFIDENCE_THRESHOLD, PredictionResult


# =============================================================================
# Constants & Paths
# =============================================================================

WINDOW_TITLE = "BGM MASTER - Offline Music Player & Moodify AI"
WINDOW_GEOMETRY = "1240x780"

DEFAULT_MOODS = ["Romantic", "Happy", "Sad", "Lonely", "Chill", "Excited"]
SUPPORTED_AUDIO_EXTS = (".mp3", ".wav", ".m4a", ".flac", ".ogg")

# Emotion to Mood Mapping
EMOTION_TO_MOOD = {
    "HAPPY": "Happy",
    "SAD": "Sad",
    "ANGRY": "Excited",
    "SURPRISED": "Excited",
    "NEUTRAL": "Chill",
    "FEAR": "Lonely",
    "DISGUST": "Lonely",
    "UNCERTAIN": "Chill"
}

# Theme Colors (Dark Modern UI)
COLOR_BG = "#1A1A1D"
COLOR_SIDEBAR = "#222226"
COLOR_CARD = "#2B2B30"
COLOR_CARD_BORDER = "#3F3F46"
COLOR_TEXT = "#E4E4E7"
COLOR_TEXT_MUTED = "#9CA3AF"
COLOR_WHITE = "#FFFFFF"

COLOR_ACCENT = "#10B981"        # Emerald Green (Moodify ON)
COLOR_ACCENT_HOVER = "#059669"
COLOR_PRIMARY = "#3B82F6"       # Blue Accent
COLOR_WARNING = "#F59E0B"       # Amber
COLOR_DANGER = "#EF4444"        # Red

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
PROFILES_FILE = os.path.join(DATA_DIR, "profiles.json")
LIBRARY_FILE = os.path.join(DATA_DIR, "music_library.json")
MOODS_FILE = os.path.join(DATA_DIR, "moods.json")


# =============================================================================
# Dual Audio Engine (PyGame + Windows WinMM MCI Fallback)
# =============================================================================

class AudioPlayer:
    """Robust MP3 Audio Engine supporting PyGame mixer with native Windows MCI fallback."""

    def __init__(self):
        self.use_pygame = False
        self.current_file: Optional[str] = None
        self.is_playing: bool = False
        self.is_paused: bool = False
        self.volume: float = 0.8
        self.duration_sec: float = 0.0
        self.start_time: float = 0.0
        self.pause_offset: float = 0.0
        self.alias = "bgm_master_mci"

        if HAS_PYGAME:
            try:
                pygame.mixer.init()
                self.use_pygame = True
                print("[AudioEngine] PyGame mixer initialized.")
            except Exception as e:
                print(f"[AudioEngine] PyGame init failed, using WinMM fallback: {e}")
                self.use_pygame = False

    def _mci_send(self, cmd: str) -> str:
        if sys.platform != "win32":
            return ""
        buf = ctypes.create_unicode_buffer(256)
        ctypes.windll.winmm.mciSendStringW(cmd, buf, 255, 0)
        return buf.value

    def load(self, file_path: str):
        """Load an audio file into player."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        self.stop()
        self.current_file = file_path
        self.is_playing = False
        self.is_paused = False
        self.pause_offset = 0.0

        if self.use_pygame:
            try:
                pygame.mixer.music.load(file_path)
                try:
                    sound = pygame.mixer.Sound(file_path)
                    self.duration_sec = sound.get_length()
                except Exception:
                    self.duration_sec = 210.0
                return
            except Exception as e:
                print(f"[AudioEngine] PyGame load failed, trying WinMM: {e}")

        if sys.platform == "win32":
            self._mci_send(f'close {self.alias}')
            short_buf = ctypes.create_unicode_buffer(512)
            ctypes.windll.kernel32.GetShortPathNameW(file_path, short_buf, 512)
            path_to_open = short_buf.value if short_buf.value else file_path

            self._mci_send(f'open "{path_to_open}" type mpegvideo alias {self.alias}')
            len_str = self._mci_send(f'status {self.alias} length')
            try:
                self.duration_sec = float(len_str) / 1000.0 if len_str else 210.0
            except ValueError:
                self.duration_sec = 210.0

    def play(self):
        """Play or resume audio playback."""
        if not self.current_file:
            return

        if self.is_paused:
            if self.use_pygame:
                pygame.mixer.music.unpause()
            elif sys.platform == "win32":
                self._mci_send(f'resume {self.alias}')
            self.is_playing = True
            self.is_paused = False
            self.start_time = time.time() - self.pause_offset
            return

        if self.use_pygame:
            pygame.mixer.music.play()
            pygame.mixer.music.set_volume(self.volume)
        elif sys.platform == "win32":
            self._mci_send(f'play {self.alias} from 0')
            self.set_volume(self.volume)

        self.is_playing = True
        self.is_paused = False
        self.start_time = time.time()
        self.pause_offset = 0.0

    def pause(self):
        """Pause playback."""
        if self.is_playing and not self.is_paused:
            if self.use_pygame:
                pygame.mixer.music.pause()
            elif sys.platform == "win32":
                self._mci_send(f'pause {self.alias}')
            self.is_paused = True
            self.pause_offset = time.time() - self.start_time

    def stop(self):
        """Stop playback completely."""
        if self.use_pygame:
            try:
                pygame.mixer.music.stop()
            except Exception:
                pass
        elif sys.platform == "win32":
            self._mci_send(f'stop {self.alias}')

        self.is_playing = False
        self.is_paused = False
        self.pause_offset = 0.0

    def set_volume(self, volume: float):
        """Set volume level [0.0, 1.0]."""
        self.volume = max(0.0, min(1.0, volume))
        if self.use_pygame:
            pygame.mixer.music.set_volume(self.volume)
        elif sys.platform == "win32":
            vol_int = int(self.volume * 1000)
            self._mci_send(f'setaudio {self.alias} volume to {vol_int}')

    def get_position(self) -> float:
        """Get current playback position in seconds."""
        if not self.is_playing:
            return 0.0

        if self.is_paused:
            return self.pause_offset

        if self.use_pygame:
            pos_ms = pygame.mixer.music.get_pos()
            if pos_ms >= 0:
                return (pos_ms / 1000.0) + self.pause_offset

        if sys.platform == "win32":
            pos_str = self._mci_send(f'status {self.alias} position')
            try:
                return float(pos_str) / 1000.0 if pos_str else (time.time() - self.start_time)
            except ValueError:
                pass

        return max(0.0, time.time() - self.start_time)

    def seek(self, seconds: float):
        """Seek to timestamp in seconds."""
        if not self.current_file:
            return
        ms = int(seconds * 1000)
        was_playing = self.is_playing and not self.is_paused

        if sys.platform == "win32":
            self._mci_send(f'seek {self.alias} to {ms}')
            if was_playing:
                self._mci_send(f'play {self.alias}')
        elif self.use_pygame:
            try:
                pygame.mixer.music.set_pos(seconds)
            except Exception:
                pass


# =============================================================================
# Managers: Moods, Profiles, and Music Library
# =============================================================================

class MoodManager:
    """Manages default and custom user mood playlists."""

    def __init__(self, filepath: str = MOODS_FILE):
        self.filepath = filepath
        self.moods: List[str] = list(DEFAULT_MOODS)
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        self.load()

    def load(self):
        """Load custom moods from JSON."""
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    custom_moods = data.get("moods", [])
                    for m in custom_moods:
                        if m not in self.moods:
                            self.moods.append(m)
            except Exception as e:
                print(f"[Warning] Error loading moods: {e}")
        else:
            self.save()

    def save(self):
        """Save moods to JSON."""
        try:
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump({"moods": self.moods}, f, indent=2)
        except Exception as e:
            print(f"[Error] Failed to save moods: {e}")

    def add_mood(self, mood_name: str) -> bool:
        """Add a new custom mood."""
        clean_name = mood_name.strip().capitalize()
        if clean_name and clean_name not in self.moods:
            self.moods.append(clean_name)
            self.save()
            return True
        return False


class ProfileManager:
    """Manages user profiles and 51D face embeddings."""

    def __init__(self, filepath: str = PROFILES_FILE):
        self.filepath = filepath
        self.profiles: List[Dict[str, Any]] = []
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        self.load()

    def load(self):
        """Load profiles from JSON."""
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.profiles = data.get("profiles", [])
            except Exception as e:
                print(f"[Warning] Could not load profiles: {e}")
                self.profiles = []
        else:
            # Default profiles
            self.profiles = [
                {"id": "prof_1", "name": "Faheem", "feature_vectors": []},
                {"id": "prof_2", "name": "User 2", "feature_vectors": []},
                {"id": "prof_3", "name": "User 3", "feature_vectors": []},
            ]
            self.save()

    def save(self):
        """Save profiles to JSON."""
        try:
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump({"profiles": self.profiles}, f, indent=2)
        except Exception as e:
            print(f"[Error] Failed to save profiles: {e}")

    def add_profile(self, name: str, feature_vectors: Optional[List[np.ndarray]] = None) -> Dict[str, Any]:
        """Create a new profile with multiple feature vectors."""
        vec_list = [v.tolist() for v in feature_vectors] if feature_vectors else []
        new_prof = {
            "id": f"prof_{uuid.uuid4().hex[:6]}",
            "name": name.strip(),
            "feature_vectors": vec_list
        }
        self.profiles.append(new_prof)
        self.save()
        return new_prof

    def update_face_data(self, profile_id: str, feature_vectors: List[np.ndarray]):
        """Update registered feature embeddings for a profile."""
        for prof in self.profiles:
            if prof["id"] == profile_id or prof["name"].lower() == profile_id.lower():
                vec_list = [v.tolist() for v in feature_vectors]
                prof["feature_vectors"] = vec_list
                self.save()
                return True
        return False

    def recognize_face(self, feature_vector: np.ndarray, threshold: float = 0.45) -> Tuple[str, float]:
        """Recognize user embedding using minimum Euclidean distance against registered profile embeddings."""
        best_name = "Unknown User"
        min_dist = float("inf")

        for prof in self.profiles:
            vecs = prof.get("feature_vectors", [])
            if not vecs and prof.get("feature_vector"):
                vecs = [prof["feature_vector"]]

            for ref_vec in vecs:
                if ref_vec and len(ref_vec) == len(feature_vector):
                    ref_arr = np.array(ref_vec, dtype=np.float32)
                    dist = float(np.linalg.norm(feature_vector - ref_arr))
                    if dist < min_dist:
                        min_dist = dist
                        if dist <= threshold:
                            best_name = prof["name"]

        return best_name, min_dist


class MusicLibraryManager:
    """Manages imported audio tracks, folder scanning, and mood tagging."""

    def __init__(self, filepath: str = LIBRARY_FILE):
        self.filepath = filepath
        self.songs: List[Dict[str, Any]] = []
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        self.load()

    def load(self):
        """Load music library from JSON."""
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.songs = data.get("songs", [])
            except Exception as e:
                print(f"[Warning] Could not load music library: {e}")
                self.songs = []
        else:
            self.songs = []
            self.save()

    def save(self):
        """Save music library to JSON."""
        try:
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump({"songs": self.songs}, f, indent=2)
        except Exception as e:
            print(f"[Error] Failed to save music library: {e}")

    def import_file(self, file_path: str, mood: str = "Chill") -> Optional[Dict[str, Any]]:
        """Import a single local audio file."""
        if not os.path.exists(file_path):
            return None

        ext = os.path.splitext(file_path)[1].lower()
        if ext not in SUPPORTED_AUDIO_EXTS:
            return None

        for s in self.songs:
            if s["path"] == file_path:
                return s

        filename = os.path.basename(file_path)
        title, _ = os.path.splitext(filename)
        artist = "Unknown Artist"
        album = "Local Album"

        if " - " in title:
            parts = title.split(" - ", 1)
            artist, title = parts[0].strip(), parts[1].strip()

        song = {
            "id": f"song_{uuid.uuid4().hex[:6]}",
            "title": title,
            "artist": artist,
            "album": album,
            "duration": "03:30",
            "mood": mood,
            "path": file_path
        }
        self.songs.append(song)
        self.save()
        return song

    def import_folder(self, folder_path: str, default_mood: str = "Chill") -> int:
        """Recursively scan directory for supported audio files and import them."""
        if not os.path.exists(folder_path):
            return 0

        count = 0
        for root_dir, _, files in os.walk(folder_path):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in SUPPORTED_AUDIO_EXTS:
                    full_path = os.path.join(root_dir, file)
                    if self.import_file(full_path, mood=default_mood):
                        count += 1
        return count

    def set_song_mood(self, song_id: str, mood: str):
        """Update mood tag for a song."""
        for s in self.songs:
            if s["id"] == song_id:
                s["mood"] = mood
                self.save()
                return

    def get_songs_by_mood(self, mood: str) -> List[Dict[str, Any]]:
        """Return songs matching mood tag."""
        return [s for s in self.songs if s.get("mood", "").lower() == mood.lower()]


# =============================================================================
# Assign Mood Combobox Dropdown Dialog
# =============================================================================

class AssignMoodDialog:
    """Modal dialog providing a Combobox dropdown for song mood assignment."""

    def __init__(self, parent: tk.Tk, current_mood: str, mood_mgr: MoodManager):
        self.top = tk.Toplevel(parent)
        self.top.title("Assign Mood to Song")
        self.top.geometry("380x230")
        self.top.configure(bg=COLOR_BG)
        self.top.grab_set()

        self.mood_mgr = mood_mgr
        self.selected_mood: Optional[str] = None

        top_frame = ttk.Frame(self.top, padding=15)
        top_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(top_frame, text="SELECT MOOD TAG", style="Header.TLabel").pack(anchor=tk.W, pady=(0, 6))
        ttk.Label(top_frame, text="Choose mood tag for the selected song:", foreground=COLOR_TEXT_MUTED).pack(anchor=tk.W, pady=(0, 10))

        options = list(self.mood_mgr.moods) + ["+ Add New Mood"]

        self.combo_var = tk.StringVar(value=current_mood if current_mood in options else options[0])
        self.combo = ttk.Combobox(
            top_frame,
            textvariable=self.combo_var,
            values=options,
            state="readonly",
            font=("Segoe UI", 10, "bold")
        )
        self.combo.pack(fill=tk.X, pady=(0, 15))

        btn_frame = ttk.Frame(top_frame)
        btn_frame.pack(fill=tk.X)

        btn_ok = tk.Button(
            btn_frame,
            text="SAVE MOOD",
            font=("Segoe UI", 10, "bold"),
            bg=COLOR_PRIMARY,
            fg=COLOR_WHITE,
            relief=tk.FLAT,
            padx=12,
            pady=4,
            cursor="hand2",
            command=self._on_confirm
        )
        btn_ok.pack(side=tk.RIGHT)

        btn_cancel = tk.Button(
            btn_frame,
            text="Cancel",
            font=("Segoe UI", 10),
            bg="#4B5563",
            fg=COLOR_WHITE,
            relief=tk.FLAT,
            padx=10,
            pady=4,
            command=self.top.destroy
        )
        btn_cancel.pack(side=tk.LEFT)

    def _on_confirm(self):
        val = self.combo_var.get().strip()
        if val == "+ Add New Mood":
            new_m = simpledialog.askstring("Add New Mood", "Enter name for new custom mood:")
            if new_m and new_m.strip():
                clean_m = new_m.strip().capitalize()
                self.mood_mgr.add_mood(clean_m)
                self.selected_mood = clean_m
            else:
                return
        else:
            self.selected_mood = val

        self.top.destroy()


# =============================================================================
# Automated Multi-Frame Face Registration Modal Window
# =============================================================================

class FaceRegistrationModal:
    """Interactive modal window for automated multi-frame profile face registration."""

    SAMPLES_PER_STEP = 10

    PROMPT_STEPS = [
        ("Step 1 of 4: NEUTRAL", "Look directly at the camera with a relaxed neutral expression"),
        ("Step 2 of 4: HAPPY", "Smile naturally into the camera"),
        ("Step 3 of 4: SAD", "Show a gentle sad expression"),
        ("Step 4 of 4: SURPRISED / EXCITED", "Show a surprised or excited expression"),
    ]

    def __init__(self, parent: tk.Tk, profile_mgr: ProfileManager, on_complete_callback):
        self.top = tk.Toplevel(parent)
        self.top.title("Add New Profile - Face Registration")
        self.top.geometry("660x600")
        self.top.configure(bg=COLOR_BG)
        self.top.grab_set()

        self.profile_mgr = profile_mgr
        self.on_complete_callback = on_complete_callback

        self.current_step = 0
        self.step_samples: List[np.ndarray] = []
        self.recorded_vectors: List[np.ndarray] = []
        self.profile_name = ""

        self._build_ui()

        self.detector: Optional[FaceDetector] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = True
        self.is_capturing = True

        self.top.protocol("WM_DELETE_WINDOW", self._on_close)
        self.top.after(100, self._start_registration)

    def _build_ui(self):
        top_frame = ttk.Frame(self.top, padding=12)
        top_frame.pack(fill=tk.X)

        ttk.Label(top_frame, text="CREATE NEW PROFILE", style="Header.TLabel").pack(anchor=tk.W)

        name_frame = ttk.Frame(top_frame)
        name_frame.pack(fill=tk.X, pady=(6, 0))

        ttk.Label(name_frame, text="Profile Name:").pack(side=tk.LEFT, padx=(0, 6))
        self.entry_name = tk.Entry(
            name_frame,
            font=("Segoe UI", 10),
            bg=COLOR_CARD,
            fg=COLOR_WHITE,
            insertbackground=COLOR_WHITE
        )
        self.entry_name.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.entry_name.insert(0, f"User {len(self.profile_mgr.profiles) + 1}")

        cam_card = ttk.Frame(self.top, style="Card.TFrame", padding=6)
        cam_card.pack(fill=tk.BOTH, expand=True, padx=12, pady=6)

        self.cam_label = tk.Label(cam_card, bg="#000000", text="Starting camera...")
        self.cam_label.pack(fill=tk.BOTH, expand=True)

        prompt_card = ttk.Frame(self.top, style="Card.TFrame", padding=10)
        prompt_card.pack(fill=tk.X, padx=12, pady=(0, 6))

        title_row = ttk.Frame(prompt_card, style="Card.TFrame")
        title_row.pack(fill=tk.X)

        self.lbl_step_title = tk.Label(
            title_row,
            text=self.PROMPT_STEPS[0][0],
            font=("Segoe UI", 11, "bold"),
            bg=COLOR_CARD,
            fg=COLOR_PRIMARY
        )
        self.lbl_step_title.pack(side=tk.LEFT)

        self.lbl_sample_badge = tk.Label(
            title_row,
            text="[ ● CAPTURING 0/10 ]",
            font=("Segoe UI", 10, "bold"),
            bg=COLOR_CARD,
            fg=COLOR_WARNING
        )
        self.lbl_sample_badge.pack(side=tk.RIGHT)

        self.lbl_step_desc = tk.Label(
            prompt_card,
            text=self.PROMPT_STEPS[0][1],
            font=("Segoe UI", 10),
            bg=COLOR_CARD,
            fg=COLOR_TEXT
        )
        self.lbl_step_desc.pack(anchor=tk.W, pady=(2, 4))

        self.lbl_val_status = tk.Label(
            prompt_card,
            text="Face Status: Detecting...",
            font=("Segoe UI", 9, "bold"),
            bg=COLOR_CARD,
            fg=COLOR_PRIMARY
        )
        self.lbl_val_status.pack(anchor=tk.W)

        btn_bar = ttk.Frame(self.top, padding=12)
        btn_bar.pack(fill=tk.X)

        btn_cancel = tk.Button(
            btn_bar,
            text="Cancel",
            font=("Segoe UI", 10),
            bg="#4B5563",
            fg=COLOR_WHITE,
            relief=tk.FLAT,
            command=self._on_close
        )
        btn_cancel.pack(side=tk.LEFT)

    def _start_registration(self):
        try:
            self.detector = FaceDetector()
            self.cap = cv2.VideoCapture(0)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize camera or detector:\n{e}")
            self._on_close()
            return

        self._update_cam_loop()

    def _update_cam_loop(self):
        if not self.is_running or self.cap is None or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if ret and frame is not None:
            frame = cv2.flip(frame, 1)

            face_detected = False
            if self.detector is not None:
                landmarks, blendshapes = self.detector.detect(frame)
                face_detected = landmarks is not None

                if face_detected:
                    frame = self.detector.draw_landmarks(frame, landmarks)
                    self.lbl_val_status.config(text="Face status: Face detected - Hold still", fg=COLOR_ACCENT)

                    # Auto Multi-Frame Capture
                    if self.is_capturing and self.current_step < len(self.PROMPT_STEPS):
                        vec, _ = extract_facial_features(landmarks, blendshapes)
                        if np.any(vec != 0):
                            self.step_samples.append(vec)
                            cnt = len(self.step_samples)
                            self.lbl_sample_badge.config(text=f"[ ● CAPTURING {cnt}/{self.SAMPLES_PER_STEP} ]", fg=COLOR_WARNING)

                            if cnt >= self.SAMPLES_PER_STEP:
                                # Step Completed
                                self.recorded_vectors.extend(self.step_samples)
                                self.step_samples.clear()
                                self.current_step += 1

                                if self.current_step < len(self.PROMPT_STEPS):
                                    step_t, step_d = self.PROMPT_STEPS[self.current_step]
                                    self.lbl_step_title.config(text=step_t)
                                    self.lbl_step_desc.config(text=step_d)
                                    self.lbl_sample_badge.config(text=f"[ ● CAPTURING 0/{self.SAMPLES_PER_STEP} ]", fg=COLOR_WARNING)
                                else:
                                    # Complete All Registration Steps!
                                    self.is_capturing = False
                                    self._finalize_registration()
                                    return
                else:
                    self.lbl_val_status.config(text="Face status: No face detected", fg=COLOR_DANGER)

            # Render frame
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame).resize((600, 360), Image.Resampling.BILINEAR)
            img_tk = ImageTk.PhotoImage(image=pil_img)
            self.cam_label.img_tk = img_tk
            self.cam_label.config(image=img_tk, text="")

        self.top.after(30, self._update_cam_loop)

    def _finalize_registration(self):
        name = self.entry_name.get().strip()
        if not name:
            name = f"User {len(self.profile_mgr.profiles) + 1}"

        new_prof = self.profile_mgr.add_profile(name, self.recorded_vectors)
        messagebox.showinfo("Registration Complete", f"✓ Face profile for '{name}' successfully registered!")
        self._on_close()

        if self.on_complete_callback:
            self.on_complete_callback(new_prof)

    def _on_close(self):
        self.is_running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.detector is not None:
            self.detector.close()
            self.detector = None
        self.top.destroy()


# =============================================================================
# Main Desktop GUI Application
# =============================================================================

class BGMMasterApp:
    """Tkinter Desktop Application for BGM MASTER."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.geometry(WINDOW_GEOMETRY)
        self.root.minsize(1100, 700)
        self.root.configure(bg=COLOR_BG)

        # Subsystems
        self.mood_mgr = MoodManager()
        self.profile_mgr = ProfileManager()
        self.library_mgr = MusicLibraryManager()
        self.audio_player = AudioPlayer()

        # AI Predictors
        self.detector: Optional[FaceDetector] = None
        self.predictor: Optional[FaceMoodPredictor] = None
        self.cap: Optional[cv2.VideoCapture] = None

        # State
        self.moodify_on: bool = False
        self.current_profile: Optional[Dict[str, Any]] = None
        self.active_playlist_mood: str = "All Music"
        self.current_song: Optional[Dict[str, Any]] = None
        self.playlist: List[Dict[str, Any]] = []
        self.current_song_idx: int = -1

        self.last_detected_user: str = "Unknown User"
        self.last_detected_mood: str = "Chill"
        self.last_mood_switch_time: float = 0.0

        # UI Setup
        self._init_styles()
        self._build_ui()

        # Set initial active profile
        if self.profile_mgr.profiles:
            self.current_profile = self.profile_mgr.profiles[0]
            self._refresh_profile_dropdown()

        self._refresh_library_table()

        self.root.after(250, self._update_player_progress)
        self.root.protocol("WM_DELETE_WINDOW", self._on_quit)

    def _init_styles(self):
        style = ttk.Style()
        style.theme_use("clam")

        style.configure("TFrame", background=COLOR_BG)
        style.configure("Sidebar.TFrame", background=COLOR_SIDEBAR)
        style.configure("Card.TFrame", background=COLOR_CARD, relief="solid", borderwidth=1)

        style.configure("TLabel", background=COLOR_BG, foreground=COLOR_TEXT, font=("Segoe UI", 10))
        style.configure("Sidebar.TLabel", background=COLOR_SIDEBAR, foreground=COLOR_TEXT, font=("Segoe UI", 10))
        style.configure("Header.TLabel", background=COLOR_BG, foreground=COLOR_WHITE, font=("Segoe UI", 14, "bold"))
        style.configure("SubHeader.TLabel", background=COLOR_SIDEBAR, foreground=COLOR_WHITE, font=("Segoe UI", 10, "bold"))

        style.configure(
            "Treeview",
            background=COLOR_CARD,
            foreground=COLOR_TEXT,
            fieldbackground=COLOR_CARD,
            rowheight=32,
            font=("Segoe UI", 10),
            borderwidth=0
        )
        style.configure(
            "Treeview.Heading",
            background="#1E1E22",
            foreground=COLOR_WHITE,
            font=("Segoe UI", 10, "bold"),
            borderwidth=1
        )
        style.map("Treeview", background=[("selected", "#0284C7")], foreground=[("selected", COLOR_WHITE)])

    def _build_ui(self):
        # Top Header Bar
        top_bar = ttk.Frame(self.root, padding=(15, 10))
        top_bar.pack(fill=tk.X)

        lbl_app_title = ttk.Label(top_bar, text="BGM MASTER", style="Header.TLabel")
        lbl_app_title.pack(side=tk.LEFT)

        lbl_subtitle = ttk.Label(top_bar, text="Offline Music Player & Moodify AI Engine", foreground=COLOR_TEXT_MUTED)
        lbl_subtitle.pack(side=tk.LEFT, padx=(15, 0), pady=(2, 0))

        # Main Container
        main_layout = ttk.Frame(self.root)
        main_layout.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # -------------------------------------------------------------
        # 1. LEFT SIDEBAR
        # -------------------------------------------------------------
        left_sidebar = ttk.Frame(main_layout, style="Sidebar.TFrame", width=250, padding=12)
        left_sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))
        left_sidebar.pack_propagate(False)

        # Profile Selector Box
        prof_box = ttk.Frame(left_sidebar, style="Card.TFrame", padding=10)
        prof_box.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(prof_box, text="Who's Listening?", style="SubHeader.TLabel").pack(anchor=tk.W)

        self.prof_var = tk.StringVar()
        self.combo_profile = ttk.Combobox(
            prof_box,
            textvariable=self.prof_var,
            state="readonly",
            font=("Segoe UI", 10, "bold")
        )
        self.combo_profile.pack(fill=tk.X, pady=(4, 4))
        self.combo_profile.bind("<<ComboboxSelected>>", self._on_profile_dropdown_change)

        # MUSIC Options Box
        music_opt_box = ttk.Frame(left_sidebar, style="Card.TFrame", padding=10)
        music_opt_box.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(music_opt_box, text="MUSIC", style="SubHeader.TLabel").pack(anchor=tk.W, pady=(0, 4))

        btn_all_music = tk.Button(
            music_opt_box,
            text="🎵 All Music",
            font=("Segoe UI", 9, "bold"),
            bg="#374151",
            fg=COLOR_WHITE,
            relief=tk.FLAT,
            anchor=tk.W,
            cursor="hand2",
            command=lambda: self._select_playlist("All Music")
        )
        btn_all_music.pack(fill=tk.X, pady=2)

        btn_add_file = tk.Button(
            music_opt_box,
            text="+ Add Music File",
            font=("Segoe UI", 9),
            bg=COLOR_CARD,
            fg=COLOR_TEXT,
            relief=tk.FLAT,
            anchor=tk.W,
            cursor="hand2",
            command=self._on_import_file
        )
        btn_add_file.pack(fill=tk.X, pady=2)

        btn_add_folder = tk.Button(
            music_opt_box,
            text="📁 + Add Folder",
            font=("Segoe UI", 9, "bold"),
            bg="#1E293B",
            fg=COLOR_PRIMARY,
            relief=tk.FLAT,
            anchor=tk.W,
            cursor="hand2",
            command=self._on_import_folder
        )
        btn_add_folder.pack(fill=tk.X, pady=2)

        # MOODS Playlists Box
        moods_box = ttk.Frame(left_sidebar, style="Card.TFrame", padding=10)
        moods_box.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        ttk.Label(moods_box, text="MOOD PLAYLISTS", style="SubHeader.TLabel").pack(anchor=tk.W, pady=(0, 4))

        self.mood_buttons_container = ttk.Frame(moods_box, style="Card.TFrame")
        self.mood_buttons_container.pack(fill=tk.BOTH, expand=True)

        self._build_mood_buttons()

        btn_add_mood = tk.Button(
            moods_box,
            text="+ Add New Mood",
            font=("Segoe UI", 9, "bold"),
            bg="#374151",
            fg=COLOR_WHITE,
            relief=tk.FLAT,
            cursor="hand2",
            command=self._on_add_custom_mood
        )
        btn_add_mood.pack(fill=tk.X, side=tk.BOTTOM, pady=(6, 0))

        # Compact Modern Moodify Toggle Box
        moodify_box = ttk.Frame(left_sidebar, style="Card.TFrame", padding=10)
        moodify_box.pack(fill=tk.X, side=tk.BOTTOM)

        self.btn_moodify_toggle = tk.Button(
            moodify_box,
            text="⚡ MOODIFY [ OFF ]",
            font=("Segoe UI", 10, "bold"),
            bg="#4B5563",
            fg=COLOR_WHITE,
            activebackground="#374151",
            activeforeground=COLOR_WHITE,
            relief=tk.RAISED,
            bd=2,
            pady=6,
            cursor="hand2",
            command=self._on_toggle_moodify
        )
        self.btn_moodify_toggle.pack(fill=tk.X)

        # -------------------------------------------------------------
        # 2. CENTER MAIN AREA (Music Table & Playlist Header)
        # -------------------------------------------------------------
        center_panel = ttk.Frame(main_layout)
        center_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        title_bar = ttk.Frame(center_panel, padding=(0, 0, 0, 8))
        title_bar.pack(fill=tk.X)

        self.lbl_playlist_title = tk.Label(
            title_bar,
            text="🎵 All Music Playlist",
            font=("Segoe UI", 13, "bold"),
            bg=COLOR_BG,
            fg=COLOR_WHITE
        )
        self.lbl_playlist_title.pack(side=tk.LEFT)

        btn_assign_mood = tk.Button(
            title_bar,
            text="Assign Mood",
            font=("Segoe UI", 9, "bold"),
            bg="#475569",
            fg=COLOR_WHITE,
            relief=tk.FLAT,
            padx=8,
            pady=3,
            cursor="hand2",
            command=self._on_assign_mood_tag
        )
        btn_assign_mood.pack(side=tk.LEFT, padx=10)

        lbl_search = ttk.Label(title_bar, text="Search:")
        lbl_search.pack(side=tk.RIGHT, padx=(10, 4))

        self.search_var = tk.StringVar()
        self.search_var.trace_add("write", lambda *args: self._refresh_library_table())
        entry_search = tk.Entry(
            title_bar,
            textvariable=self.search_var,
            font=("Segoe UI", 10),
            bg=COLOR_CARD,
            fg=COLOR_WHITE,
            insertbackground=COLOR_WHITE,
            bd=1,
            relief=tk.SOLID,
            width=18
        )
        entry_search.pack(side=tk.RIGHT)

        table_container = ttk.Frame(center_panel, style="Card.TFrame")
        table_container.pack(fill=tk.BOTH, expand=True)

        columns = ("name", "artist", "album", "duration", "mood")
        self.tree_library = ttk.Treeview(table_container, columns=columns, show="headings", selectmode="browse")
        self.tree_library.heading("name", text="Song Name", anchor=tk.W)
        self.tree_library.heading("artist", text="Artist", anchor=tk.W)
        self.tree_library.heading("album", text="Album", anchor=tk.W)
        self.tree_library.heading("duration", text="Duration", anchor=tk.CENTER)
        self.tree_library.heading("mood", text="Mood", anchor=tk.W)

        self.tree_library.column("name", width=200, minwidth=140, stretch=True)
        self.tree_library.column("artist", width=140, minwidth=100, stretch=True)
        self.tree_library.column("album", width=130, minwidth=90, stretch=True)
        self.tree_library.column("duration", width=80, minwidth=60, stretch=False, anchor=tk.CENTER)
        self.tree_library.column("mood", width=100, minwidth=80, stretch=True)

        scrollbar = ttk.Scrollbar(table_container, orient=tk.VERTICAL, command=self.tree_library.yview)
        self.tree_library.configure(yscroll=scrollbar.set)

        self.tree_library.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.tree_library.bind("<Double-1>", lambda e: self._on_play_selected_song())

        # -------------------------------------------------------------
        # 3. RIGHT SIDEBAR (Live Video Preview when Moodify is ON)
        # -------------------------------------------------------------
        right_sidebar = ttk.Frame(main_layout, width=280)
        right_sidebar.pack(side=tk.RIGHT, fill=tk.Y)
        right_sidebar.pack_propagate(False)

        cam_card = ttk.Frame(right_sidebar, style="Card.TFrame", padding=8)
        cam_card.pack(fill=tk.BOTH, expand=True)

        ttk.Label(cam_card, text="User Video Preview", style="SubHeader.TLabel").pack(anchor=tk.W, pady=(0, 6))

        self.cam_canvas = tk.Label(
            cam_card,
            bg="#000000",
            text="Camera OFF\n(Moodify is Disabled)",
            fg=COLOR_TEXT_MUTED,
            font=("Segoe UI", 11)
        )
        self.cam_canvas.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        info_badge_frame = ttk.Frame(cam_card, style="Card.TFrame", padding=8)
        info_badge_frame.pack(fill=tk.X)

        self.lbl_det_user = tk.Label(
            info_badge_frame,
            text="Recognized User: Unknown User",
            font=("Segoe UI", 10, "bold"),
            bg=COLOR_CARD,
            fg=COLOR_WHITE,
            anchor=tk.W
        )
        self.lbl_det_user.pack(fill=tk.X, pady=2)

        self.lbl_det_mood = tk.Label(
            info_badge_frame,
            text="Current Mood: 😊 Chill",
            font=("Segoe UI", 10, "bold"),
            bg=COLOR_CARD,
            fg=COLOR_PRIMARY,
            anchor=tk.W
        )
        self.lbl_det_mood.pack(fill=tk.X, pady=2)

        self.lbl_det_playlist = tk.Label(
            info_badge_frame,
            text="Playlist: Chill",
            font=("Segoe UI", 10, "bold"),
            bg=COLOR_CARD,
            fg=COLOR_ACCENT,
            anchor=tk.W
        )
        self.lbl_det_playlist.pack(fill=tk.X, pady=2)

        # -------------------------------------------------------------
        # 4. BOTTOM PLAYER CONTROL BAR (Matching Sketch)
        # -------------------------------------------------------------
        bottom_bar = ttk.Frame(self.root, style="Card.TFrame", padding=(15, 10))
        bottom_bar.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=10)

        player_info_row = ttk.Frame(bottom_bar, style="Card.TFrame")
        player_info_row.pack(fill=tk.X, pady=(0, 6))

        lbl_art_icon = tk.Label(player_info_row, text="🎵", font=("Segoe UI", 16), bg=COLOR_CARD, fg=COLOR_PRIMARY)
        lbl_art_icon.pack(side=tk.LEFT, padx=(0, 8))

        self.lbl_now_playing = tk.Label(
            player_info_row,
            text="Now Playing:  No song selected",
            font=("Segoe UI", 11, "bold"),
            bg=COLOR_CARD,
            fg=COLOR_WHITE
        )
        self.lbl_now_playing.pack(side=tk.LEFT)

        self.lbl_mood_badge = tk.Label(
            player_info_row,
            text="Playlist : All Music",
            font=("Segoe UI", 10, "bold"),
            bg="#1E293B",
            fg=COLOR_ACCENT,
            padx=10,
            pady=3
        )
        self.lbl_mood_badge.pack(side=tk.RIGHT)

        controls_row = ttk.Frame(bottom_bar, style="Card.TFrame")
        controls_row.pack(fill=tk.X)

        btn_prev = tk.Button(
            controls_row,
            text="⏮",
            font=("Segoe UI", 12, "bold"),
            bg="#374151",
            fg=COLOR_WHITE,
            relief=tk.FLAT,
            width=3,
            cursor="hand2",
            command=self._on_prev_song
        )
        btn_prev.pack(side=tk.LEFT, padx=(0, 4))

        self.btn_play_pause = tk.Button(
            controls_row,
            text="▶",
            font=("Segoe UI", 12, "bold"),
            bg=COLOR_ACCENT,
            fg=COLOR_WHITE,
            relief=tk.FLAT,
            width=4,
            cursor="hand2",
            command=self._on_toggle_play_pause
        )
        self.btn_play_pause.pack(side=tk.LEFT, padx=4)

        btn_next = tk.Button(
            controls_row,
            text="⏭",
            font=("Segoe UI", 12, "bold"),
            bg="#374151",
            fg=COLOR_WHITE,
            relief=tk.FLAT,
            width=3,
            cursor="hand2",
            command=self._on_next_song
        )
        btn_next.pack(side=tk.LEFT, padx=(4, 12))

        self.lbl_time_elapsed = ttk.Label(controls_row, text="00:00", style="Card.TLabel")
        self.lbl_time_elapsed.pack(side=tk.LEFT, padx=(0, 6))

        self.progress_var = tk.DoubleVar(value=0.0)
        self.scale_progress = ttk.Scale(
            controls_row,
            from_=0.0,
            to=100.0,
            variable=self.progress_var,
            orient=tk.HORIZONTAL,
            command=self._on_seek_progress
        )
        self.scale_progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)

        self.lbl_time_total = ttk.Label(controls_row, text="00:00", style="Card.TLabel")
        self.lbl_time_total.pack(side=tk.LEFT, padx=(6, 15))

        lbl_vol_icon = ttk.Label(controls_row, text="🔊", style="Card.TLabel")
        lbl_vol_icon.pack(side=tk.LEFT, padx=(5, 2))

        self.vol_var = tk.DoubleVar(value=80.0)
        scale_vol = ttk.Scale(
            controls_row,
            from_=0.0,
            to=100.0,
            variable=self.vol_var,
            orient=tk.HORIZONTAL,
            length=90,
            command=self._on_volume_change
        )
        scale_vol.pack(side=tk.LEFT, padx=(0, 5))

    def _build_mood_buttons(self):
        for widget in self.mood_buttons_container.winfo_children():
            widget.destroy()

        for mood in self.mood_mgr.moods:
            bg_col = "#0284C7" if self.active_playlist_mood == mood else COLOR_CARD
            btn = tk.Button(
                self.mood_buttons_container,
                text=f"  {mood}",
                font=("Segoe UI", 9, "bold" if self.active_playlist_mood == mood else "normal"),
                bg=bg_col,
                fg=COLOR_WHITE if self.active_playlist_mood == mood else COLOR_TEXT,
                activebackground="#3E3E42",
                activeforeground=COLOR_WHITE,
                anchor=tk.W,
                relief=tk.FLAT,
                cursor="hand2",
                command=lambda m=mood: self._select_playlist(m)
            )
            btn.pack(fill=tk.X, pady=2)

    def _refresh_profile_dropdown(self):
        names = [p["name"] for p in self.profile_mgr.profiles] + ["+ Add New Profile"]
        self.combo_profile["values"] = names

        if self.current_profile:
            self.prof_var.set(self.current_profile["name"])
        elif names and names[0] != "+ Add New Profile":
            self.prof_var.set(names[0])
            self.current_profile = self.profile_mgr.profiles[0]

    def _on_profile_dropdown_change(self, event=None):
        sel_name = self.prof_var.get()
        if sel_name == "+ Add New Profile":
            self._on_click_add_profile()
            return

        for p in self.profile_mgr.profiles:
            if p["name"] == sel_name:
                self.current_profile = p
                messagebox.showinfo("Profile Changed", f"Active profile set to: {p['name']}")
                return

    def _on_click_add_profile(self):
        FaceRegistrationModal(self.root, self.profile_mgr, self._on_profile_registration_complete)

    def _on_profile_registration_complete(self, new_profile: Dict[str, Any]):
        self.current_profile = new_profile
        self._refresh_profile_dropdown()

    def _select_playlist(self, mood: str):
        self.active_playlist_mood = mood
        if mood == "All Music":
            self.lbl_playlist_title.config(text="🎵 All Music Playlist")
        else:
            self.lbl_playlist_title.config(text=f"🎧 {mood} Playlist")

        self._build_mood_buttons()
        self._refresh_library_table()

    def _on_add_custom_mood(self):
        new_m = simpledialog.askstring("Add New Mood", "Enter name for new custom mood:")
        if new_m:
            if self.mood_mgr.add_mood(new_m):
                self._select_playlist(new_m.strip().capitalize())
                messagebox.showinfo("Mood Created", f"Custom mood '{new_m.strip().capitalize()}' created!")
            else:
                messagebox.showwarning("Exists", "Mood already exists or invalid.")

    def _on_import_file(self):
        files = filedialog.askopenfilenames(
            title="Select Audio Files",
            filetypes=[("Audio Files", "*.mp3 *.wav *.m4a *.flac *.ogg"), ("All Files", "*.*")]
        )
        if files:
            count = 0
            curr_mood = self.active_playlist_mood if self.active_playlist_mood in self.mood_mgr.moods else "Chill"
            for f in files:
                if self.library_mgr.import_file(f, mood=curr_mood):
                    count += 1
            self._refresh_library_table()
            messagebox.showinfo("Import Complete", f"Imported {count} track(s).")

    def _on_import_folder(self):
        folder = filedialog.askdirectory(title="Select Music Folder")
        if folder:
            curr_mood = self.active_playlist_mood if self.active_playlist_mood in self.mood_mgr.moods else "Chill"
            count = self.library_mgr.import_folder(folder, default_mood=curr_mood)
            self._refresh_library_table()
            messagebox.showinfo("Folder Scan Complete", f"Scanned & imported {count} local audio track(s).")

    def _on_assign_mood_tag(self):
        selected_item = self.tree_library.selection()
        if not selected_item:
            messagebox.showwarning("No Selection", "Please select a song from the playlist table first.")
            return

        song_id = selected_item[0]
        target_song = next((s for s in self.library_mgr.songs if s["id"] == song_id), None)
        if not target_song:
            return

        dialog = AssignMoodDialog(self.root, target_song.get("mood", "Chill"), self.mood_mgr)
        self.root.wait_window(dialog.top)

        if dialog.selected_mood:
            self.library_mgr.set_song_mood(song_id, dialog.selected_mood)
            self._build_mood_buttons()
            self._refresh_library_table()

    def _refresh_library_table(self):
        for item in self.tree_library.get_children():
            self.tree_library.delete(item)

        search_query = self.search_var.get().strip().lower()

        self.playlist = []
        for s in self.library_mgr.songs:
            if self.active_playlist_mood != "All Music":
                if s.get("mood", "").lower() != self.active_playlist_mood.lower():
                    continue

            t_title = s.get("title", "").lower()
            t_artist = s.get("artist", "").lower()
            t_album = s.get("album", "").lower()
            t_mood = s.get("mood", "").lower()

            if not search_query or search_query in t_title or search_query in t_artist or search_query in t_album or search_query in t_mood:
                self.playlist.append(s)
                self.tree_library.insert(
                    "",
                    tk.END,
                    iid=s["id"],
                    values=(s["title"], s["artist"], s.get("album", "Local"), s.get("duration", "03:30"), s["mood"])
                )

    # =========================================================================
    # Audio Playback
    # =========================================================================

    def _play_song_object(self, song: Dict[str, Any]):
        fpath = song.get("path")
        if not fpath or not os.path.exists(fpath):
            messagebox.showerror("File Missing", f"Audio file not found on disk:\n{fpath}")
            return

        self.current_song = song
        try:
            self.current_song_idx = self.playlist.index(song)
        except ValueError:
            self.current_song_idx = -1

        try:
            self.audio_player.load(fpath)
            self.audio_player.play()
            self.btn_play_pause.config(text="⏸", bg=COLOR_WARNING)
            self.lbl_now_playing.config(text=f"Now Playing:  {song['title']}  |  {song['artist']}")
            self.lbl_mood_badge.config(text=f"Playlist : {song['mood']}")
        except Exception as e:
            messagebox.showerror("Playback Error", f"Failed to play MP3 file:\n{e}")

    def _on_play_selected_song(self):
        selected_item = self.tree_library.selection()
        if not selected_item:
            return
        song_id = selected_item[0]
        song = next((s for s in self.library_mgr.songs if s["id"] == song_id), None)
        if song:
            self._play_song_object(song)

    def _on_toggle_play_pause(self):
        if not self.current_song:
            if self.playlist:
                self._play_song_object(self.playlist[0])
            else:
                messagebox.showinfo("Empty Playlist", "Please import audio files into your library.")
            return

        if self.audio_player.is_playing and not self.audio_player.is_paused:
            self.audio_player.pause()
            self.btn_play_pause.config(text="▶", bg=COLOR_ACCENT)
        else:
            self.audio_player.play()
            self.btn_play_pause.config(text="⏸", bg=COLOR_WARNING)

    def _on_prev_song(self):
        if not self.playlist:
            return
        if self.current_song_idx > 0:
            self.current_song_idx -= 1
        else:
            self.current_song_idx = len(self.playlist) - 1

        self._play_song_object(self.playlist[self.current_song_idx])

    def _on_next_song(self):
        if not self.playlist:
            return
        if self.current_song_idx < len(self.playlist) - 1:
            self.current_song_idx += 1
        else:
            self.current_song_idx = 0

        self._play_song_object(self.playlist[self.current_song_idx])

    def _on_volume_change(self, val_str: str):
        try:
            vol_pct = float(val_str) / 100.0
            self.audio_player.set_volume(vol_pct)
        except ValueError:
            pass

    def _on_seek_progress(self, val_str: str):
        if self.audio_player.duration_sec > 0:
            try:
                pct = float(val_str) / 100.0
                target_sec = pct * self.audio_player.duration_sec
                self.audio_player.seek(target_sec)
            except ValueError:
                pass

    def _update_player_progress(self):
        if self.audio_player.is_playing:
            pos_sec = self.audio_player.get_position()
            dur_sec = self.audio_player.duration_sec

            if dur_sec > 0:
                pct = (pos_sec / dur_sec) * 100.0
                self.progress_var.set(pct)

            m_pos, s_pos = divmod(int(pos_sec), 60)
            m_dur, s_dur = divmod(int(dur_sec), 60)

            self.lbl_time_elapsed.config(text=f"{m_pos:02d}:{s_pos:02d}")
            self.lbl_time_total.config(text=f"{m_dur:02d}:{s_dur:02d}")

        self.root.after(250, self._update_player_progress)

    # =========================================================================
    # MOODIFY ENGINE & CAMERA LOOP
    # =========================================================================

    def _on_toggle_moodify(self):
        if not self.moodify_on:
            self._start_moodify()
        else:
            self._stop_moodify()

    def _start_moodify(self):
        if self.detector is None:
            try:
                self.detector = FaceDetector()
            except Exception as e:
                messagebox.showerror("Detector Error", f"Failed to initialize FaceDetector:\n{e}")
                return

        if self.predictor is None:
            try:
                self.predictor = FaceMoodPredictor()
            except Exception as e:
                messagebox.showerror("Predictor Error", f"Failed to initialize FaceMoodPredictor:\n{e}")
                return

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", "Could not open webcam for Moodify mode.")
            return

        self.moodify_on = True
        self.btn_moodify_toggle.config(
            text="⚡ MOODIFY [ ON ]",
            bg=COLOR_ACCENT,
            activebackground=COLOR_ACCENT_HOVER
        )
        self._update_camera_moodify()

    def _stop_moodify(self):
        self.moodify_on = False
        self.btn_moodify_toggle.config(
            text="⚡ MOODIFY [ OFF ]",
            bg="#4B5563",
            activebackground="#374151"
        )

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        if self.detector is not None:
            self.detector.close()
            self.detector = None

        self.predictor = None

        self.cam_canvas.config(
            image="",
            text="Camera OFF\n(Moodify is Disabled)",
            bg="#000000",
            fg=COLOR_TEXT_MUTED
        )
        self.lbl_det_user.config(text="Recognized User: Unknown User")
        self.lbl_det_mood.config(text="Current Mood: 😊 Chill", fg=COLOR_PRIMARY)
        self.lbl_det_playlist.config(text="Playlist: Chill")

    def _update_camera_moodify(self):
        if not self.moodify_on or self.cap is None or not self.cap.isOpened():
            return

        ret, frame_bgr = self.cap.read()
        if not ret or frame_bgr is None:
            self.root.after(30, self._update_camera_moodify)
            return

        frame_bgr = cv2.flip(frame_bgr, 1)

        landmarks, blendshapes = self.detector.detect(frame_bgr)
        face_detected = landmarks is not None

        if face_detected:
            frame_bgr = self.detector.draw_landmarks(frame_bgr, landmarks)

            feat_vec, _ = extract_facial_features(landmarks, blendshapes)
            rec_user, _ = self.profile_mgr.recognize_face(feat_vec)
            self.last_detected_user = rec_user

            if self.predictor is not None:
                res = self.predictor.predict_vector(feat_vec)
                raw_emo = res.emotion.upper()

                detected_mood = EMOTION_TO_MOOD.get(raw_emo, "Chill")
                self.last_detected_mood = detected_mood

                curr_t = time.time()
                if curr_t - self.last_mood_switch_time > 3.0:
                    self._recommend_matching_song(detected_mood)
                    self.last_mood_switch_time = curr_t
        else:
            self.last_detected_user = "No Face Detected"

        self.lbl_det_user.config(text=f"Recognized User: {self.last_detected_user}")
        self.lbl_det_mood.config(text=f"Current Mood: {self.last_detected_mood}")
        self.lbl_det_playlist.config(text=f"Playlist: {self.last_detected_mood}")

        self._render_camera_preview(frame_bgr)
        self.root.after(30, self._update_camera_moodify)

    def _render_camera_preview(self, frame_bgr: np.ndarray):
        h, w, _ = frame_bgr.shape
        cw = self.cam_canvas.winfo_width()
        ch = self.cam_canvas.winfo_height()

        if cw < 40 or ch < 40:
            cw, ch = 260, 190

        scale = min(cw / w, ch / h)
        new_w, new_h = int(w * scale), int(h * scale)

        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)

        if new_w > 0 and new_h > 0:
            pil_img = pil_img.resize((new_w, new_h), Image.Resampling.BILINEAR)

        img_tk = ImageTk.PhotoImage(image=pil_img)
        self.cam_canvas.img_tk = img_tk
        self.cam_canvas.config(image=img_tk, text="")

    def _recommend_matching_song(self, target_mood: str):
        if self.active_playlist_mood != target_mood:
            self._select_playlist(target_mood)

        matching_songs = self.library_mgr.get_songs_by_mood(target_mood)
        if matching_songs:
            if self.current_song is None or self.current_song.get("mood", "").lower() != target_mood.lower():
                self._play_song_object(matching_songs[0])
        else:
            self.lbl_now_playing.config(text=f"No matching '{target_mood}' song in library")

    def _on_quit(self):
        self.audio_player.stop()
        if self.moodify_on:
            self._stop_moodify()
        self.root.destroy()


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    root = tk.Tk()
    app = BGMMasterApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
