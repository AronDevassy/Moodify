"""
BGM MASTER V1 - Offline Music Player & Moodify AI Engine.

Complete offline desktop music player application built with CustomTkinter (CTk):
  - Functional Moodify AI CTkSwitch anchored in Right AI Panel below live camera preview
  - Custom Mood Deletion (- button) with confirmation dialog (Default moods preserved)
  - Visible Scrollable Sidebar for Moods & Custom User Playlists
  - Dedicated Grid Geometry: Overlap-Free Player Bar & Full-Page Expanding Music Library Table
  - Dark & Light Mode Theme Support via CTk appearance modes
  - Single-Window Page Navigation with Header Back Buttons
  - Profile Management & Manual Button-Based Face Registration (2 Compulsory Captures)
  - Moodify AI Engine: 10s Initial Analysis, Continuous Background Mood Buffer,
    Automatic Next-Track Queueing, 4s Crossfade, and Facial Emotion Detection.

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
import threading
import warnings
from collections import Counter
from typing import Optional, List, Dict, Tuple, Any

import cv2
import numpy as np
import customtkinter as ctk

# Suppress protobuf symbol_database deprecation warning emitted by MediaPipe internals
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")
warnings.filterwarnings("ignore", message=".*SymbolDatabase.GetPrototype.*")

from tkinter import filedialog, messagebox, simpledialog
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
# CustomTkinter Configuration & Constants
# =============================================================================

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("green")

WINDOW_TITLE = "BGM MASTER - Offline Music Player & Moodify AI"
WINDOW_GEOMETRY = "1240x780"

DEFAULT_MOODS = ["Romantic", "Happy", "Sad", "Lonely", "Chill", "Excited"]
MOOD_EMOJIS = {
    "Romantic": "❤️",
    "Happy": "😊",
    "Sad": "😔",
    "Lonely": "🖤",
    "Chill": "🌙",
    "Excited": "⚡"
}

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

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
PROFILES_FILE = os.path.join(DATA_DIR, "profiles.json")
LIBRARY_FILE = os.path.join(DATA_DIR, "music_library.json")
MOODS_FILE = os.path.join(DATA_DIR, "moods.json")
PLAYLISTS_FILE = os.path.join(DATA_DIR, "playlists.json")
SETTINGS_FILE = os.path.join(DATA_DIR, "settings.json")


# =============================================================================
# Theme Manager (Dark & Light Mode Support via CTk)
# =============================================================================

class ThemeManager:
    """Manages CTk appearance mode settings and persists user preference."""

    def __init__(self, filepath: str = SETTINGS_FILE):
        self.filepath = filepath
        self.current_theme_name = "dark"
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        self.load()

    def load(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.current_theme_name = data.get("theme", "dark")
                    ctk.set_appearance_mode("Dark" if self.current_theme_name == "dark" else "Light")
            except Exception as e:
                print(f"[Warning] Error loading settings: {e}")
                ctk.set_appearance_mode("Dark")
        else:
            self.save()

    def save(self):
        try:
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump({"theme": self.current_theme_name}, f, indent=2)
        except Exception as e:
            print(f"[Error] Failed to save settings: {e}")

    def set_theme(self, theme_name: str):
        if theme_name in ("dark", "light"):
            self.current_theme_name = theme_name
            ctk.set_appearance_mode("Dark" if theme_name == "dark" else "Light")
            self.save()


theme_mgr = ThemeManager()


# =============================================================================
# Dual Audio Engine (Crossfade Support via PyGame / Dual MCI Channels)
# =============================================================================

class AudioPlayer:
    """Audio Engine supporting dual track crossfade and WinMM MCI / PyGame volume control."""

    def __init__(self):
        self.use_pygame = False
        self.current_file: Optional[str] = None
        self.is_playing: bool = False
        self.is_paused: bool = False
        self.volume: float = 0.8
        self.duration_sec: float = 0.0
        self.start_time: float = 0.0
        self.pause_offset: float = 0.0

        # Dual MCI channels for crossfade
        self.alias_a = "bgm_master_mci_a"
        self.alias_b = "bgm_master_mci_b"
        self.active_alias = self.alias_a
        self.standby_alias = self.alias_b

        # Crossfade State
        self.is_crossfading = False
        self.crossfade_start_time = 0.0
        self.crossfade_duration = 4.0

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
            self._mci_send(f'close {self.active_alias}')
            short_buf = ctypes.create_unicode_buffer(512)
            ctypes.windll.kernel32.GetShortPathNameW(file_path, short_buf, 512)
            path_to_open = short_buf.value if short_buf.value else file_path

            self._mci_send(f'open "{path_to_open}" type mpegvideo alias {self.active_alias}')
            len_str = self._mci_send(f'status {self.active_alias} length')
            try:
                self.duration_sec = float(len_str) / 1000.0 if len_str else 210.0
            except ValueError:
                self.duration_sec = 210.0

    def play(self):
        if not self.current_file:
            return

        if self.is_paused:
            if self.use_pygame:
                pygame.mixer.music.unpause()
            elif sys.platform == "win32":
                self._mci_send(f'resume {self.active_alias}')
            self.is_playing = True
            self.is_paused = False
            self.start_time = time.time() - self.pause_offset
            return

        if self.use_pygame:
            pygame.mixer.music.play()
            pygame.mixer.music.set_volume(self.volume)
        elif sys.platform == "win32":
            self._mci_send(f'play {self.active_alias} from 0')
            self.set_volume(self.volume)

        self.is_playing = True
        self.is_paused = False
        self.start_time = time.time()
        self.pause_offset = 0.0

    def start_crossfade(self, next_file_path: str, duration: float = 4.0) -> float:
        if not os.path.exists(next_file_path):
            return 0.0

        self.is_crossfading = True
        self.crossfade_start_time = time.time()
        self.crossfade_duration = duration

        next_duration = 210.0

        if sys.platform == "win32" and not self.use_pygame:
            self._mci_send(f'close {self.standby_alias}')
            short_buf = ctypes.create_unicode_buffer(512)
            ctypes.windll.kernel32.GetShortPathNameW(next_file_path, short_buf, 512)
            path_to_open = short_buf.value if short_buf.value else next_file_path

            self._mci_send(f'open "{path_to_open}" type mpegvideo alias {self.standby_alias}')
            self._mci_send(f'setaudio {self.standby_alias} volume to 0')
            self._mci_send(f'play {self.standby_alias} from 0')

            len_str = self._mci_send(f'status {self.standby_alias} length')
            try:
                next_duration = float(len_str) / 1000.0 if len_str else 210.0
            except ValueError:
                next_duration = 210.0
        elif self.use_pygame:
            try:
                sound = pygame.mixer.Sound(next_file_path)
                next_duration = sound.get_length()
            except Exception:
                pass
            self.load(next_file_path)
            self.play()

        return next_duration

    def update_crossfade_ramp(self):
        if not self.is_crossfading:
            return

        elapsed = time.time() - self.crossfade_start_time
        progress = min(1.0, elapsed / max(0.1, self.crossfade_duration))

        vol_active = int((1.0 - progress) * self.volume * 1000)
        vol_standby = int(progress * self.volume * 1000)

        if sys.platform == "win32" and not self.use_pygame:
            self._mci_send(f'setaudio {self.active_alias} volume to {vol_active}')
            self._mci_send(f'setaudio {self.standby_alias} volume to {vol_standby}')

        if progress >= 1.0:
            if sys.platform == "win32" and not self.use_pygame:
                self._mci_send(f'stop {self.active_alias}')
                self._mci_send(f'close {self.active_alias}')
                self.active_alias, self.standby_alias = self.standby_alias, self.active_alias

            self.is_crossfading = False
            self.start_time = time.time()

    def pause(self):
        if self.is_playing and not self.is_paused:
            if self.use_pygame:
                pygame.mixer.music.pause()
            elif sys.platform == "win32":
                self._mci_send(f'pause {self.active_alias}')
            self.is_paused = True
            self.pause_offset = time.time() - self.start_time

    def stop(self):
        if self.use_pygame:
            try:
                pygame.mixer.music.stop()
            except Exception:
                pass
        elif sys.platform == "win32":
            self._mci_send(f'stop {self.active_alias}')
            self._mci_send(f'stop {self.standby_alias}')

        self.is_playing = False
        self.is_paused = False
        self.is_crossfading = False
        self.pause_offset = 0.0

    def set_volume(self, volume: float):
        self.volume = max(0.0, min(1.0, volume))
        if self.use_pygame:
            pygame.mixer.music.set_volume(self.volume)
        elif sys.platform == "win32":
            vol_int = int(self.volume * 1000)
            self._mci_send(f'setaudio {self.active_alias} volume to {vol_int}')

    def get_position(self) -> float:
        if not self.is_playing:
            return 0.0

        if self.is_paused:
            return self.pause_offset

        if self.use_pygame:
            pos_ms = pygame.mixer.music.get_pos()
            if pos_ms >= 0:
                return (pos_ms / 1000.0) + self.pause_offset

        if sys.platform == "win32":
            pos_str = self._mci_send(f'status {self.active_alias} position')
            try:
                return float(pos_str) / 1000.0 if pos_str else (time.time() - self.start_time)
            except ValueError:
                pass

        return max(0.0, time.time() - self.start_time)

    def seek(self, seconds: float):
        if not self.current_file:
            return
        ms = int(seconds * 1000)
        was_playing = self.is_playing and not self.is_paused

        if sys.platform == "win32":
            self._mci_send(f'seek {self.active_alias} to {ms}')
            if was_playing:
                self._mci_send(f'play {self.active_alias}')
        elif self.use_pygame:
            try:
                pygame.mixer.music.set_pos(seconds)
            except Exception:
                pass


# =============================================================================
# Managers: Moods, Profiles, Music Library, and Custom User Playlists
# =============================================================================

class MoodManager:
    """Manages default and custom user mood playlists. Default moods CANNOT be deleted."""

    def __init__(self, filepath: str = MOODS_FILE):
        self.filepath = filepath
        self.moods: List[str] = list(DEFAULT_MOODS)
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        self.load()

    def load(self):
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
        try:
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump({"moods": self.moods}, f, indent=2)
        except Exception as e:
            print(f"[Error] Failed to save moods: {e}")

    def add_mood(self, mood_name: str) -> bool:
        clean_name = mood_name.strip().capitalize()
        if clean_name and clean_name not in self.moods:
            self.moods.append(clean_name)
            self.save()
            return True
        return False

    def delete_mood(self, mood_name: str) -> bool:
        """Delete custom mood definition ONLY. Default moods CANNOT be deleted."""
        clean_name = mood_name.strip().capitalize()
        if clean_name in DEFAULT_MOODS:
            return False  # Protect default moods!

        if clean_name in self.moods:
            self.moods.remove(clean_name)
            self.save()
            return True
        return False


class PlaylistManager:
    """Manages custom user playlists, track assignments, and Home pinning persistently."""

    def __init__(self, filepath: str = PLAYLISTS_FILE):
        self.filepath = filepath
        self.playlists: List[Dict[str, Any]] = []
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        self.load()

    def load(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.playlists = data.get("playlists", [])
                    for pl in self.playlists:
                        if "pinned" not in pl:
                            pl["pinned"] = False
            except Exception as e:
                print(f"[Warning] Error loading playlists: {e}")
                self.playlists = []
        else:
            # Starter playlists
            self.playlists = [
                {"id": "pl_workout", "name": "Workout", "song_ids": [], "pinned": True},
                {"id": "pl_favorites", "name": "Favorites", "song_ids": [], "pinned": True}
            ]
            self.save()

    def save(self):
        try:
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump({"playlists": self.playlists}, f, indent=2)
        except Exception as e:
            print(f"[Error] Failed to save playlists: {e}")

    def create_playlist(self, name: str) -> Dict[str, Any]:
        clean_name = name.strip()
        pl = {
            "id": f"pl_{uuid.uuid4().hex[:6]}",
            "name": clean_name,
            "song_ids": [],
            "pinned": False
        }
        self.playlists.append(pl)
        self.save()
        return pl

    def rename_playlist(self, playlist_id: str, new_name: str) -> bool:
        clean_name = new_name.strip()
        if not clean_name:
            return False
        for pl in self.playlists:
            if pl["id"] == playlist_id:
                pl["name"] = clean_name
                self.save()
                return True
        return False

    def toggle_pin(self, playlist_id: str) -> bool:
        for pl in self.playlists:
            if pl["id"] == playlist_id:
                pl["pinned"] = not pl.get("pinned", False)
                self.save()
                return pl["pinned"]
        return False

    def delete_playlist(self, playlist_id: str) -> bool:
        """Delete playlist definition ONLY. Physical audio files are NEVER touched."""
        init_len = len(self.playlists)
        self.playlists = [pl for pl in self.playlists if pl["id"] != playlist_id]
        if len(self.playlists) < init_len:
            self.save()
            return True
        return False

    def add_song_to_playlist(self, playlist_id: str, song_id: str):
        for pl in self.playlists:
            if pl["id"] == playlist_id:
                if song_id not in pl["song_ids"]:
                    pl["song_ids"].append(song_id)
                    self.save()
                return

    def remove_song_from_playlist(self, playlist_id: str, song_id: str):
        for pl in self.playlists:
            if pl["id"] == playlist_id:
                if song_id in pl["song_ids"]:
                    pl["song_ids"].remove(song_id)
                    self.save()
                return

    def get_pinned_playlists(self) -> List[Dict[str, Any]]:
        return [pl for pl in self.playlists if pl.get("pinned", False)]


class ProfileManager:
    """Manages user profiles and 51D face embeddings."""

    def __init__(self, filepath: str = PROFILES_FILE):
        self.filepath = filepath
        self.profiles: List[Dict[str, Any]] = []
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        self.load()

    def load(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.profiles = data.get("profiles", [])
            except Exception as e:
                print(f"[Warning] Could not load profiles: {e}")
                self.profiles = []
        else:
            self.profiles = [
                {"id": "prof_1", "name": "Faheem", "feature_vectors": []},
                {"id": "prof_2", "name": "User 2", "feature_vectors": []},
                {"id": "prof_3", "name": "User 3", "feature_vectors": []},
            ]
            self.save()

    def save(self):
        try:
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump({"profiles": self.profiles}, f, indent=2)
        except Exception as e:
            print(f"[Error] Failed to save profiles: {e}")

    def add_profile(self, name: str, feature_vectors: Optional[List[np.ndarray]] = None) -> Dict[str, Any]:
        vec_list = [v.tolist() for v in feature_vectors] if feature_vectors else []
        new_prof = {
            "id": f"prof_{uuid.uuid4().hex[:6]}",
            "name": name.strip(),
            "feature_vectors": vec_list
        }
        self.profiles.append(new_prof)
        self.save()
        return new_prof

    def delete_profile(self, profile_id_or_name: str) -> bool:
        initial_count = len(self.profiles)
        self.profiles = [
            p for p in self.profiles
            if p["id"] != profile_id_or_name and p["name"].lower() != profile_id_or_name.lower()
        ]
        if len(self.profiles) < initial_count:
            self.save()
            return True
        return False

    def update_face_data(self, profile_id: str, feature_vectors: List[np.ndarray]):
        for prof in self.profiles:
            if prof["id"] == profile_id or prof["name"].lower() == profile_id.lower():
                vec_list = [v.tolist() for v in feature_vectors]
                prof["feature_vectors"] = vec_list
                self.save()
                return True
        return False

    def recognize_face(self, feature_vector: np.ndarray, threshold: float = 0.45) -> Tuple[str, float]:
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
        try:
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump({"songs": self.songs}, f, indent=2)
        except Exception as e:
            print(f"[Error] Failed to save music library: {e}")

    def import_file(self, file_path: str, mood: str = "Chill") -> Optional[Dict[str, Any]]:
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
        for s in self.songs:
            if s["id"] == song_id:
                s["mood"] = mood
                self.save()
                return

    def get_songs_by_mood(self, mood: str) -> List[Dict[str, Any]]:
        return [s for s in self.songs if s.get("mood", "").lower() == mood.lower()]


# =============================================================================
# CustomTkinter Dialogs & Modals
# =============================================================================

class PlaylistOptionsMenuModal:
    """Modal dialog providing options (Rename, Delete, Pin/Unpin) for a custom playlist."""

    def __init__(self, parent: ctk.CTk, playlist: Dict[str, Any], playlist_mgr: PlaylistManager, on_action_callback):
        self.top = ctk.CTkToplevel(parent)
        self.top.title(f"Options — {playlist['name']}")
        self.top.geometry("340x260")
        self.top.grab_set()

        self.playlist = playlist
        self.playlist_mgr = playlist_mgr
        self.on_action_callback = on_action_callback

        self._build_ui()

    def _build_ui(self):
        container = ctk.CTkFrame(self.top, corner_radius=10)
        container.pack(fill="both", expand=True, padx=15, pady=15)

        ctk.CTkLabel(container, text=f"🎵 {self.playlist['name'].upper()}", font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=15, pady=(15, 4))
        ctk.CTkLabel(container, text="Select an action for this playlist:", text_color="#A1A1AA").pack(anchor="w", padx=15, pady=(0, 15))

        is_pinned = self.playlist.get("pinned", False)
        pin_text = "📍 Unpin from Home" if is_pinned else "📌 Pin to Home"

        btn_pin = ctk.CTkButton(
            container,
            text=pin_text,
            font=ctk.CTkFont(weight="bold"),
            fg_color="#0284C7",
            hover_color="#0369A1",
            command=self._on_toggle_pin
        )
        btn_pin.pack(fill="x", padx=15, pady=4)

        btn_rename = ctk.CTkButton(
            container,
            text="✏️ Rename Playlist",
            fg_color="#374151",
            hover_color="#4B5563",
            command=self._on_rename
        )
        btn_rename.pack(fill="x", padx=15, pady=4)

        btn_delete = ctk.CTkButton(
            container,
            text="🗑️ Delete Playlist",
            font=ctk.CTkFont(weight="bold"),
            fg_color="#EF4444",
            hover_color="#DC2626",
            command=self._on_delete
        )
        btn_delete.pack(fill="x", padx=15, pady=4)

    def _on_toggle_pin(self):
        self.playlist_mgr.toggle_pin(self.playlist["id"])
        self.top.destroy()
        if self.on_action_callback:
            self.on_action_callback("pin_toggled")

    def _on_rename(self):
        new_name = simpledialog.askstring("Rename Playlist", f"Enter new name for '{self.playlist['name']}':")
        if new_name and new_name.strip():
            self.playlist_mgr.rename_playlist(self.playlist["id"], new_name.strip())
            self.top.destroy()
            if self.on_action_callback:
                self.on_action_callback("renamed")

    def _on_delete(self):
        if messagebox.askyesno("Confirm Deletion", f"Delete playlist '{self.playlist['name']}'?\n\nNOTE: Physical audio files on disk will NOT be deleted."):
            self.playlist_mgr.delete_playlist(self.playlist["id"])
            self.top.destroy()
            if self.on_action_callback:
                self.on_action_callback("deleted")


class AddSongsModal:
    """Modal dialog for picking songs from the library to add to a custom playlist."""

    def __init__(self, parent: ctk.CTk, playlist: Dict[str, Any], library_mgr: MusicLibraryManager, on_save_callback):
        self.top = ctk.CTkToplevel(parent)
        self.top.title(f"Add Songs to '{playlist['name']}'")
        self.top.geometry("520x480")
        self.top.grab_set()

        self.playlist = playlist
        self.library_mgr = library_mgr
        self.on_save_callback = on_save_callback
        self.checkbox_vars: Dict[str, ctk.StringVar] = {}

        self._build_ui()

    def _build_ui(self):
        container = ctk.CTkFrame(self.top, corner_radius=10)
        container.pack(fill="both", expand=True, padx=15, pady=15)

        ctk.CTkLabel(container, text=f"ADD SONGS TO {self.playlist['name'].upper()}", font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=15, pady=(15, 6))
        ctk.CTkLabel(container, text="Check songs from your library to add to this playlist:").pack(anchor="w", padx=15, pady=(0, 10))

        scroll = ctk.CTkScrollableFrame(container, corner_radius=8)
        scroll.pack(fill="both", expand=True, padx=15, pady=(0, 15))

        existing_song_ids = set(self.playlist.get("song_ids", []))

        if not self.library_mgr.songs:
            ctk.CTkLabel(scroll, text="Your music library is currently empty.").pack(anchor="w", padx=10, pady=10)
        else:
            for s in self.library_mgr.songs:
                var = ctk.StringVar(value="on" if s["id"] in existing_song_ids else "off")
                self.checkbox_vars[s["id"]] = var

                chk = ctk.CTkCheckBox(
                    scroll,
                    text=f"{s['title']} — {s['artist']} ({s['mood']})",
                    variable=var,
                    onvalue="on",
                    offvalue="off"
                )
                chk.pack(anchor="w", padx=10, pady=6)

        btn_row = ctk.CTkFrame(container, fg_color="transparent")
        btn_row.pack(fill="x", padx=15, pady=(0, 15))

        btn_save = ctk.CTkButton(
            btn_row,
            text="SAVE PLAYLIST TRACKS",
            font=ctk.CTkFont(weight="bold"),
            command=self._on_save
        )
        btn_save.pack(side="right")

        btn_cancel = ctk.CTkButton(
            btn_row,
            text="Cancel",
            fg_color="#4B5563",
            hover_color="#374151",
            command=self.top.destroy
        )
        btn_cancel.pack(side="left")

    def _on_save(self):
        selected_ids = [sid for sid, var in self.checkbox_vars.items() if var.get() == "on"]
        self.playlist["song_ids"] = selected_ids
        if self.on_save_callback:
            self.on_save_callback(self.playlist)
        self.top.destroy()


class AssignMoodDialog:
    """Modal dialog providing a CTkComboBox dropdown for song mood assignment."""

    def __init__(self, parent: ctk.CTk, current_mood: str, mood_mgr: MoodManager):
        self.top = ctk.CTkToplevel(parent)
        self.top.title("Assign Mood to Song")
        self.top.geometry("380x230")
        self.top.grab_set()

        self.mood_mgr = mood_mgr
        self.selected_mood: Optional[str] = None

        top_frame = ctk.CTkFrame(self.top, corner_radius=10)
        top_frame.pack(fill="both", expand=True, padx=15, pady=15)

        ctk.CTkLabel(top_frame, text="SELECT MOOD TAG", font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=15, pady=(15, 6))
        ctk.CTkLabel(top_frame, text="Choose mood tag for the selected song:").pack(anchor="w", padx=15, pady=(0, 10))

        options = list(self.mood_mgr.moods) + ["+ Add New Mood"]

        self.combo_var = ctk.StringVar(value=current_mood if current_mood in options else options[0])
        self.combo = ctk.CTkComboBox(
            top_frame,
            variable=self.combo_var,
            values=options,
            state="readonly"
        )
        self.combo.pack(fill="x", padx=15, pady=(0, 15))

        btn_frame = ctk.CTkFrame(top_frame, fg_color="transparent")
        btn_frame.pack(fill="x", padx=15, pady=(0, 15))

        btn_ok = ctk.CTkButton(
            btn_frame,
            text="SAVE MOOD",
            font=ctk.CTkFont(weight="bold"),
            command=self._on_confirm
        )
        btn_ok.pack(side="right")

        btn_cancel = ctk.CTkButton(
            btn_frame,
            text="Cancel",
            fg_color="#4B5563",
            hover_color="#374151",
            command=self.top.destroy
        )
        btn_cancel.pack(side="left")

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


class ManageProfilesModal:
    """Modal dialog for viewing and deleting user profiles using CTk."""

    def __init__(self, parent: ctk.CTk, profile_mgr: ProfileManager, on_profile_deleted_callback):
        self.top = ctk.CTkToplevel(parent)
        self.top.title("Manage Profiles")
        self.top.geometry("450x420")
        self.top.grab_set()

        self.profile_mgr = profile_mgr
        self.on_profile_deleted_callback = on_profile_deleted_callback

        self._build_ui()

    def _build_ui(self):
        top_frame = ctk.CTkFrame(self.top, corner_radius=10)
        top_frame.pack(fill="both", expand=True, padx=15, pady=15)

        ctk.CTkLabel(top_frame, text="MANAGE PROFILES", font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=15, pady=(15, 6))
        ctk.CTkLabel(top_frame, text="View registered profiles or remove profiles:").pack(anchor="w", padx=15, pady=(0, 10))

        self.list_container = ctk.CTkScrollableFrame(top_frame, corner_radius=8)
        self.list_container.pack(fill="both", expand=True, padx=15, pady=(0, 15))

        self._refresh_list()

        btn_close = ctk.CTkButton(
            top_frame,
            text="Close",
            fg_color="#4B5563",
            hover_color="#374151",
            command=self.top.destroy
        )
        btn_close.pack(side="right", padx=15, pady=(0, 15))

    def _refresh_list(self):
        for widget in self.list_container.winfo_children():
            widget.destroy()

        if not self.profile_mgr.profiles:
            ctk.CTkLabel(self.list_container, text="No registered profiles found.").pack(anchor="w", padx=5)
            return

        for prof in self.profile_mgr.profiles:
            row_frame = ctk.CTkFrame(self.list_container, corner_radius=6)
            row_frame.pack(fill="x", pady=4, padx=2)

            vec_count = len(prof.get("feature_vectors", []))
            lbl_info = ctk.CTkLabel(
                row_frame,
                text=f"👤 {prof['name']}  ({vec_count} samples)",
                font=ctk.CTkFont(size=11, weight="bold")
            )
            lbl_info.pack(side="left", padx=10, pady=8)

            btn_del = ctk.CTkButton(
                row_frame,
                text="Delete",
                fg_color="#EF4444",
                hover_color="#DC2626",
                width=70,
                command=lambda p=prof: self._on_delete_profile(p)
            )
            btn_del.pack(side="right", padx=10, pady=8)

    def _on_delete_profile(self, prof: Dict[str, Any]):
        if messagebox.askyesno("Confirm Deletion", f"Delete profile '{prof['name']}'?\nThis will remove registered face data (your music files will not be deleted)."):
            self.profile_mgr.delete_profile(prof["id"])
            self._refresh_list()
            if self.on_profile_deleted_callback:
                self.on_profile_deleted_callback()


# =============================================================================
# Manual Button-Based Face Registration CTkToplevel Window
# =============================================================================

class FaceRegistrationModal:
    """
    Interactive CTk modal window for manual button-based profile face registration.
    All 4 default steps (Neutral, Happy, Sad, Surprised/Excited) strictly require 2 compulsory manual captures.
    """

    DEFAULT_EXPRESSIONS = [
        ("Neutral", "Show a relaxed neutral expression"),
        ("Happy", "Show a clear happy expression / smile"),
        ("Sad", "Show a sad expression"),
        ("Surprised/Excited", "Show a surprised or excited expression"),
    ]

    REQUIRED_CAPTURES_PER_EXP = 2

    def __init__(self, parent: ctk.CTk, profile_mgr: ProfileManager, on_complete_callback):
        self.top = ctk.CTkToplevel(parent)
        self.top.title("Create New Profile - Face Registration")
        self.top.geometry("700x660")
        self.top.grab_set()

        self.profile_mgr = profile_mgr
        self.on_complete_callback = on_complete_callback

        # State
        self.stage = 1
        self.profile_name = ""
        self.expressions: List[Tuple[str, str]] = list(self.DEFAULT_EXPRESSIONS)
        self.current_exp_idx = 0
        self.captured_vectors_by_exp: Dict[str, List[np.ndarray]] = {exp[0]: [] for exp in self.expressions}

        # Camera & Detector
        self.detector: Optional[FaceDetector] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_camera_running = False
        self.last_valid_landmarks = None
        self.last_valid_blendshapes = None

        self._build_ui_stage1_name()
        self.top.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui_stage1_name(self):
        for widget in self.top.winfo_children():
            widget.destroy()

        container = ctk.CTkFrame(self.top, corner_radius=12)
        container.pack(fill="both", expand=True, padx=25, pady=25)

        ctk.CTkLabel(container, text="CREATE NEW PROFILE", font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w", padx=20, pady=(20, 6))
        ctk.CTkLabel(container, text="Step 1: Enter profile name before starting camera registration").pack(anchor="w", padx=20, pady=(0, 15))

        card = ctk.CTkFrame(container, corner_radius=10)
        card.pack(fill="x", padx=20, pady=(0, 20))

        ctk.CTkLabel(card, text="Profile Name:", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=15, pady=(15, 6))

        self.entry_name = ctk.CTkEntry(card, font=ctk.CTkFont(size=12))
        self.entry_name.pack(fill="x", padx=15, pady=(0, 15))
        self.entry_name.insert(0, f"User {len(self.profile_mgr.profiles) + 1}")
        self.entry_name.focus_set()

        btn_start = ctk.CTkButton(
            card,
            text="Start Camera Registration ➔",
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self._on_start_registration_click
        )
        btn_start.pack(fill="x", padx=15, pady=(0, 15))

    def _on_start_registration_click(self):
        name = self.entry_name.get().strip()
        if not name:
            messagebox.showwarning("Missing Name", "Please enter a profile name first.")
            return

        self.profile_name = name
        self.stage = 2
        self._build_ui_stage2_registration()
        self._start_camera_hardware()

    def _build_ui_stage2_registration(self):
        for widget in self.top.winfo_children():
            widget.destroy()

        top_frame = ctk.CTkFrame(self.top, fg_color="transparent")
        top_frame.pack(fill="x", padx=15, pady=10)

        ctk.CTkLabel(top_frame, text=f"FACE REGISTRATION — {self.profile_name.upper()}", font=ctk.CTkFont(size=14, weight="bold")).pack(side="left")

        body = ctk.CTkFrame(self.top, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=15, pady=(0, 10))

        left_col = ctk.CTkFrame(body, fg_color="transparent")
        left_col.pack(side="left", fill="both", expand=True, padx=(0, 10))

        cam_card = ctk.CTkFrame(left_col, corner_radius=10)
        cam_card.pack(fill="both", expand=True)

        self.cam_label = ctk.CTkLabel(cam_card, text="Initializing camera...")
        self.cam_label.pack(fill="both", expand=True)

        self.lbl_face_status = ctk.CTkLabel(
            left_col,
            text="Face Status: Detecting...",
            font=ctk.CTkFont(size=11, weight="bold"),
            anchor="w"
        )
        self.lbl_face_status.pack(fill="x", pady=(4, 0))

        right_col = ctk.CTkFrame(body, width=310, corner_radius=10)
        right_col.pack(side="right", fill="y")
        right_col.pack_propagate(False)

        exp_card = ctk.CTkFrame(right_col, corner_radius=8)
        exp_card.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(exp_card, text="SELECT EXPRESSION", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=10, pady=(10, 4))

        exp_names = [e[0] for e in self.expressions] + ["+ Add Expression"]
        curr_exp_name = self.expressions[self.current_exp_idx][0]

        self.combo_exp_var = ctk.StringVar(value=curr_exp_name)
        self.combo_exp = ctk.CTkComboBox(
            exp_card,
            variable=self.combo_exp_var,
            values=exp_names,
            state="readonly",
            command=self._on_expression_dropdown_selected
        )
        self.combo_exp.pack(fill="x", padx=10, pady=(0, 8))

        self.lbl_exp_instruct = ctk.CTkLabel(
            exp_card,
            text=self.expressions[self.current_exp_idx][1],
            font=ctk.CTkFont(size=10),
            wraplength=270,
            justify="left"
        )
        self.lbl_exp_instruct.pack(anchor="w", padx=10, pady=(0, 10))

        self.lbl_capture_count = ctk.CTkLabel(
            exp_card,
            text="0 / 2 required captures",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#F59E0B"
        )
        self.lbl_capture_count.pack(anchor="w", padx=10, pady=(0, 10))

        self.btn_capture = ctk.CTkButton(
            exp_card,
            text="📷 CAPTURE",
            font=ctk.CTkFont(size=11, weight="bold"),
            command=self._on_click_capture
        )
        self.btn_capture.pack(fill="x", padx=10, pady=(0, 6))

        action_row = ctk.CTkFrame(exp_card, fg_color="transparent")
        action_row.pack(fill="x", padx=10, pady=(0, 10))

        self.btn_skip = ctk.CTkButton(
            action_row,
            text="SKIP",
            font=ctk.CTkFont(size=9, weight="bold"),
            fg_color="#4B5563",
            hover_color="#374151",
            state="disabled",
            command=self._on_click_skip
        )
        self.btn_skip.pack(side="left", fill="x", expand=True, padx=(0, 2))

        self.btn_next = ctk.CTkButton(
            action_row,
            text="NEXT ➔",
            font=ctk.CTkFont(size=9, weight="bold"),
            state="disabled",
            command=self._on_click_next_expression
        )
        self.btn_next.pack(side="right", fill="x", expand=True, padx=(2, 0))

        prog_card = ctk.CTkFrame(right_col, corner_radius=8)
        prog_card.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        ctk.CTkLabel(prog_card, text="REGISTRATION PROGRESS", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=10, pady=(10, 6))

        self.prog_list_frame = ctk.CTkScrollableFrame(prog_card, corner_radius=6)
        self.prog_list_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self._refresh_progress_checklist()
        self._update_capture_status_ui()

    def _refresh_progress_checklist(self):
        for w in self.prog_list_frame.winfo_children():
            w.destroy()

        for exp_name, _ in self.expressions:
            vecs = self.captured_vectors_by_exp.get(exp_name, [])
            cnt = len(vecs)
            status_icon = "✓" if cnt >= self.REQUIRED_CAPTURES_PER_EXP else "○"
            col = "#10B981" if cnt >= self.REQUIRED_CAPTURES_PER_EXP else "#A1A1AA"

            row = ctk.CTkFrame(self.prog_list_frame, corner_radius=4)
            row.pack(fill="x", pady=2, padx=2)

            ctk.CTkLabel(row, text=f"{status_icon} {exp_name}", font=ctk.CTkFont(size=10, weight="bold"), text_color=col).pack(side="left", padx=6, pady=4)
            ctk.CTkLabel(row, text=f"{cnt} captures", font=ctk.CTkFont(size=9), text_color="#A1A1AA").pack(side="right", padx=6, pady=4)

    def _update_capture_status_ui(self):
        curr_exp = self.expressions[self.current_exp_idx][0]
        cnt = len(self.captured_vectors_by_exp.get(curr_exp, []))

        if cnt < self.REQUIRED_CAPTURES_PER_EXP:
            self.lbl_capture_count.configure(text=f"{cnt} / {self.REQUIRED_CAPTURES_PER_EXP} required captures", text_color="#F59E0B")
            self.btn_skip.configure(state="disabled", fg_color="#4B5563")
            self.btn_next.configure(state="disabled", fg_color="#4B5563")
            self.btn_capture.configure(text="📷 CAPTURE")
        else:
            self.lbl_capture_count.configure(text=f"✓ {cnt} captures completed", text_color="#10B981")
            self.btn_skip.configure(state="normal", fg_color="#4B5563")
            self.btn_next.configure(state="normal")
            self.btn_capture.configure(text="📷 CAPTURE MORE")

        all_done = all(
            len(self.captured_vectors_by_exp.get(e[0], [])) >= self.REQUIRED_CAPTURES_PER_EXP
            for e in self.expressions[:4]
        )
        if all_done:
            self._build_ui_stage3_complete()

    def _on_expression_dropdown_selected(self, choice=None):
        sel = self.combo_exp_var.get()
        if sel == "+ Add Expression":
            new_exp = simpledialog.askstring("Add Expression", "Enter custom expression name:")
            if new_exp and new_exp.strip():
                clean_e = new_exp.strip().capitalize()
                self.expressions.append((clean_e, f"Show a {clean_e.lower()} expression"))
                if clean_e not in self.captured_vectors_by_exp:
                    self.captured_vectors_by_exp[clean_e] = []
                self.current_exp_idx = len(self.expressions) - 1
                self._update_expression_view()
        else:
            for idx, (exp_name, _) in enumerate(self.expressions):
                if exp_name == sel:
                    self.current_exp_idx = idx
                    self._update_expression_view()
                    break

    def _update_expression_view(self):
        exp_name, exp_desc = self.expressions[self.current_exp_idx]
        self.combo_exp_var.set(exp_name)
        self.lbl_exp_instruct.configure(text=exp_desc)
        self._update_capture_status_ui()
        self._refresh_progress_checklist()

    def _start_camera_hardware(self):
        try:
            self.detector = FaceDetector()
            self.cap = cv2.VideoCapture(0)
            self.is_camera_running = True
            self._update_cam_loop()
        except Exception as e:
            messagebox.showerror("Camera Error", f"Failed to initialize camera:\n{e}")

    def _update_cam_loop(self):
        if not self.is_camera_running or self.cap is None or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if ret and frame is not None:
            frame = cv2.flip(frame, 1)

            if self.detector is not None:
                landmarks, blendshapes = self.detector.detect(frame)
                if landmarks is not None:
                    self.last_valid_landmarks = landmarks
                    self.last_valid_blendshapes = blendshapes
                    frame = self.detector.draw_landmarks(frame, landmarks)
                    self.lbl_face_status.configure(text="Face status: Face detected ✓", text_color="#10B981")
                else:
                    self.last_valid_landmarks = None
                    self.last_valid_blendshapes = None
                    self.lbl_face_status.configure(text="Face status: No face detected — position face in camera", text_color="#EF4444")

            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame).resize((420, 310), Image.Resampling.BILINEAR)
            img_ctk = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(420, 310))
            self.cam_label.configure(image=img_ctk, text="")

        self.top.after(30, self._update_cam_loop)

    def _on_click_capture(self):
        if self.last_valid_landmarks is None:
            messagebox.showwarning("No Face Detected", "Cannot capture: No valid face detected in camera view.")
            return

        vec, _ = extract_facial_features(self.last_valid_landmarks, self.last_valid_blendshapes)
        if np.any(vec != 0):
            curr_exp = self.expressions[self.current_exp_idx][0]
            if curr_exp not in self.captured_vectors_by_exp:
                self.captured_vectors_by_exp[curr_exp] = []

            self.captured_vectors_by_exp[curr_exp].append(vec)
            cnt = len(self.captured_vectors_by_exp[curr_exp])

            self.lbl_face_status.configure(text=f"✓ Capture {cnt} saved successfully!", text_color="#10B981")
            self._update_capture_status_ui()
            self._refresh_progress_checklist()

    def _on_click_skip(self):
        if self.current_exp_idx < len(self.expressions) - 1:
            self.current_exp_idx += 1
            self._update_expression_view()

    def _on_click_next_expression(self):
        curr_exp = self.expressions[self.current_exp_idx][0]
        cnt = len(self.captured_vectors_by_exp.get(curr_exp, []))
        if cnt < self.REQUIRED_CAPTURES_PER_EXP:
            messagebox.showwarning("Incomplete", f"Please complete at least {self.REQUIRED_CAPTURES_PER_EXP} captures for '{curr_exp}' first.")
            return

        if self.current_exp_idx < len(self.expressions) - 1:
            self.current_exp_idx += 1
            self._update_expression_view()
        else:
            self._build_ui_stage3_complete()

    def _build_ui_stage3_complete(self):
        self.is_camera_running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        for widget in self.top.winfo_children():
            widget.destroy()

        container = ctk.CTkFrame(self.top, corner_radius=12)
        container.pack(fill="both", expand=True, padx=25, pady=25)

        ctk.CTkLabel(container, text="REGISTRATION COMPLETE ✓", font=ctk.CTkFont(size=16, weight="bold"), text_color="#10B981").pack(anchor="w", padx=20, pady=(20, 6))
        ctk.CTkLabel(container, text=f"Profile: {self.profile_name}", font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w", padx=20, pady=(0, 15))

        card = ctk.CTkFrame(container, corner_radius=10)
        card.pack(fill="both", expand=True, padx=20, pady=(0, 20))

        ctk.CTkLabel(card, text="RECORDED EXPRESSIONS SUMMARY:", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=15, pady=(15, 10))

        all_vectors = []
        for exp_name, vecs in self.captured_vectors_by_exp.items():
            if vecs:
                all_vectors.extend(vecs)
                row = ctk.CTkFrame(card, corner_radius=6)
                row.pack(fill="x", pady=3, padx=15)
                ctk.CTkLabel(row, text=f"✓ {exp_name}", font=ctk.CTkFont(size=11, weight="bold"), text_color="#10B981").pack(side="left", padx=10, pady=6)
                ctk.CTkLabel(row, text=f"{len(vecs)} valid captures saved", font=ctk.CTkFont(size=10)).pack(side="right", padx=10, pady=6)

        btn_save = ctk.CTkButton(
            container,
            text="💾 SAVE PROFILE",
            font=ctk.CTkFont(size=12, weight="bold"),
            command=lambda: self._on_save_final_profile(all_vectors)
        )
        btn_save.pack(fill="x", padx=20, pady=(0, 20))

    def _on_save_final_profile(self, all_vectors: List[np.ndarray]):
        if not all_vectors:
            messagebox.showerror("No Data", "No facial data was captured.")
            return

        new_prof = self.profile_mgr.add_profile(self.profile_name, all_vectors)
        messagebox.showinfo("Saved", f"Profile '{self.profile_name}' created and activated!")
        self._on_close()

        if self.on_complete_callback:
            self.on_complete_callback(new_prof)

    def _on_close(self):
        self.is_camera_running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.detector is not None:
            self.detector.close()
            self.detector = None
        self.top.destroy()


# =============================================================================
# Main CustomTkinter Application Class
# =============================================================================

class BGMMasterApp(ctk.CTk):
    """Main CustomTkinter Desktop Application for BGM MASTER with Functional Moodify AI Toggle in Right Panel."""

    INITIAL_ANALYSIS_SECONDS = 10.0
    FRAME_SKIP = 2  # Throttles camera landmark detection to every 2nd frame for low CPU usage

    def __init__(self):
        super().__init__()
        self.title(WINDOW_TITLE)
        self.geometry(WINDOW_GEOMETRY)
        self.minsize(1100, 700)

        # Subsystems
        self.mood_mgr = MoodManager()
        self.profile_mgr = ProfileManager()
        self.library_mgr = MusicLibraryManager()
        self.playlist_mgr = PlaylistManager()
        self.audio_player = AudioPlayer()

        # AI Predictors & Performance State
        self.detector: Optional[FaceDetector] = None
        self.predictor: Optional[FaceMoodPredictor] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.ai_frame_counter: int = 0

        # Player & Playlist State
        self.moodify_on: bool = False
        self.current_profile: Optional[Dict[str, Any]] = None
        self.active_view: str = "home"  # "home", "playlist", "custom_playlist", "settings"
        self.active_playlist_mood: str = "All Music"
        self.active_custom_playlist: Optional[Dict[str, Any]] = None
        self.current_song: Optional[Dict[str, Any]] = None
        self.next_queued_song: Optional[Dict[str, Any]] = None
        self.playlist: List[Dict[str, Any]] = []
        self.current_song_idx: int = -1
        self.selected_song_id: Optional[str] = None

        # Moodify AI Queue State
        self.ai_state: str = "IDLE"
        self.ai_analysis_start_time: float = 0.0
        self.initial_analysis_moods: List[str] = []
        self.background_mood_buffer: List[str] = []
        self.stable_mood: str = "Chill"
        self.ai_confidence: float = 88.0
        self.next_song_prepared: bool = False

        self._build_ui()
        self._refresh_profile_dropdown()
        self._show_home_view()

        self.after(200, self._main_timer_loop)
        self.protocol("WM_DELETE_WINDOW", self._on_quit)

    def _build_ui(self):
        # Grid Configuration (Row 0 for Sidebar & Content, Row 1 for Player Bar)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)

        # -------------------------------------------------------------
        # 1. PERMANENT SIDEBAR (Row 0, Column 0)
        # -------------------------------------------------------------
        self.sidebar_frame = ctk.CTkFrame(self, width=240, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=10)
        self.sidebar_frame.grid_rowconfigure(3, weight=1)

        # App Title
        lbl_title = ctk.CTkLabel(self.sidebar_frame, text="BGM MASTER", font=ctk.CTkFont(size=18, weight="bold"))
        lbl_title.grid(row=0, column=0, padx=18, pady=(15, 8), sticky="w")

        # Profile Box (Fixed Top)
        prof_box = ctk.CTkFrame(self.sidebar_frame, corner_radius=8)
        prof_box.grid(row=1, column=0, padx=12, pady=(0, 10), sticky="ew")

        ctk.CTkLabel(prof_box, text="Who's Listening?", font=ctk.CTkFont(size=10, weight="bold")).pack(anchor="w", padx=10, pady=(8, 2))

        self.combo_profile_var = ctk.StringVar()
        self.combo_profile = ctk.CTkComboBox(
            prof_box,
            variable=self.combo_profile_var,
            state="readonly",
            command=self._on_profile_dropdown_change
        )
        self.combo_profile.pack(fill="x", padx=8, pady=(0, 8))

        # LIBRARY Nav Box (Fixed Top)
        lib_box = ctk.CTkFrame(self.sidebar_frame, corner_radius=8)
        lib_box.grid(row=2, column=0, padx=12, pady=(0, 10), sticky="ew")

        ctk.CTkLabel(lib_box, text="LIBRARY", font=ctk.CTkFont(size=10, weight="bold"), text_color="#A1A1AA").pack(anchor="w", padx=10, pady=(8, 2))

        btn_home = ctk.CTkButton(lib_box, text="🏠 Home", font=ctk.CTkFont(weight="bold"), anchor="w", command=self._show_home_view)
        btn_home.pack(fill="x", padx=4, pady=2)

        btn_all = ctk.CTkButton(lib_box, text="🎵 All Music", anchor="w", fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"), command=lambda: self._select_playlist("All Music"))
        btn_all.pack(fill="x", padx=4, pady=2)

        btn_add_file = ctk.CTkButton(lib_box, text="+ Add Music File", anchor="w", fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"), command=self._on_import_file)
        btn_add_file.pack(fill="x", padx=4, pady=2)

        btn_add_folder = ctk.CTkButton(lib_box, text="📁 + Add Folder", anchor="w", fg_color="transparent", text_color="#10B981", hover_color=("gray70", "gray30"), command=self._on_import_folder)
        btn_add_folder.pack(fill="x", padx=4, pady=2)

        # MIDDLE SCROLLABLE FRAME (MOODS & USER PLAYLISTS with Visible Scrollbar!)
        self.sidebar_scroll_frame = ctk.CTkScrollableFrame(self.sidebar_frame, corner_radius=8, fg_color="transparent")
        self.sidebar_scroll_frame.grid(row=3, column=0, padx=12, pady=(0, 10), sticky="nsew")

        # MOODS Section inside scrollable area
        ctk.CTkLabel(self.sidebar_scroll_frame, text="MOOD PLAYLISTS", font=ctk.CTkFont(size=10, weight="bold"), text_color="#A1A1AA").pack(anchor="w", padx=4, pady=(4, 2))

        self.mood_buttons_container = ctk.CTkFrame(self.sidebar_scroll_frame, fg_color="transparent")
        self.mood_buttons_container.pack(fill="x", pady=2)
        self._build_mood_buttons()

        btn_add_mood = ctk.CTkButton(self.sidebar_scroll_frame, text="+ Add New Mood", font=ctk.CTkFont(size=10, weight="bold"), fg_color="#374151", hover_color="#4B5563", command=self._on_add_custom_mood)
        btn_add_mood.pack(fill="x", pady=(4, 12))

        # USER PLAYLISTS Section inside scrollable area
        header_row = ctk.CTkFrame(self.sidebar_scroll_frame, fg_color="transparent")
        header_row.pack(fill="x", pady=(4, 2))

        ctk.CTkLabel(header_row, text="USER PLAYLISTS", font=ctk.CTkFont(size=10, weight="bold"), text_color="#A1A1AA").pack(side="left")

        btn_plus_pl = ctk.CTkButton(
            header_row,
            text="+",
            font=ctk.CTkFont(size=14, weight="bold"),
            width=24,
            height=20,
            fg_color="#0284C7",
            hover_color="#0369A1",
            command=self._on_create_playlist_click
        )
        btn_plus_pl.pack(side="right")

        self.custom_playlists_scroll = ctk.CTkFrame(self.sidebar_scroll_frame, fg_color="transparent")
        self.custom_playlists_scroll.pack(fill="x", pady=2)

        self._build_custom_playlist_buttons()

        # Settings Nav Button (Fixed Bottom)
        btn_settings = ctk.CTkButton(self.sidebar_frame, text="⚙ Settings", font=ctk.CTkFont(weight="bold"), fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"), anchor="w", command=self._show_settings_view)
        btn_settings.grid(row=4, column=0, padx=12, pady=(0, 10), sticky="ew")

        # -------------------------------------------------------------
        # 2. CENTER CONTENT CONTAINER (Row 0, Column 1)
        # -------------------------------------------------------------
        self.content_container = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.content_container.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=10)
        self.content_container.grid_columnconfigure(0, weight=1)
        self.content_container.grid_rowconfigure(0, weight=1)

        # -------------------------------------------------------------
        # 3. FIXED BOTTOM PLAYER BAR (Row 1, Columns 0-1, Height 80px)
        # -------------------------------------------------------------
        player_bar = ctk.CTkFrame(self, height=80, corner_radius=10)
        player_bar.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 10))

        # Track Info Row
        info_row = ctk.CTkFrame(player_bar, fg_color="transparent")
        info_row.pack(fill="x", padx=15, pady=(6, 2))

        lbl_art = ctk.CTkLabel(info_row, text="🎵", font=ctk.CTkFont(size=16))
        lbl_art.pack(side="left", padx=(0, 8))

        self.lbl_now_playing = ctk.CTkLabel(info_row, text="Now Playing:  No song selected", font=ctk.CTkFont(size=11, weight="bold"))
        self.lbl_now_playing.pack(side="left")

        self.lbl_next_queue_banner = ctk.CTkLabel(info_row, text="Next: None", font=ctk.CTkFont(size=9, weight="bold"), text_color="#A1A1AA")
        self.lbl_next_queue_banner.pack(side="left", padx=(20, 0))

        self.lbl_mood_badge = ctk.CTkLabel(info_row, text="Playlist : All Music", font=ctk.CTkFont(size=10, weight="bold"), text_color="#10B981")
        self.lbl_mood_badge.pack(side="right")

        # Controls Row
        ctrl_row = ctk.CTkFrame(player_bar, fg_color="transparent")
        ctrl_row.pack(fill="x", padx=15, pady=(0, 6))

        btn_prev = ctk.CTkButton(ctrl_row, text="⏮", width=35, font=ctk.CTkFont(size=12, weight="bold"), fg_color="#374151", command=self._on_prev_song)
        btn_prev.pack(side="left", padx=(0, 4))

        self.btn_play_pause = ctk.CTkButton(ctrl_row, text="▶", width=45, font=ctk.CTkFont(size=12, weight="bold"), command=self._on_toggle_play_pause)
        self.btn_play_pause.pack(side="left", padx=4)

        btn_next = ctk.CTkButton(ctrl_row, text="⏭", width=35, font=ctk.CTkFont(size=12, weight="bold"), fg_color="#374151", command=self._on_next_song)
        btn_next.pack(side="left", padx=(4, 12))

        self.lbl_time_elapsed = ctk.CTkLabel(ctrl_row, text="00:00")
        self.lbl_time_elapsed.pack(side="left", padx=(0, 6))

        self.slider_progress = ctk.CTkSlider(ctrl_row, from_=0.0, to=100.0, command=self._on_seek_progress)
        self.slider_progress.pack(side="left", fill="x", expand=True, padx=6)
        self.slider_progress.set(0)

        self.lbl_time_total = ctk.CTkLabel(ctrl_row, text="00:00")
        self.lbl_time_total.pack(side="left", padx=(6, 15))

        lbl_vol = ctk.CTkLabel(ctrl_row, text="🔊")
        lbl_vol.pack(side="left", padx=(5, 2))

        self.slider_vol = ctk.CTkSlider(ctrl_row, from_=0.0, to=100.0, width=90, command=self._on_volume_change)
        self.slider_vol.pack(side="left", padx=(0, 5))
        self.slider_vol.set(80)

    # -------------------------------------------------------------------------
    # PAGE NAVIGATION VIEWS (Home with Pinned Playlist Cards, Playlist, Settings)
    # -------------------------------------------------------------------------

    def _show_home_view(self):
        self.active_view = "home"
        for w in self.content_container.winfo_children():
            w.destroy()

        main_split = ctk.CTkFrame(self.content_container, fg_color="transparent")
        main_split.pack(fill="both", expand=True)
        main_split.columnconfigure(0, weight=1)
        main_split.rowconfigure(0, weight=1)

        center_area = ctk.CTkFrame(main_split, fg_color="transparent")
        center_area.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        right_area = ctk.CTkFrame(main_split, width=280, corner_radius=10)
        right_area.grid(row=0, column=1, sticky="nsew")
        right_area.grid_propagate(False)

        self._build_right_ai_panel(right_area)

        header = ctk.CTkFrame(center_area, fg_color="transparent")
        header.pack(fill="x", pady=(0, 10))

        user_name = self.current_profile["name"] if self.current_profile else "User"
        ctk.CTkLabel(header, text=f"Good evening, {user_name} 👋", font=ctk.CTkFont(size=18, weight="bold")).pack(anchor="w")
        ctk.CTkLabel(header, text="Welcome back to BGM Master", text_color="#A1A1AA").pack(anchor="w")

        # Pinned Playlists Section (YOUR PLAYLISTS)
        pinned_pls = self.playlist_mgr.get_pinned_playlists()
        if pinned_pls:
            ctk.CTkLabel(center_area, text="YOUR PLAYLISTS", font=ctk.CTkFont(size=12, weight="bold"), text_color="#A1A1AA").pack(anchor="w", pady=(10, 6))

            pinned_grid = ctk.CTkFrame(center_area, fg_color="transparent")
            pinned_grid.pack(fill="x", pady=(0, 12))

            for idx, pl in enumerate(pinned_pls[:6]):
                card = ctk.CTkFrame(pinned_grid, corner_radius=8)
                r, c = divmod(idx, 3)
                card.grid(row=r, column=c, sticky="ew", padx=4, pady=4)
                pinned_grid.columnconfigure(c, weight=1)

                cnt = len(pl.get("song_ids", []))
                btn_pl_card = ctk.CTkButton(
                    card,
                    text=f"🎵  {pl['name']}\n{cnt} song{'s' if cnt != 1 else ''}",
                    font=ctk.CTkFont(size=11, weight="bold"),
                    anchor="w",
                    fg_color="transparent",
                    text_color=("gray10", "gray90"),
                    hover_color=("gray70", "gray30"),
                    command=lambda p=pl: self._select_custom_playlist(p)
                )
                btn_pl_card.pack(fill="both", expand=True, padx=8, pady=8)

        # Quick Mood Playlists Grid
        ctk.CTkLabel(center_area, text="YOUR MOOD PLAYLISTS", font=ctk.CTkFont(size=12, weight="bold"), text_color="#A1A1AA").pack(anchor="w", pady=(10, 6))

        grid = ctk.CTkFrame(center_area, fg_color="transparent")
        grid.pack(fill="x", pady=(0, 15))

        for idx, mood in enumerate(DEFAULT_MOODS[:6]):
            emoji = MOOD_EMOJIS.get(mood, "🎧")
            card = ctk.CTkFrame(grid, corner_radius=8)
            r, c = divmod(idx, 3)
            card.grid(row=r, column=c, sticky="ew", padx=4, pady=4)
            grid.columnconfigure(c, weight=1)

            btn = ctk.CTkButton(
                card,
                text=f"{emoji}  {mood}",
                font=ctk.CTkFont(size=12, weight="bold"),
                anchor="w",
                fg_color="transparent",
                text_color=("gray10", "gray90"),
                hover_color=("gray70", "gray30"),
                command=lambda m=mood: self._select_playlist(m)
            )
            btn.pack(fill="both", expand=True, padx=8, pady=8)

        # Recently Imported Music Table
        ctk.CTkLabel(center_area, text="RECENTLY IMPORTED MUSIC", font=ctk.CTkFont(size=12, weight="bold"), text_color="#A1A1AA").pack(anchor="w", pady=(5, 6))
        self.scroll_music_table = ctk.CTkScrollableFrame(center_area, corner_radius=10)
        self.scroll_music_table.pack(fill="both", expand=True)
        self._refresh_library_table()

    def _select_playlist(self, mood: str):
        self.active_view = "playlist"
        self.active_playlist_mood = mood
        self.active_custom_playlist = None

        for w in self.content_container.winfo_children():
            w.destroy()

        main_split = ctk.CTkFrame(self.content_container, fg_color="transparent")
        main_split.pack(fill="both", expand=True)
        main_split.columnconfigure(0, weight=1)
        main_split.rowconfigure(0, weight=1)

        center_area = ctk.CTkFrame(main_split, fg_color="transparent")
        center_area.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        right_area = ctk.CTkFrame(main_split, width=280, corner_radius=10)
        right_area.grid(row=0, column=1, sticky="nsew")
        right_area.grid_propagate(False)

        self._build_right_ai_panel(right_area)

        # Header Row with ← Back Button
        title_bar = ctk.CTkFrame(center_area, fg_color="transparent")
        title_bar.pack(fill="x", pady=(0, 10))

        btn_back = ctk.CTkButton(
            title_bar,
            text="← Back",
            font=ctk.CTkFont(size=11, weight="bold"),
            width=75,
            fg_color="#374151",
            hover_color="#4B5563",
            command=self._show_home_view
        )
        btn_back.pack(side="left", padx=(0, 10))

        emoji = MOOD_EMOJIS.get(mood, "🎵")
        lbl_title = ctk.CTkLabel(
            title_bar,
            text=f"{emoji} {mood} Playlist" if mood != "All Music" else "🎵 All Music Playlist",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        lbl_title.pack(side="left")

        btn_assign = ctk.CTkButton(
            title_bar,
            text="Assign Mood",
            font=ctk.CTkFont(size=10, weight="bold"),
            width=90,
            fg_color="#475569",
            command=self._on_assign_mood_tag
        )
        btn_assign.pack(side="left", padx=10)

        self.search_var = ctk.StringVar()
        self.search_var.trace_add("write", lambda *args: self._refresh_library_table())
        entry_search = ctk.CTkEntry(
            title_bar,
            placeholder_text="Search...",
            textvariable=self.search_var,
            width=160
        )
        entry_search.pack(side="right")

        table_header = ctk.CTkFrame(center_area, corner_radius=6, height=28)
        table_header.pack(fill="x", pady=(0, 4))

        ctk.CTkLabel(table_header, text="TITLE", font=ctk.CTkFont(size=10, weight="bold"), text_color="#A1A1AA", anchor="w", width=220).pack(side="left", padx=(12, 4))
        ctk.CTkLabel(table_header, text="ARTIST", font=ctk.CTkFont(size=10, weight="bold"), text_color="#A1A1AA", anchor="w", width=150).pack(side="left", padx=4)
        ctk.CTkLabel(table_header, text="ALBUM", font=ctk.CTkFont(size=10, weight="bold"), text_color="#A1A1AA", anchor="w", width=130).pack(side="left", padx=4)
        ctk.CTkLabel(table_header, text="MOOD", font=ctk.CTkFont(size=10, weight="bold"), text_color="#A1A1AA", anchor="w", width=90).pack(side="left", padx=4)
        ctk.CTkLabel(table_header, text="TIME", font=ctk.CTkFont(size=10, weight="bold"), text_color="#A1A1AA", anchor="e").pack(side="right", padx=12)

        self.scroll_music_table = ctk.CTkScrollableFrame(center_area, corner_radius=10)
        self.scroll_music_table.pack(fill="both", expand=True)

        self._build_mood_buttons()
        self._build_custom_playlist_buttons()
        self._refresh_library_table()

    def _select_custom_playlist(self, playlist: Dict[str, Any]):
        self.active_view = "custom_playlist"
        self.active_custom_playlist = playlist
        self.active_playlist_mood = ""

        for w in self.content_container.winfo_children():
            w.destroy()

        main_split = ctk.CTkFrame(self.content_container, fg_color="transparent")
        main_split.pack(fill="both", expand=True)
        main_split.columnconfigure(0, weight=1)
        main_split.rowconfigure(0, weight=1)

        center_area = ctk.CTkFrame(main_split, fg_color="transparent")
        center_area.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        right_area = ctk.CTkFrame(main_split, width=280, corner_radius=10)
        right_area.grid(row=0, column=1, sticky="nsew")
        right_area.grid_propagate(False)

        self._build_right_ai_panel(right_area)

        # Custom Playlist Header
        title_bar = ctk.CTkFrame(center_area, fg_color="transparent")
        title_bar.pack(fill="x", pady=(0, 10))

        btn_back = ctk.CTkButton(
            title_bar,
            text="← Back",
            font=ctk.CTkFont(size=11, weight="bold"),
            width=75,
            fg_color="#374151",
            hover_color="#4B5563",
            command=self._show_home_view
        )
        btn_back.pack(side="left", padx=(0, 10))

        lbl_title = ctk.CTkLabel(
            title_bar,
            text=f"🎵 {playlist['name']} Playlist",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        lbl_title.pack(side="left")

        btn_add = ctk.CTkButton(
            title_bar,
            text="+ Add Songs",
            font=ctk.CTkFont(size=10, weight="bold"),
            width=90,
            fg_color="#10B981",
            hover_color="#059669",
            command=lambda: self._on_click_add_songs_to_playlist(playlist)
        )
        btn_add.pack(side="left", padx=8)

        btn_options = ctk.CTkButton(
            title_bar,
            text="⋯",
            font=ctk.CTkFont(size=14, weight="bold"),
            width=36,
            fg_color="#4B5563",
            hover_color="#374151",
            command=lambda: self._open_playlist_options_modal(playlist)
        )
        btn_options.pack(side="left", padx=4)

        self.search_var = ctk.StringVar()
        self.search_var.trace_add("write", lambda *args: self._refresh_library_table())
        entry_search = ctk.CTkEntry(
            title_bar,
            placeholder_text="Search...",
            textvariable=self.search_var,
            width=140
        )
        entry_search.pack(side="right")

        table_header = ctk.CTkFrame(center_area, corner_radius=6, height=28)
        table_header.pack(fill="x", pady=(0, 4))

        ctk.CTkLabel(table_header, text="TITLE", font=ctk.CTkFont(size=10, weight="bold"), text_color="#A1A1AA", anchor="w", width=220).pack(side="left", padx=(12, 4))
        ctk.CTkLabel(table_header, text="ARTIST", font=ctk.CTkFont(size=10, weight="bold"), text_color="#A1A1AA", anchor="w", width=150).pack(side="left", padx=4)
        ctk.CTkLabel(table_header, text="ALBUM", font=ctk.CTkFont(size=10, weight="bold"), text_color="#A1A1AA", anchor="w", width=130).pack(side="left", padx=4)
        ctk.CTkLabel(table_header, text="MOOD", font=ctk.CTkFont(size=10, weight="bold"), text_color="#A1A1AA", anchor="w", width=90).pack(side="left", padx=4)
        ctk.CTkLabel(table_header, text="TIME", font=ctk.CTkFont(size=10, weight="bold"), text_color="#A1A1AA", anchor="e").pack(side="right", padx=12)

        self.scroll_music_table = ctk.CTkScrollableFrame(center_area, corner_radius=10)
        self.scroll_music_table.pack(fill="both", expand=True)

        self._build_mood_buttons()
        self._build_custom_playlist_buttons()
        self._refresh_library_table()

    def _show_settings_view(self):
        self.active_view = "settings"
        for w in self.content_container.winfo_children():
            w.destroy()

        main_split = ctk.CTkFrame(self.content_container, fg_color="transparent")
        main_split.pack(fill="both", expand=True)
        main_split.columnconfigure(0, weight=1)

        center_area = ctk.CTkFrame(main_split, fg_color="transparent")
        center_area.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        right_area = ctk.CTkFrame(main_split, width=280, corner_radius=10)
        right_area.grid(row=0, column=1, sticky="nsew")
        right_area.grid_propagate(False)

        self._build_right_ai_panel(right_area)

        header = ctk.CTkFrame(center_area, fg_color="transparent")
        header.pack(fill="x", pady=(0, 15))

        ctk.CTkLabel(header, text="SETTINGS", font=ctk.CTkFont(size=18, weight="bold")).pack(anchor="w")
        ctk.CTkLabel(header, text="Customize application preferences & appearance", text_color="#A1A1AA").pack(anchor="w")

        card = ctk.CTkFrame(center_area, corner_radius=10)
        card.pack(fill="x", pady=10, padx=5)

        ctk.CTkLabel(card, text="APPEARANCE MODE", font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w", padx=15, pady=(15, 6))

        self.theme_segment = ctk.CTkSegmentedButton(
            card,
            values=["Dark", "Light"],
            command=self._on_theme_changed
        )
        self.theme_segment.pack(anchor="w", padx=15, pady=(0, 15))
        self.theme_segment.set("Dark" if theme_mgr.current_theme_name == "dark" else "Light")

        info_card = ctk.CTkFrame(center_area, corner_radius=10)
        info_card.pack(fill="x", pady=15, padx=5)

        ctk.CTkLabel(info_card, text="ABOUT BGM MASTER", font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w", padx=15, pady=(15, 6))
        ctk.CTkLabel(info_card, text="BGM MASTER V1 — Modern Offline AI Facial Mood Music Player", text_color="#A1A1AA").pack(anchor="w", padx=15, pady=(0, 15))

    def _on_theme_changed(self, value):
        new_theme = value.lower()
        theme_mgr.set_theme(new_theme)
        if self.active_view == "settings":
            self._show_settings_view()

    # -------------------------------------------------------------------------
    # RIGHT AI PANEL (Camera Preview & Functional Moodify AI Toggle!)
    # -------------------------------------------------------------------------

    def _build_right_ai_panel(self, parent_frame):
        cam_card = ctk.CTkFrame(parent_frame, corner_radius=10)
        cam_card.pack(fill="both", expand=True, padx=8, pady=8)

        ctk.CTkLabel(cam_card, text="AI MOOD PANEL", font=ctk.CTkFont(size=11, weight="bold")).pack(anchor="w", padx=10, pady=(10, 6))

        # Live Camera Preview Window
        self.cam_canvas = ctk.CTkLabel(
            cam_card,
            text="Camera OFF\n(Moodify is Disabled)",
            font=ctk.CTkFont(size=11),
            text_color="#A1A1AA"
        )
        self.cam_canvas.pack(fill="both", expand=True, padx=6, pady=(0, 8))

        # FUNCTIONAL MOODIFY AI SWITCH IN RIGHT PANEL DIRECTLY BELOW CAMERA!
        ai_switch_box = ctk.CTkFrame(cam_card, corner_radius=8)
        ai_switch_box.pack(fill="x", padx=6, pady=(0, 6))

        ctk.CTkLabel(ai_switch_box, text="MOODIFY AI ENGINE", font=ctk.CTkFont(size=10, weight="bold"), text_color="#A1A1AA").pack(anchor="w", padx=8, pady=(6, 2))

        self.switch_moodify_var = ctk.StringVar(value="on" if self.moodify_on else "off")
        self.switch_moodify = ctk.CTkSwitch(
            ai_switch_box,
            text="MOODIFY [ ON ]" if self.moodify_on else "MOODIFY [ OFF ]",
            font=ctk.CTkFont(weight="bold"),
            variable=self.switch_moodify_var,
            onvalue="on",
            offvalue="off",
            command=self._on_toggle_moodify
        )
        self.switch_moodify.pack(anchor="w", padx=8, pady=(0, 6))

        badge = ctk.CTkFrame(cam_card, corner_radius=8)
        badge.pack(fill="x", padx=6, pady=6)

        initial_status_text = "AI Status: Moodify OFF" if not self.moodify_on else f"CURRENT MOOD: {MOOD_EMOJIS.get(self.stable_mood, '😊')} {self.stable_mood} ({int(self.ai_confidence)}%)"
        initial_status_col = "#3B82F6" if not self.moodify_on else "#10B981"

        self.lbl_det_mood = ctk.CTkLabel(
            badge,
            text=initial_status_text,
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=initial_status_col,
            anchor="w"
        )
        self.lbl_det_mood.pack(fill="x", padx=8, pady=4)

        self.lbl_det_playlist = ctk.CTkLabel(
            badge,
            text=f"PLAYING FROM: {self.stable_mood if self.moodify_on else 'All Music'} Playlist",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="#10B981",
            anchor="w"
        )
        self.lbl_det_playlist.pack(fill="x", padx=8, pady=(0, 4))

    def _build_mood_buttons(self):
        """Builds mood list. Default moods are preserved; custom moods get a subtle '-' delete button."""
        for widget in self.mood_buttons_container.winfo_children():
            widget.destroy()

        for mood in self.mood_mgr.moods:
            emoji = MOOD_EMOJIS.get(mood, "🎧")
            is_active = (self.active_view == "playlist" and self.active_playlist_mood == mood)
            fg_col = "#0284C7" if is_active else "transparent"
            txt_col = "#FFFFFF" if is_active else ("gray10", "gray90")

            row_frame = ctk.CTkFrame(self.mood_buttons_container, corner_radius=6, fg_color=fg_col)
            row_frame.pack(fill="x", pady=2)

            btn_mood = ctk.CTkButton(
                row_frame,
                text=f" {emoji}  {mood}",
                font=ctk.CTkFont(size=11, weight="bold" if is_active else "normal"),
                anchor="w",
                fg_color="transparent",
                text_color=txt_col,
                hover_color=("gray70", "gray30"),
                command=lambda m=mood: self._select_playlist(m)
            )
            btn_mood.pack(side="left", fill="x", expand=True)

            # Subtle '-' delete button ONLY for custom moods (Default moods CANNOT be deleted!)
            if mood not in DEFAULT_MOODS:
                btn_del_mood = ctk.CTkButton(
                    row_frame,
                    text="-",
                    font=ctk.CTkFont(size=14, weight="bold"),
                    width=22,
                    height=20,
                    fg_color="transparent",
                    text_color="#EF4444",
                    hover_color="#DC2626",
                    command=lambda m=mood: self._on_delete_custom_mood_click(m)
                )
                btn_del_mood.pack(side="right", padx=4)

    def _on_delete_custom_mood_click(self, mood: str):
        if messagebox.askyesno("Confirm Mood Deletion", f"Delete '{mood}' custom mood?\n\nNOTE: Songs with this mood tag will default back to 'Chill'. Physical audio files will NOT be deleted."):
            if self.mood_mgr.delete_mood(mood):
                # Update songs that had this mood
                for s in self.library_mgr.songs:
                    if s.get("mood", "").lower() == mood.lower():
                        s["mood"] = "Chill"
                self.library_mgr.save()

                self._build_mood_buttons()
                if self.active_playlist_mood == mood:
                    self._show_home_view()
                else:
                    self._refresh_library_table()

    def _build_custom_playlist_buttons(self):
        """Populates custom playlist rows inside the scrollable sidebar container."""
        for widget in self.custom_playlists_scroll.winfo_children():
            widget.destroy()

        for pl in self.playlist_mgr.playlists:
            is_selected = (self.active_view == "custom_playlist" and self.active_custom_playlist and self.active_custom_playlist["id"] == pl["id"])
            fg_col = "#10B981" if is_selected else "transparent"
            txt_col = "#FFFFFF" if is_selected else ("gray10", "gray90")

            row_frame = ctk.CTkFrame(self.custom_playlists_scroll, corner_radius=6, fg_color=fg_col)
            row_frame.pack(fill="x", pady=2)

            pin_indicator = "📍 " if pl.get("pinned", False) else "♫ "

            btn_pl = ctk.CTkButton(
                row_frame,
                text=f"{pin_indicator}{pl['name']}",
                font=ctk.CTkFont(size=11, weight="bold" if is_selected else "normal"),
                anchor="w",
                fg_color="transparent",
                text_color=txt_col,
                hover_color=("gray70", "gray30"),
                command=lambda p=pl: self._select_custom_playlist(p)
            )
            btn_pl.pack(side="left", fill="x", expand=True, padx=(4, 0))

            btn_opt = ctk.CTkButton(
                row_frame,
                text="⋯",
                font=ctk.CTkFont(size=12, weight="bold"),
                width=24,
                height=22,
                fg_color="transparent",
                text_color=txt_col,
                hover_color=("gray70", "gray30"),
                command=lambda p=pl: self._open_playlist_options_modal(p)
            )
            btn_opt.pack(side="right", padx=4)

    def _open_playlist_options_modal(self, playlist: Dict[str, Any]):
        modal = PlaylistOptionsMenuModal(self, playlist, self.playlist_mgr, self._on_playlist_options_action)
        self.wait_window(modal.top)

    def _on_playlist_options_action(self, action_type: str):
        self._build_custom_playlist_buttons()
        if self.active_view == "home":
            self._show_home_view()
        elif action_type == "deleted":
            self._show_home_view()

    # -------------------------------------------------------------------------
    # CUSTOM PLAYLIST ACTIONS
    # -------------------------------------------------------------------------

    def _on_create_playlist_click(self):
        name = simpledialog.askstring("Create Playlist", "Enter name for new playlist:")
        if name and name.strip():
            new_pl = self.playlist_mgr.create_playlist(name.strip())
            self._build_custom_playlist_buttons()
            self._select_custom_playlist(new_pl)

    def _on_click_add_songs_to_playlist(self, playlist: Dict[str, Any]):
        modal = AddSongsModal(self, playlist, self.library_mgr, self._on_playlist_songs_updated)
        self.wait_window(modal.top)

    def _on_playlist_songs_updated(self, updated_playlist: Dict[str, Any]):
        self.playlist_mgr.save()
        self._refresh_library_table()

    def _refresh_profile_dropdown(self):
        names = [p["name"] for p in self.profile_mgr.profiles] + ["+ Add New Profile", "Manage Profiles"]
        self.combo_profile.configure(values=names)

        if self.current_profile and self.current_profile["name"] in names:
            self.combo_profile_var.set(self.current_profile["name"])
        elif names and names[0] not in ("+ Add New Profile", "Manage Profiles"):
            self.combo_profile_var.set(names[0])
            self.current_profile = self.profile_mgr.profiles[0]
        else:
            self.combo_profile_var.set("No profile selected")
            self.current_profile = None

    def _on_profile_dropdown_change(self, choice=None):
        sel_name = self.combo_profile_var.get()
        if sel_name == "+ Add New Profile":
            self._on_click_add_profile()
            return
        elif sel_name == "Manage Profiles":
            ManageProfilesModal(self, self.profile_mgr, self._refresh_profile_dropdown)
            return

        for p in self.profile_mgr.profiles:
            if p["name"] == sel_name:
                self.current_profile = p
                messagebox.showinfo("Profile Changed", f"Active profile set to: {p['name']}")
                return

    def _on_click_add_profile(self):
        FaceRegistrationModal(self, self.profile_mgr, self._on_profile_registration_complete)

    def _on_profile_registration_complete(self, new_profile: Dict[str, Any]):
        self.current_profile = new_profile
        self._refresh_profile_dropdown()

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

            def _async_import():
                count = self.library_mgr.import_folder(folder, default_mood=curr_mood)
                self.after(0, lambda: self._on_import_folder_complete(count))

            threading.Thread(target=_async_import, daemon=True).start()

    def _on_import_folder_complete(self, count: int):
        self._refresh_library_table()
        messagebox.showinfo("Folder Scan Complete", f"Scanned & imported {count} local audio track(s).")

    def _on_assign_mood_tag(self):
        if not hasattr(self, "selected_song_id") or not self.selected_song_id:
            messagebox.showwarning("No Selection", "Please select a song row first.")
            return

        target_song = next((s for s in self.library_mgr.songs if s["id"] == self.selected_song_id), None)
        if not target_song:
            return

        dialog = AssignMoodDialog(self, target_song.get("mood", "Chill"), self.mood_mgr)
        self.wait_window(dialog.top)

        if dialog.selected_mood:
            self.library_mgr.set_song_mood(self.selected_song_id, dialog.selected_mood)
            self._build_mood_buttons()
            self._refresh_library_table()

    def _refresh_library_table(self):
        if not hasattr(self, "scroll_music_table"):
            return

        for item in self.scroll_music_table.winfo_children():
            item.destroy()

        search_query = self.search_var.get().strip().lower() if hasattr(self, "search_var") else ""

        self.playlist = []

        if self.active_view == "custom_playlist" and self.active_custom_playlist:
            allowed_ids = set(self.active_custom_playlist.get("song_ids", []))
            target_songs = [s for s in self.library_mgr.songs if s["id"] in allowed_ids]
        elif self.active_view == "playlist" and self.active_playlist_mood != "All Music":
            target_songs = [s for s in self.library_mgr.songs if s.get("mood", "").lower() == self.active_playlist_mood.lower()]
        else:
            target_songs = self.library_mgr.songs

        if not target_songs:
            empty_lbl = ctk.CTkLabel(
                self.scroll_music_table,
                text="No songs found in this view. Click '+ Add Music File' or '+ Add Folder' to import audio files.",
                font=ctk.CTkFont(size=12),
                text_color="#A1A1AA"
            )
            empty_lbl.pack(padx=20, pady=30)
            return

        for s in target_songs:
            t_title = s.get("title", "").lower()
            t_artist = s.get("artist", "").lower()
            t_album = s.get("album", "").lower()
            t_mood = s.get("mood", "").lower()

            if not search_query or search_query in t_title or search_query in t_artist or search_query in t_album or search_query in t_mood:
                self.playlist.append(s)

                row = ctk.CTkFrame(self.scroll_music_table, corner_radius=6)
                row.pack(fill="x", pady=2, padx=4)

                lbl_icon = ctk.CTkLabel(row, text="🎵", font=ctk.CTkFont(size=14))
                lbl_icon.pack(side="left", padx=(10, 8), pady=8)

                lbl_t = ctk.CTkLabel(row, text=s["title"], font=ctk.CTkFont(size=11, weight="bold"), anchor="w", width=200)
                lbl_t.pack(side="left", padx=4)

                lbl_a = ctk.CTkLabel(row, text=s["artist"], font=ctk.CTkFont(size=10), text_color="#A1A1AA", anchor="w", width=140)
                lbl_a.pack(side="left", padx=4)

                lbl_alb = ctk.CTkLabel(row, text=s.get("album", "Local Album"), font=ctk.CTkFont(size=10), text_color="#A1A1AA", anchor="w", width=120)
                lbl_alb.pack(side="left", padx=4)

                emoji = MOOD_EMOJIS.get(s["mood"], "")
                lbl_m = ctk.CTkLabel(row, text=f"{emoji} {s['mood']}", font=ctk.CTkFont(size=10, weight="bold"), text_color="#10B981", width=90)
                lbl_m.pack(side="left", padx=4)

                lbl_d = ctk.CTkLabel(row, text=s.get("duration", "03:30"), font=ctk.CTkFont(size=10), text_color="#A1A1AA")
                lbl_d.pack(side="right", padx=12)

                # Double click to play
                row.bind("<Double-1>", lambda e, song_obj=s: self._play_song_object(song_obj))
                lbl_t.bind("<Double-1>", lambda e, song_obj=s: self._play_song_object(song_obj))
                row.bind("<Button-1>", lambda e, song_obj=s: self._on_select_row(song_obj["id"]))
                lbl_t.bind("<Button-1>", lambda e, song_obj=s: self._on_select_row(song_obj["id"]))

    def _on_select_row(self, song_id: str):
        self.selected_song_id = song_id

    # =========================================================================
    # Audio Playback Engine
    # =========================================================================

    def _play_song_object(self, song: Dict[str, Any]):
        fpath = song.get("path")
        if not fpath or not os.path.exists(fpath):
            messagebox.showerror("File Missing", f"Audio file not found on disk:\n{fpath}")
            return

        self.current_song = song
        self.next_song_prepared = False

        try:
            self.current_song_idx = self.playlist.index(song)
        except ValueError:
            self.current_song_idx = -1

        try:
            self.audio_player.load(fpath)
            self.audio_player.play()
            self.btn_play_pause.configure(text="⏸", fg_color="#F59E0B")
            self.lbl_now_playing.configure(text=f"Now Playing:  {song['title']}  |  {song['artist']}")
            self.lbl_mood_badge.configure(text=f"Playlist : {song['mood']}")
        except Exception as e:
            messagebox.showerror("Playback Error", f"Failed to play MP3 file:\n{e}")

    def _on_toggle_play_pause(self):
        if not self.current_song:
            if self.playlist:
                self._play_song_object(self.playlist[0])
            else:
                messagebox.showinfo("Empty Playlist", "Please import audio files into your library.")
            return

        if self.audio_player.is_playing and not self.audio_player.is_paused:
            self.audio_player.pause()
            self.btn_play_pause.configure(text="▶", fg_color="#10B981")
        else:
            self.audio_player.play()
            self.btn_play_pause.configure(text="⏸", fg_color="#F59E0B")

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

    def _on_volume_change(self, value):
        try:
            vol_pct = float(value) / 100.0
            self.audio_player.set_volume(vol_pct)
        except ValueError:
            pass

    def _on_seek_progress(self, value):
        if self.audio_player.duration_sec > 0:
            try:
                pct = float(value) / 100.0
                target_sec = pct * self.audio_player.duration_sec
                self.audio_player.seek(target_sec)
            except ValueError:
                pass

    # =========================================================================
    # FUNCTIONAL MOODIFY AI ENGINE (Right Panel Switch Connection)
    # =========================================================================

    def _on_toggle_moodify(self):
        """Directly connected callback to the CTkSwitch in the right panel."""
        val = self.switch_moodify_var.get()
        if val == "on" and not self.moodify_on:
            self._start_moodify()
        elif val == "off" and self.moodify_on:
            self._stop_moodify()

    def _start_moodify(self):
        """Starts Moodify AI: initializes FaceDetector & Predictor, opens webcam, starts 10s initial analysis."""
        if self.detector is None:
            try:
                self.detector = FaceDetector()
            except Exception as e:
                messagebox.showerror("Detector Error", f"Failed to initialize FaceDetector:\n{e}")
                self.switch_moodify_var.set("off")
                return

        if self.predictor is None:
            try:
                self.predictor = FaceMoodPredictor()
            except Exception as e:
                messagebox.showerror("Predictor Error", f"Failed to initialize FaceMoodPredictor:\n{e}")
                self.switch_moodify_var.set("off")
                return

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", "Could not open webcam for Moodify mode.")
            self.switch_moodify_var.set("off")
            return

        self.moodify_on = True
        self.ai_state = "INITIAL_ANALYSIS"
        self.ai_analysis_start_time = time.time()
        self.ai_frame_counter = 0
        self.initial_analysis_moods.clear()
        self.background_mood_buffer.clear()
        self.next_queued_song = None
        self.next_song_prepared = False

        if hasattr(self, "switch_moodify"):
            self.switch_moodify.configure(text="MOODIFY [ ON ]")
        if hasattr(self, "lbl_det_mood"):
            self.lbl_det_mood.configure(text="AI Status: Analyzing mood... (10s)", text_color="#F59E0B")
        self._update_camera_moodify()

    def _stop_moodify(self):
        """Stops Moodify AI: releases camera hardware, closes detector, resets AI status."""
        self.moodify_on = False
        self.ai_state = "IDLE"
        self.next_queued_song = None
        self.next_song_prepared = False

        if hasattr(self, "switch_moodify"):
            self.switch_moodify.configure(text="MOODIFY [ OFF ]")
            self.switch_moodify_var.set("off")

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        if self.detector is not None:
            self.detector.close()
            self.detector = None

        self.predictor = None

        if hasattr(self, "cam_canvas"):
            self.cam_canvas.configure(
                image="",
                text="Camera OFF\n(Moodify is Disabled)",
                text_color="#A1A1AA"
            )
        if hasattr(self, "lbl_det_mood"):
            self.lbl_det_mood.configure(text="AI Status: Moodify OFF", text_color="#3B82F6")
        if hasattr(self, "lbl_det_playlist"):
            self.lbl_det_playlist.configure(text="PLAYING FROM: All Music")
        if hasattr(self, "lbl_next_queue_banner"):
            self.lbl_next_queue_banner.configure(text="Next: None", text_color="#A1A1AA")

    def _update_camera_moodify(self):
        if not self.moodify_on or self.cap is None or not self.cap.isOpened():
            return

        ret, frame_bgr = self.cap.read()
        if not ret or frame_bgr is None:
            self.after(50, self._update_camera_moodify)
            return

        frame_bgr = cv2.flip(frame_bgr, 1)

        # Throttled MediaPipe Landmark Detection (Every 2nd Frame for Low CPU Usage!)
        self.ai_frame_counter += 1
        if self.ai_frame_counter % self.FRAME_SKIP == 0 and self.detector is not None:
            landmarks, blendshapes = self.detector.detect(frame_bgr)
            face_detected = landmarks is not None

            if face_detected:
                frame_bgr = self.detector.draw_landmarks(frame_bgr, landmarks)
                feat_vec, _ = extract_facial_features(landmarks, blendshapes)

                if self.predictor is not None:
                    res = self.predictor.predict_vector(feat_vec)
                    raw_emo = res.emotion.upper()
                    detected_mood = EMOTION_TO_MOOD.get(raw_emo, "Chill")
                    self.ai_confidence = float(res.confidence * 100.0)

                    if self.ai_state == "INITIAL_ANALYSIS":
                        self.initial_analysis_moods.append(detected_mood)
                        elapsed = time.time() - self.ai_analysis_start_time
                        rem_sec = max(0, int(self.INITIAL_ANALYSIS_SECONDS - elapsed))
                        self.lbl_det_mood.configure(text=f"AI Status: Analyzing mood... ({rem_sec}s)", text_color="#F59E0B")

                        if elapsed >= self.INITIAL_ANALYSIS_SECONDS:
                            most_common_mood = Counter(self.initial_analysis_moods).most_common(1)
                            final_start_mood = most_common_mood[0][0] if most_common_mood else "Chill"
                            self.stable_mood = final_start_mood
                            self.ai_state = "PLAYING"

                            emoji = MOOD_EMOJIS.get(self.stable_mood, "😊")
                            self.lbl_det_mood.configure(text=f"CURRENT MOOD: {emoji} {self.stable_mood} ({int(self.ai_confidence)}%)", text_color="#10B981")
                            self._recommend_and_start_first_song(self.stable_mood)

                    elif self.ai_state == "PLAYING":
                        self.background_mood_buffer.append(detected_mood)
                        if len(self.background_mood_buffer) > 200:
                            self.background_mood_buffer.pop(0)

                        most_common = Counter(self.background_mood_buffer).most_common(1)
                        if most_common:
                            self.stable_mood = most_common[0][0]

                        emoji = MOOD_EMOJIS.get(self.stable_mood, "😊")
                        self.lbl_det_mood.configure(text=f"CURRENT MOOD: {emoji} {self.stable_mood} ({int(self.ai_confidence)}%)", text_color="#10B981")
            else:
                if self.ai_state == "INITIAL_ANALYSIS":
                    elapsed = time.time() - self.ai_analysis_start_time
                    rem_sec = max(0, int(self.INITIAL_ANALYSIS_SECONDS - elapsed))
                    self.lbl_det_mood.configure(text=f"AI Status: Position face... ({rem_sec}s)", text_color="#F59E0B")

        self.lbl_det_playlist.configure(text=f"PLAYING FROM: {self.stable_mood} Playlist")

        self._render_camera_preview(frame_bgr)
        self.after(50, self._update_camera_moodify)

    def _render_camera_preview(self, frame_bgr: np.ndarray):
        if not hasattr(self, "cam_canvas"):
            return
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

        img_ctk = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(new_w, new_h))
        self.cam_canvas.configure(image=img_ctk, text="")

    def _recommend_and_start_first_song(self, target_mood: str):
        if self.active_playlist_mood != target_mood:
            self._select_playlist(target_mood)

        matching_songs = self.library_mgr.get_songs_by_mood(target_mood)
        if not matching_songs:
            matching_songs = self.library_mgr.songs

        if matching_songs:
            self._play_song_object(matching_songs[0])

    def _prepare_next_song_queue(self):
        target_mood = self.stable_mood
        matching_songs = self.library_mgr.get_songs_by_mood(target_mood)
        if not matching_songs:
            matching_songs = self.library_mgr.songs

        if matching_songs:
            candidates = [s for s in matching_songs if s.get("id") != (self.current_song.get("id") if self.current_song else None)]
            chosen = candidates[0] if candidates else matching_songs[0]
            self.next_queued_song = chosen
            self.next_song_prepared = True
            self.lbl_next_queue_banner.configure(text=f"Next: {chosen['title']}", text_color="#10B981")

    def _main_timer_loop(self):
        if self.audio_player.is_playing:
            pos_sec = self.audio_player.get_position()
            dur_sec = self.audio_player.duration_sec

            if dur_sec > 0:
                pct = (pos_sec / dur_sec) * 100.0
                self.slider_progress.set(pct)

                rem_sec = dur_sec - pos_sec

                if self.audio_player.is_crossfading:
                    self.audio_player.update_crossfade_ramp()

                if self.moodify_on and self.ai_state == "PLAYING":
                    if rem_sec <= 12.0 and not self.next_song_prepared:
                        self._prepare_next_song_queue()

                    if rem_sec <= 4.0 and self.next_queued_song and not self.audio_player.is_crossfading:
                        next_song = self.next_queued_song
                        next_path = next_song.get("path", "")

                        next_dur = self.audio_player.start_crossfade(next_path, duration=4.0)
                        self.current_song = next_song
                        self.audio_player.duration_sec = next_dur
                        self.audio_player.start_time = time.time()
                        self.next_queued_song = None
                        self.next_song_prepared = False

                        self.lbl_now_playing.configure(text=f"Now Playing:  {next_song['title']}  |  {next_song['artist']}")
                        self.lbl_mood_badge.configure(text=f"Playlist : {next_song['mood']}")
                        self.lbl_next_queue_banner.configure(text="Next: None", text_color="#A1A1AA")

            m_pos, s_pos = divmod(int(pos_sec), 60)
            m_dur, s_dur = divmod(int(dur_sec), 60)

            self.lbl_time_elapsed.configure(text=f"{m_pos:02d}:{s_pos:02d}")
            self.lbl_time_total.configure(text=f"{m_dur:02d}:{s_dur:02d}")

        self.after(200, self._main_timer_loop)

    def _on_quit(self):
        self.audio_player.stop()
        if self.moodify_on:
            self._stop_moodify()
        self.destroy()


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    app = BGMMasterApp()
    app.mainloop()


if __name__ == "__main__":
    main()
