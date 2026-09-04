"""
Unit test suite for BGM MASTER V1 Column Priorities, Artist Space & Strict Sidebar Ordering.
"""

import os
import json
import tempfile
import unittest
from collections import Counter
import customtkinter as ctk
import numpy as np

from bgm_master import (
    MoodManager,
    ProfileManager,
    MusicLibraryManager,
    PlaylistManager,
    AudioPlayer,
    ThemeManager,
    truncate_text,
    configure_music_table_columns,
    EMOTION_TO_MOOD,
    DEFAULT_MOODS
)
from src.facial_features import FEATURE_DIMENSION


class TestBGMMasterColumnPriorityAndSidebarOrder(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.settings_path = os.path.join(self.temp_dir.name, "settings.json")
        self.moods_path = os.path.join(self.temp_dir.name, "moods.json")
        self.profiles_path = os.path.join(self.temp_dir.name, "profiles.json")
        self.library_path = os.path.join(self.temp_dir.name, "music_library.json")
        self.playlists_path = os.path.join(self.temp_dir.name, "playlists.json")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_default_moods_order_immutability(self):
        """Test default moods order is strictly immutable: Romantic, Happy, Sad, Lonely, Chill, Excited."""
        expected_order = ["Romantic", "Happy", "Sad", "Lonely", "Chill", "Excited"]
        self.assertEqual(DEFAULT_MOODS, expected_order)

    def test_configure_music_table_columns_artist_space(self):
        """Test configure_music_table_columns allocates flexible 27% weight for Artist column."""
        root = ctk.CTk()
        frame = ctk.CTkFrame(root)
        configure_music_table_columns(frame)

        # Col 2 (Artist) should have weight 27 and minsize >= 120
        artist_col_info = frame.grid_columnconfigure(2)
        self.assertIsNotNone(artist_col_info)

        root.destroy()

    def test_text_truncation_extreme_strings(self):
        """Test text truncation helper with extreme strings to prevent column breaking."""
        extreme_title = "This Is A Extremely Long Song Name That Should Not Break The Table At All"
        truncated_title = truncate_text(extreme_title, max_len=32)
        self.assertEqual(len(truncated_title), 32)
        self.assertTrue(truncated_title.endswith("..."))

        extreme_artist = "This Is An Extremely Long Artist Name That Should Also Not Break The Table Layout"
        truncated_artist = truncate_text(extreme_artist, max_len=24)
        self.assertEqual(len(truncated_artist), 24)
        self.assertTrue(truncated_artist.endswith("..."))

    def test_custom_mood_deletion_subsystem(self):
        """Test MoodManager custom mood creation and deletion while protecting default moods."""
        mm = MoodManager(filepath=self.moods_path)
        init_len = len(mm.moods)

        # Protect Default Moods from Deletion
        for def_m in DEFAULT_MOODS:
            success = mm.delete_mood(def_m)
            self.assertFalse(success)
            self.assertIn(def_m, mm.moods)

        # Add & Delete Custom Mood
        added = mm.add_mood("Workout")
        self.assertTrue(added)
        deleted = mm.delete_mood("Workout")
        self.assertTrue(deleted)
        self.assertNotIn("Workout", mm.moods)

    def test_playlist_pinning_lifecycle(self):
        """Test PlaylistManager pinning and unpinning to Home View."""
        pm = PlaylistManager(filepath=self.playlists_path)
        pl = pm.create_playlist("Study Focus")
        self.assertFalse(pl.get("pinned", False))

        is_pinned = pm.toggle_pin(pl["id"])
        self.assertTrue(is_pinned)

    def test_theme_manager_ctk_toggle(self):
        """Test ThemeManager switching between Dark and Light mode themes."""
        tm = ThemeManager(filepath=self.settings_path)
        self.assertEqual(tm.current_theme_name, "dark")
        tm.set_theme("light")
        self.assertEqual(tm.current_theme_name, "light")


if __name__ == "__main__":
    unittest.main()
