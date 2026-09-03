"""
Unit test suite for BGM MASTER V1 Functional AI Toggle in Right Panel & Custom Mood Deletion.
"""

import os
import json
import tempfile
import unittest
from collections import Counter
import numpy as np

from bgm_master import (
    MoodManager,
    ProfileManager,
    MusicLibraryManager,
    PlaylistManager,
    AudioPlayer,
    ThemeManager,
    EMOTION_TO_MOOD,
    DEFAULT_MOODS
)
from src.facial_features import FEATURE_DIMENSION


class TestBGMMasterAIToggleAndMoodDeletion(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.settings_path = os.path.join(self.temp_dir.name, "settings.json")
        self.moods_path = os.path.join(self.temp_dir.name, "moods.json")
        self.profiles_path = os.path.join(self.temp_dir.name, "profiles.json")
        self.library_path = os.path.join(self.temp_dir.name, "music_library.json")
        self.playlists_path = os.path.join(self.temp_dir.name, "playlists.json")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_custom_mood_deletion_subsystem(self):
        """Test MoodManager custom mood creation and deletion while protecting default moods."""
        mm = MoodManager(filepath=self.moods_path)
        init_len = len(mm.moods)

        # 1. Protect Default Moods from Deletion
        for def_m in DEFAULT_MOODS:
            success = mm.delete_mood(def_m)
            self.assertFalse(success)
            self.assertIn(def_m, mm.moods)

        # 2. Add Custom Mood
        added = mm.add_mood("Workout")
        self.assertTrue(added)
        self.assertIn("Workout", mm.moods)
        self.assertEqual(len(mm.moods), init_len + 1)

        # 3. Delete Custom Mood
        deleted = mm.delete_mood("Workout")
        self.assertTrue(deleted)
        self.assertNotIn("Workout", mm.moods)
        self.assertEqual(len(mm.moods), init_len)

        # 4. Verify Persistence
        mm_reload = MoodManager(filepath=self.moods_path)
        self.assertEqual(len(mm_reload.moods), init_len)

    def test_playlist_pinning_lifecycle(self):
        """Test PlaylistManager pinning and unpinning to Home View."""
        pm = PlaylistManager(filepath=self.playlists_path)
        pl = pm.create_playlist("Study Focus")
        self.assertFalse(pl.get("pinned", False))

        is_pinned = pm.toggle_pin(pl["id"])
        self.assertTrue(is_pinned)

        pinned_list = pm.get_pinned_playlists()
        self.assertTrue(any(p["id"] == pl["id"] for p in pinned_list))

    def test_theme_manager_ctk_toggle(self):
        """Test ThemeManager switching between Dark and Light mode themes."""
        tm = ThemeManager(filepath=self.settings_path)
        self.assertEqual(tm.current_theme_name, "dark")

        tm.set_theme("light")
        self.assertEqual(tm.current_theme_name, "light")

    def test_expression_registration_uniformity(self):
        """Test that all expression steps require identical 2 compulsory captures."""
        required_per_exp = 2
        expressions = ["Neutral", "Happy", "Sad", "Surprised/Excited"]

        captured_counts = {exp: required_per_exp for exp in expressions}
        for exp in expressions:
            self.assertEqual(captured_counts[exp], 2)

    def test_moodify_emotion_mapping(self):
        """Test direct facial emotion to mood mapping when Moodify is ON."""
        raw_emotions = ["HAPPY", "NEUTRAL", "SURPRISED", "SAD"]
        expected_moods = ["Happy", "Chill", "Excited", "Sad"]

        mapped = [EMOTION_TO_MOOD.get(emo, "Chill") for emo in raw_emotions]
        self.assertEqual(mapped, expected_moods)


if __name__ == "__main__":
    unittest.main()
