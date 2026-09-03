"""
Unit test suite for BGM MASTER V1 polished subsystems:
MoodManager, ProfileManager, MusicLibraryManager, AudioPlayer, and Emotion-to-Mood mapping.
"""

import os
import json
import tempfile
import unittest
import numpy as np

from bgm_master import (
    MoodManager,
    ProfileManager,
    MusicLibraryManager,
    AudioPlayer,
    EMOTION_TO_MOOD,
    DEFAULT_MOODS
)
from src.facial_features import FEATURE_DIMENSION


class TestBGMMasterV1PolishedSubsystems(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.moods_path = os.path.join(self.temp_dir.name, "moods.json")
        self.profiles_path = os.path.join(self.temp_dir.name, "profiles.json")
        self.library_path = os.path.join(self.temp_dir.name, "music_library.json")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_mood_manager_custom_moods(self):
        """Test default moods and adding custom user moods."""
        mgr = MoodManager(filepath=self.moods_path)
        self.assertEqual(mgr.moods[:len(DEFAULT_MOODS)], DEFAULT_MOODS)

        added = mgr.add_mood("Workout")
        self.assertTrue(added)
        self.assertIn("Workout", mgr.moods)

        mgr_reloaded = MoodManager(filepath=self.moods_path)
        self.assertIn("Workout", mgr_reloaded.moods)

    def test_profile_manager_multi_vector_automated(self):
        """Test profile creation with automated multi-frame vectors."""
        mgr = ProfileManager(filepath=self.profiles_path)
        
        # Create profile 'Faheem' with 40 recorded expression samples (10 per step)
        samples = [np.ones(FEATURE_DIMENSION, dtype=np.float32) * (0.3 + 0.01 * (i % 4)) for i in range(40)]
        prof = mgr.add_profile("Faheem", samples)

        self.assertEqual(prof["name"], "Faheem")
        self.assertEqual(len(prof["feature_vectors"]), 40)

        # Test face recognition match
        query_vec = np.ones(FEATURE_DIMENSION, dtype=np.float32) * 0.31
        rec_name, dist = mgr.recognize_face(query_vec, threshold=0.45)
        self.assertEqual(rec_name, "Faheem")

        # Test unknown face
        diff_vec = np.zeros(FEATURE_DIMENSION, dtype=np.float32)
        rec_name2, dist2 = mgr.recognize_face(diff_vec, threshold=0.1)
        self.assertEqual(rec_name2, "Unknown User")

    def test_music_library_folder_import(self):
        """Test recursively scanning and importing a folder of audio files."""
        mgr = MusicLibraryManager(filepath=self.library_path)

        sub_folder = os.path.join(self.temp_dir.name, "AlbumFolder")
        os.makedirs(sub_folder, exist_ok=True)

        song1_path = os.path.join(sub_folder, "Artist A - Track 1.mp3")
        song2_path = os.path.join(sub_folder, "Track 2.wav")

        with open(song1_path, "wb") as f:
            f.write(b"MOCK_AUDIO_1")
        with open(song2_path, "wb") as f:
            f.write(b"MOCK_AUDIO_2")

        count = mgr.import_folder(sub_folder, default_mood="Happy")
        self.assertEqual(count, 2)
        self.assertEqual(len(mgr.songs), 2)

        happy_songs = mgr.get_songs_by_mood("Happy")
        self.assertEqual(len(happy_songs), 2)

    def test_audio_player_instantiation(self):
        """Test AudioPlayer initializes smoothly."""
        player = AudioPlayer()
        self.assertFalse(player.is_playing)
        self.assertFalse(player.is_paused)


if __name__ == "__main__":
    unittest.main()
