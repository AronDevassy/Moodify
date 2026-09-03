"""
Unit test suite for BGM MASTER V1 manual registration and profile management subsystems.
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


class TestBGMMasterManualRegistrationSubsystems(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.moods_path = os.path.join(self.temp_dir.name, "moods.json")
        self.profiles_path = os.path.join(self.temp_dir.name, "profiles.json")
        self.library_path = os.path.join(self.temp_dir.name, "music_library.json")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_profile_creation_and_deletion(self):
        """Test profile creation, multi-vector addition, and deletion without affecting other files."""
        mgr = ProfileManager(filepath=self.profiles_path)
        initial_count = len(mgr.profiles)

        # Create profile 'Faheem'
        vec1 = np.ones(FEATURE_DIMENSION, dtype=np.float32) * 0.4
        vec2 = np.ones(FEATURE_DIMENSION, dtype=np.float32) * 0.42
        prof = mgr.add_profile("Faheem", [vec1, vec2])

        self.assertEqual(len(mgr.profiles), initial_count + 1)
        self.assertEqual(prof["name"], "Faheem")

        # Test face recognition match
        query_vec = np.ones(FEATURE_DIMENSION, dtype=np.float32) * 0.41
        rec_name, dist = mgr.recognize_face(query_vec, threshold=0.45)
        self.assertEqual(rec_name, "Faheem")

        # Test profile deletion
        deleted = mgr.delete_profile(prof["id"])
        self.assertTrue(deleted)
        self.assertEqual(len(mgr.profiles), initial_count)

        # Verify recognition returns Unknown User after deletion
        rec_name2, dist2 = mgr.recognize_face(query_vec, threshold=0.45)
        self.assertNotEqual(rec_name2, "Faheem")

    def test_music_library_isolation(self):
        """Test that profile operations leave music library completely intact."""
        lib_mgr = MusicLibraryManager(filepath=self.library_path)
        prof_mgr = ProfileManager(filepath=self.profiles_path)

        dummy_mp3 = os.path.join(self.temp_dir.name, "Track.mp3")
        with open(dummy_mp3, "wb") as f:
            f.write(b"AUDIO")

        song = lib_mgr.import_file(dummy_mp3, mood="Happy")
        self.assertIsNotNone(song)

        # Add & delete profile
        p = prof_mgr.add_profile("TestUser", [np.ones(FEATURE_DIMENSION, dtype=np.float32)])
        prof_mgr.delete_profile(p["id"])

        # Check music library is completely intact
        self.assertEqual(len(lib_mgr.songs), 1)
        self.assertEqual(lib_mgr.songs[0]["path"], dummy_mp3)


if __name__ == "__main__":
    unittest.main()
