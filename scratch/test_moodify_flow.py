import sys
import os
import unittest
from unittest.mock import MagicMock

# Add root directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bgm_master import BGMMasterApp, EMOTION_TO_MOOD, RELATED_MOODS

class TestMoodifyFlow(unittest.TestCase):
    def setUp(self):
        self.app = BGMMasterApp()
        self.app.withdraw() # Hide window during test

    def tearDown(self):
        self.app.audio_player.stop()
        self.app.destroy()

    def test_mock_emotions(self):
        print("\n--- Testing Mock Emotion Triggers ---")
        self.app._on_mock_emotion_click("Happy")
        self.assertEqual(self.app.stable_mood, "Happy")
        self.assertTrue(self.app.moodify_on)

        self.app._on_mock_emotion_click("Sad")
        self.assertEqual(self.app.stable_mood, "Sad")

        self.app._on_mock_emotion_click("Excited")
        self.assertEqual(self.app.stable_mood, "Excited")

        self.app._on_mock_emotion_click("Lonely")
        self.assertEqual(self.app.stable_mood, "Lonely")

        self.app._on_mock_emotion_click("Romantic")
        self.assertEqual(self.app.stable_mood, "Romantic")

        self.app._on_mock_emotion_click("Chill")
        self.assertEqual(self.app.stable_mood, "Chill")
        print("--- All Mock Emotion Triggers Passed Successfully! ---\n")

if __name__ == "__main__":
    unittest.main()
