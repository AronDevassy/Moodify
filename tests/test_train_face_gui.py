"""
Unit tests for train_face_gui.py dataset collector GUI logic and CSV persistence.
"""

import os
import tempfile
import csv
import unittest
import numpy as np
import tkinter as tk

from train_face_gui import FacialDatasetGUI, AppState
from src.facial_features import FEATURE_NAMES, FEATURE_DIMENSION


class TestFacialDatasetGUI(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory for dataset testing
        self.test_dir = tempfile.TemporaryDirectory()
        self.dataset_path = os.path.join(self.test_dir.name, "test_dataset.csv")

        # Headless Tk root
        self.root = tk.Tk()
        self.root.withdraw()  # Hide GUI window during unit tests

        # Instantiate GUI with headless root and test dataset path
        self.app = FacialDatasetGUI(self.root, dataset_path=self.dataset_path)

    def tearDown(self):
        try:
            self.app._on_quit()
        except Exception:
            pass
        self.test_dir.cleanup()

    def test_initial_state(self):
        """Verify default initial app state is LIVE."""
        self.assertEqual(self.app.state, AppState.LIVE)
        self.assertEqual(len(self.app.session_samples), 0)
        self.assertIsNone(self.app.selected_label)

    def test_state_transitions(self):
        """Test LIVE -> RECORDING -> LABELING state machine transitions."""
        # 1. LIVE -> RECORDING
        self.app._on_toggle_recording()
        self.assertEqual(self.app.state, AppState.RECORDING)

        # Mock sample collection
        mock_sample = np.ones(FEATURE_DIMENSION, dtype=np.float32)
        self.app.session_samples.append(mock_sample)
        self.assertEqual(len(self.app.session_samples), 1)

        # 2. RECORDING -> LABELING
        self.app._on_toggle_recording()
        self.assertEqual(self.app.state, AppState.LABELING)

    def test_save_behavior(self):
        """Test selecting a label and saving samples to CSV dataset."""
        self.app._on_toggle_recording()  # Start recording
        mock_sample_1 = np.ones(FEATURE_DIMENSION, dtype=np.float32) * 0.5
        mock_sample_2 = np.ones(FEATURE_DIMENSION, dtype=np.float32) * 0.8
        self.app.session_samples.extend([mock_sample_1, mock_sample_2])

        self.app._on_toggle_recording()  # Stop recording -> LABELING

        # Select emotion
        self.app.emotion_var.set("happy")
        self.app._on_save()

        # Check returned to LIVE state
        self.assertEqual(self.app.state, AppState.LIVE)
        self.assertEqual(len(self.app.session_samples), 0)

        # Verify CSV file contents
        self.assertTrue(os.path.exists(self.dataset_path))
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            self.assertEqual(header[:3], ["session_id", "timestamp", "label"])
            self.assertEqual(header[3:], FEATURE_NAMES)

            rows = list(reader)
            self.assertEqual(len(rows), 2)
            for row in rows:
                self.assertEqual(row[2], "happy")
                self.assertTrue(row[0].startswith("sess_"))

    def test_discard_behavior(self):
        """Test discarding session samples without writing to CSV."""
        self.app._on_toggle_recording()  # Start recording
        self.app.session_samples.append(np.ones(FEATURE_DIMENSION, dtype=np.float32))
        self.app._on_toggle_recording()  # Stop recording -> LABELING

        self.app._on_discard()

        # Check returned to LIVE state and buffer cleared
        self.assertEqual(self.app.state, AppState.LIVE)
        self.assertEqual(len(self.app.session_samples), 0)
        self.assertFalse(os.path.exists(self.dataset_path))


if __name__ == "__main__":
    unittest.main()
