import unittest
import os
import sys
import wave
import uuid
import ctypes

# Import helper duration logic to test
def get_audio_duration(file_path: str) -> float:
    if not file_path or not os.path.exists(file_path):
        return 0.0
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".wav":
        try:
            with wave.open(file_path, "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                if rate > 0:
                    return float(frames) / float(rate)
        except Exception:
            pass

    if sys.platform == "win32":
        try:
            norm = os.path.normpath(os.path.abspath(file_path))
            buf = ctypes.create_unicode_buffer(512)
            ctypes.windll.kernel32.GetShortPathNameW(norm, buf, 512)
            short_path = buf.value if buf.value else norm

            alias = f"dur_{uuid.uuid4().hex[:8]}"
            device_type = "waveaudio" if ext == ".wav" else "mpegvideo"
            err = ctypes.windll.winmm.mciSendStringW(f'open "{short_path}" type {device_type} alias {alias}', None, 0, 0)
            if err != 0 and device_type == "mpegvideo":
                err = ctypes.windll.winmm.mciSendStringW(f'open "{short_path}" alias {alias}', None, 0, 0)

            if err == 0:
                buf_len = ctypes.create_unicode_buffer(512)
                ctypes.windll.winmm.mciSendStringW(f'status {alias} length', buf_len, 511, 0)
                ctypes.windll.winmm.mciSendStringW(f'close {alias}', None, 0, 0)
                val = buf_len.value.strip()
                if val.isdigit():
                    dur = float(val) / 1000.0
                    if dur > 0:
                        return dur
        except Exception:
            pass

    return 0.0


class TestAudioFixes(unittest.TestCase):

    def test_duration_calculation(self):
        wav_path = os.path.abspath("test_dur_sample.wav")
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(44100)
            # 3 seconds of audio = 44100 * 3 = 132300 frames
            wf.writeframes(b"\x00\x00\x00\x00" * 132300)

        dur = get_audio_duration(wav_path)
        if os.path.exists(wav_path):
            os.remove(wav_path)

        self.assertAlmostEqual(dur, 3.0, delta=0.1)

    def test_next_prev_index_logic(self):
        playlist = [
            {"id": "song_1", "title": "Song One"},
            {"id": "song_2", "title": "Song Two"},
            {"id": "song_3", "title": "Song Three"},
        ]
        
        # Current song is song_2
        current_song = {"id": "song_2", "title": "Song Two"}
        curr_id = current_song.get("id")
        
        curr_idx = next((i for i, s in enumerate(playlist) if s.get("id") == curr_id), -1)
        self.assertEqual(curr_idx, 1)
        
        # Next song index
        next_idx = (curr_idx + 1) % len(playlist)
        self.assertEqual(next_idx, 2)
        self.assertEqual(playlist[next_idx]["id"], "song_3")

        # Prev song index from song_1
        curr_idx = 0
        prev_idx = (curr_idx - 1) % len(playlist)
        self.assertEqual(prev_idx, 2)
        self.assertEqual(playlist[prev_idx]["id"], "song_3")


if __name__ == "__main__":
    unittest.main()
