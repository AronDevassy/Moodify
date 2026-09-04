import sys
import os
import time
import ctypes

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bgm_master import MusicLibraryManager

lib = MusicLibraryManager()
valid_songs = [s for s in lib.songs if os.path.exists(s.get("path", ""))]

if len(valid_songs) < 2:
    print("Need at least 2 songs to test crossfade!")
    sys.exit(1)

song1 = valid_songs[0]["path"]
song2 = valid_songs[1]["path"]

class NativeMCIAudioPlayer:
    def __init__(self):
        self.current_file = None
        self.is_playing = False
        self.is_paused = False
        self.volume = 0.8
        self.duration_sec = 0.0
        self.start_time = 0.0
        self.pause_offset = 0.0

        self.alias_a = "bgm_master_mci_a"
        self.alias_b = "bgm_master_mci_b"
        self.active_alias = self.alias_a
        self.standby_alias = self.alias_b

        self.is_crossfading = False
        self.crossfade_start_time = 0.0
        self.crossfade_duration = 4.0

    def _mci_send(self, cmd: str) -> str:
        if sys.platform != "win32":
            return ""
        buf = ctypes.create_unicode_buffer(512)
        err = ctypes.windll.winmm.mciSendStringW(cmd, buf, 511, 0)
        if err != 0:
            err_buf = ctypes.create_unicode_buffer(512)
            ctypes.windll.winmm.mciGetErrorStringW(err, err_buf, 511)
            print(f"[MCI Error {err}] CMD: '{cmd}' -> {err_buf.value}")
        return buf.value

    def _get_short_path(self, path: str) -> str:
        if sys.platform != "win32":
            return path
        norm = os.path.normpath(os.path.abspath(path))
        buf = ctypes.create_unicode_buffer(512)
        ctypes.windll.kernel32.GetShortPathNameW(norm, buf, 512)
        return buf.value if buf.value else norm

    def load(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        self.stop()
        self.current_file = os.path.normpath(os.path.abspath(file_path))
        self.is_playing = False
        self.is_paused = False
        self.pause_offset = 0.0

        short_path = self._get_short_path(self.current_file)
        self._mci_send(f'close {self.active_alias}')
        self._mci_send(f'open "{short_path}" type mpegvideo alias {self.active_alias}')
        
        len_str = self._mci_send(f'status {self.active_alias} length')
        try:
            self.duration_sec = float(len_str) / 1000.0 if len_str else 210.0
        except ValueError:
            self.duration_sec = 210.0

    def play(self):
        if not self.current_file:
            return

        if self.is_paused:
            self._mci_send(f'resume {self.active_alias}')
            self.is_playing = True
            self.is_paused = False
            self.start_time = time.time() - self.pause_offset
            return

        self._mci_send(f'play {self.active_alias} from 0')
        self.set_volume(self.volume)

        self.is_playing = True
        self.is_paused = False
        self.start_time = time.time()
        self.pause_offset = 0.0

    def pause(self):
        if self.is_playing and not self.is_paused:
            self._mci_send(f'pause {self.active_alias}')
            self.is_paused = True
            self.pause_offset = time.time() - self.start_time

    def stop(self):
        self._mci_send(f'stop {self.active_alias}')
        self._mci_send(f'close {self.active_alias}')
        self._mci_send(f'stop {self.standby_alias}')
        self._mci_send(f'close {self.standby_alias}')
        self.is_playing = False
        self.is_paused = False
        self.is_crossfading = False
        self.pause_offset = 0.0

    def set_volume(self, volume: float):
        self.volume = max(0.0, min(1.0, volume))
        vol_int = int(self.volume * 1000)
        self._mci_send(f'setaudio {self.active_alias} volume to {vol_int}')

    def get_position(self) -> float:
        if not self.is_playing:
            return 0.0
        if self.is_paused:
            return self.pause_offset

        pos_str = self._mci_send(f'status {self.active_alias} position')
        try:
            if pos_str:
                return float(pos_str) / 1000.0
        except ValueError:
            pass
        return max(0.0, time.time() - self.start_time)

    def seek(self, seconds: float):
        if not self.current_file:
            return
        target_ms = int(seconds * 1000)
        if self.is_playing and not self.is_paused:
            self._mci_send(f'play {self.active_alias} from {target_ms}')
        else:
            self._mci_send(f'seek {self.active_alias} to {target_ms}')
        self.start_time = time.time() - seconds
        self.pause_offset = seconds

    def start_crossfade(self, next_file_path: str, duration: float = 4.0) -> float:
        if not os.path.exists(next_file_path):
            return 0.0

        self.is_crossfading = True
        self.crossfade_start_time = time.time()
        self.crossfade_duration = duration

        short_next = self._get_short_path(next_file_path)
        self._mci_send(f'close {self.standby_alias}')
        self._mci_send(f'open "{short_next}" type mpegvideo alias {self.standby_alias}')
        self._mci_send(f'setaudio {self.standby_alias} volume to 0')
        self._mci_send(f'play {self.standby_alias} from 0')

        len_str = self._mci_send(f'status {self.standby_alias} length')
        try:
            next_dur = float(len_str) / 1000.0 if len_str else 210.0
        except ValueError:
            next_dur = 210.0

        self.current_file = os.path.normpath(os.path.abspath(next_file_path))
        return next_dur

    def update_crossfade_ramp(self):
        if not self.is_crossfading:
            return

        elapsed = time.time() - self.crossfade_start_time
        progress = min(1.0, elapsed / max(0.1, self.crossfade_duration))

        vol_active = int((1.0 - progress) * self.volume * 1000)
        vol_standby = int(progress * self.volume * 1000)
        self._mci_send(f'setaudio {self.active_alias} volume to {vol_active}')
        self._mci_send(f'setaudio {self.standby_alias} volume to {vol_standby}')

        if progress >= 1.0:
            self._mci_send(f'stop {self.active_alias}')
            self._mci_send(f'close {self.active_alias}')
            self.active_alias, self.standby_alias = self.standby_alias, self.active_alias
            self.is_crossfading = False
            self.start_time = time.time()

print("Testing NativeMCIAudioPlayer with normalized paths...")
player = NativeMCIAudioPlayer()
print(f"Loading song 1: {valid_songs[0]['title']}")
player.load(song1)
print(f"Song duration: {player.duration_sec:.2f}s")
print("Playing song 1...")
player.play()

time.sleep(2)
print(f"Position: {player.get_position():.2f}s")

print(f"Loading and crossfading to song 2: {valid_songs[1]['title']}")
next_dur = player.start_crossfade(song2, duration=3.0)
start_t = time.time()
while time.time() - start_t < 3.5:
    player.update_crossfade_ramp()
    time.sleep(0.1)

print("Crossfade completed. Stopping player.")
player.stop()
print("Test completed successfully!")
