import sys
import os
import time
import subprocess

p1 = r"C:\Users\ASUS\Music\TAMIL NEW\Adaavadi Video Song  LIK  Pradeep Ranganathan  Anirudh Ravichander  Krithi Shetty.mp3"
p2 = r"C:\Users\ASUS\Music\ENGLISH\02 Maroon 5 - Payphone.mp3"

class NetAudioEngine:
    """Robust .NET System.Windows.Media.MediaPlayer audio controller via persistent PowerShell stdin."""

    def __init__(self):
        self.proc = None
        self.current_file = None
        self.is_playing = False
        self.is_paused = False
        self.volume = 0.8
        self.duration_sec = 210.0
        self.start_time = 0.0
        self.pause_offset = 0.0

        if sys.platform == "win32":
            try:
                self.proc = subprocess.Popen(
                    ["powershell", "-NoExit", "-Command", "-"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1
                )
                self._send("Add-Type -AssemblyName PresentationCore")
                self._send("$p1 = New-Object System.Windows.Media.MediaPlayer")
                self._send("$p2 = New-Object System.Windows.Media.MediaPlayer")
                self._send("$active = $p1")
                self._send("$standby = $p2")
                print("[NetAudioEngine] PowerShell .NET MediaPlayer initialized.")
            except Exception as e:
                print(f"[NetAudioEngine] Init failed: {e}")

    def _send(self, cmd: str):
        if self.proc and self.proc.stdin and not self.proc.poll():
            try:
                self.proc.stdin.write(cmd + "\n")
                self.proc.stdin.flush()
            except Exception as e:
                print(f"[NetAudioEngine] Send failed: {e}")

    def load(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        self.stop()
        self.current_file = os.path.normpath(os.path.abspath(file_path))
        self.is_playing = False
        self.is_paused = False
        self.pause_offset = 0.0

        esc_path = self.current_file.replace("'", "''")
        self._send(f"$active.Open([System.Uri]'{esc_path}')")
        self._send(f"$active.Volume = {self.volume}")
        self.duration_sec = 210.0

    def play(self):
        if not self.current_file:
            return

        if self.is_paused:
            self._send("$active.Play()")
            self.is_playing = True
            self.is_paused = False
            self.start_time = time.time() - self.pause_offset
            return

        self._send("$active.Play()")
        self._send(f"$active.Volume = {self.volume}")
        self.is_playing = True
        self.is_paused = False
        self.start_time = time.time()
        self.pause_offset = 0.0

    def pause(self):
        if self.is_playing and not self.is_paused:
            self._send("$active.Pause()")
            self.is_paused = True
            self.pause_offset = time.time() - self.start_time

    def stop(self):
        self._send("$active.Stop()")
        self._send("$active.Close()")
        self._send("$standby.Stop()")
        self._send("$standby.Close()")
        self.is_playing = False
        self.is_paused = False
        self.pause_offset = 0.0

    def set_volume(self, volume: float):
        self.volume = max(0.0, min(1.0, volume))
        self._send(f"$active.Volume = {self.volume}")

    def get_position(self) -> float:
        if not self.is_playing:
            return 0.0
        if self.is_paused:
            return self.pause_offset
        return max(0.0, time.time() - self.start_time)

    def seek(self, seconds: float):
        if not self.current_file:
            return
        self._send(f"$active.Position = [System.TimeSpan]::FromSeconds({seconds:.2f})")
        self.start_time = time.time() - seconds
        self.pause_offset = seconds

    def close(self):
        if self.proc:
            try:
                self.stop()
                self._send("exit")
                self.proc.terminate()
            except Exception:
                pass

print("Testing NetAudioEngine with Payphone.mp3...")
engine = NetAudioEngine()
engine.load(p2) # Payphone.mp3!
print(f"Loaded: {p2}")
engine.play()
print("Playing Payphone.mp3...")
time.sleep(3)
print("Pausing...")
engine.pause()
time.sleep(1)
print("Resuming...")
engine.play()
time.sleep(2)
print("Seeking to 15s...")
engine.seek(15.0)
time.sleep(3)
print("Stopping engine.")
engine.close()
print("NetAudioEngine test completed!")
