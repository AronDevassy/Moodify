import sys
import os
import time
import subprocess

p1 = r"C:\Users\ASUS\Music\TAMIL NEW\Adaavadi Video Song  LIK  Pradeep Ranganathan  Anirudh Ravichander  Krithi Shetty.mp3"
p2 = r"C:\Users\ASUS\Music\ENGLISH\02 Maroon 5 - Payphone.mp3"

class NetAudioEngine:
    """Robust Windows Media Engine using PowerShell System.Windows.Media.MediaPlayer (.NET)."""

    def __init__(self):
        self.proc = None
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
                self.proc = None

    def _send(self, cmd: str):
        if self.proc and self.proc.stdin and not self.proc.poll():
            try:
                self.proc.stdin.write(cmd + "\n")
                self.proc.stdin.flush()
            except Exception as e:
                print(f"[NetAudioEngine] Send failed: {e}")

    def load(self, file_path: str):
        abs_p = os.path.normpath(os.path.abspath(file_path)).replace("'", "''")
        self._send(f"$active.Open([System.Uri]'{abs_p}')")

    def play(self, volume: float = 0.8):
        self._send(f"$active.Volume = {volume:.2f}")
        self._send("$active.Play()")

    def pause(self):
        self._send("$active.Pause()")

    def resume(self):
        self._send("$active.Play()")

    def stop(self):
        self._send("$active.Stop()")
        self._send("$active.Close()")
        self._send("$standby.Stop()")
        self._send("$standby.Close()")

    def set_volume(self, volume: float):
        self._send(f"$active.Volume = {volume:.2f}")

    def seek(self, seconds: float):
        self._send(f"$active.Position = [System.TimeSpan]::FromSeconds({seconds:.2f})")

    def start_crossfade(self, next_file_path: str, volume: float = 0.8):
        abs_p = os.path.normpath(os.path.abspath(next_file_path)).replace("'", "''")
        self._send(f"$standby.Open([System.Uri]'{abs_p}')")
        self._send("$standby.Volume = 0.0")
        self._send("$standby.Play()")

    def update_crossfade_ramp(self, progress: float, max_vol: float):
        vol_act = max(0.0, min(1.0, (1.0 - progress) * max_vol))
        vol_sb = max(0.0, min(1.0, progress * max_vol))
        self._send(f"$active.Volume = {vol_act:.2f}")
        self._send(f"$standby.Volume = {vol_sb:.2f}")

    def finish_crossfade(self):
        self._send("$active.Stop()")
        self._send("$active.Close()")
        self._send("$tmp = $active; $active = $standby; $standby = $tmp")

    def close(self):
        if self.proc:
            try:
                self.stop()
                self._send("exit")
                self.proc.terminate()
            except Exception:
                pass

print("Testing NetAudioEngine crossfade...")
engine = NetAudioEngine()
print(f"Loading song 1: {p1}")
engine.load(p1)
engine.play(volume=0.8)
print("Playing song 1...")
time.sleep(2)

print(f"Starting crossfade to song 2: {p2}")
engine.start_crossfade(p2, volume=0.8)

start_t = time.time()
dur = 3.0
while time.time() - start_t < dur:
    prog = (time.time() - start_t) / dur
    engine.update_crossfade_ramp(prog, 0.8)
    time.sleep(0.1)

engine.finish_crossfade()
print("Crossfade finished! Playing song 2...")
time.sleep(2)

print("Stopping engine.")
engine.close()
print("Test completed successfully!")
