import sys
import os
import time
import ctypes

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bgm_master import MusicLibraryManager

lib = MusicLibraryManager()
valid_songs = [s for s in lib.songs if os.path.exists(s.get("path", ""))]

if not valid_songs:
    print("No valid songs found!")
    sys.exit(1)

song_path = valid_songs[0]["path"]
print(f"Testing real audio playback for song: {valid_songs[0]['title']}")
print(f"File path: {song_path}")

# Test 1: WinMM MCI (ctypes windll.winmm)
print("\n--- TEST 1: Native Windows WinMM MCI ---")
try:
    buf = ctypes.create_unicode_buffer(512)
    ctypes.windll.kernel32.GetShortPathNameW(song_path, buf, 512)
    short_path = buf.value if buf.value else song_path
    
    cmd_open = f'open "{short_path}" type mpegvideo alias test_mci'
    res_open = ctypes.windll.winmm.mciSendStringW(cmd_open, None, 0, 0)
    print(f"MCI open result: {res_open}")
    
    cmd_play = 'play test_mci from 0'
    res_play = ctypes.windll.winmm.mciSendStringW(cmd_play, None, 0, 0)
    print(f"MCI play result: {res_play}")
    
    time.sleep(3)
    ctypes.windll.winmm.mciSendStringW('stop test_mci', None, 0, 0)
    ctypes.windll.winmm.mciSendStringW('close test_mci', None, 0, 0)
    print("MCI test finished.")
except Exception as e:
    print(f"MCI exception: {e}")

# Test 2: Pygame
print("\n--- TEST 2: Pygame Mixer ---")
try:
    import pygame
    pygame.mixer.init()
    pygame.mixer.music.load(song_path)
    pygame.mixer.music.play()
    print("Pygame playing...")
    time.sleep(3)
    pygame.mixer.music.stop()
    print("Pygame test finished.")
except Exception as e:
    print(f"Pygame failed: {e}")

# Test 3: PowerShell WMP COM
print("\n--- TEST 3: PowerShell WMP COM ---")
import subprocess
try:
    proc = subprocess.Popen(
        ["powershell", "-NoExit", "-Command", "-"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    proc.stdin.write('$wmp = New-Object -ComObject WMPlayer.OCX\n')
    proc.stdin.write('$wmp.settings.autoStart = $true\n')
    proc.stdin.write(f'$wmp.URL = "{song_path}"\n')
    proc.stdin.write('$wmp.controls.play()\n')
    proc.stdin.flush()
    time.sleep(3)
    proc.terminate()
    print("WMP test finished.")
except Exception as e:
    print(f"WMP test failed: {e}")
