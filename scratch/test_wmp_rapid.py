import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bgm_master import WMPController, MusicLibraryManager

lib = MusicLibraryManager()
valid_songs = [s for s in lib.songs if os.path.exists(s.get("path", ""))]

if valid_songs:
    fpath = valid_songs[0]["path"]
    print("Test 1: Normal load_and_play once")
    wmp = WMPController()
    time.sleep(1) # wait for powershell init
    wmp.load_and_play(fpath)
    wmp.set_volume(0.8)
    time.sleep(3)
    wmp.stop()
    wmp.close()
    print("Test 1 complete.")

    print("\nTest 2: Rapid back-to-back load_and_play (simulating load() followed by play())")
    wmp = WMPController()
    time.sleep(1)
    # Rapid calls as done by load() + play()
    wmp.load_and_play(fpath)
    wmp.set_volume(0.8)
    wmp.load_and_play(fpath)
    wmp.set_volume(0.8)
    time.sleep(3)
    wmp.stop()
    wmp.close()
    print("Test 2 complete.")
