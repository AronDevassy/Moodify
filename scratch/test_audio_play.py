import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bgm_master import AudioPlayer, MusicLibraryManager

lib = MusicLibraryManager()
valid_songs = [s for s in lib.songs if os.path.exists(s.get("path", ""))]

print(f"Found {len(valid_songs)} valid songs with existing files.")

if valid_songs:
    song = valid_songs[0]
    fpath = song["path"]
    print(f"Testing playback for: {song['title']}")
    print(f"Path: {fpath}")

    player = AudioPlayer()
    print("Loading song...")
    player.load(fpath)
    print("Playing song...")
    player.play()
    print("Sleeping 3 seconds to check if audio is playing...")
    time.sleep(3)
    print(f"is_playing: {player.is_playing}, get_position: {player.get_position()}")
    player.stop()
    if hasattr(player, "wmp") and player.wmp:
        player.wmp.close()
    print("Done.")
