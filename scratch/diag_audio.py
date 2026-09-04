import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bgm_master import AudioPlayer, MusicLibraryManager

print("Initializing AudioPlayer...")
player = AudioPlayer()
print(f"use_pygame: {player.use_pygame}")
print(f"use_wmp: {player.use_wmp}")
print(f"volume: {player.volume}")

lib = MusicLibraryManager()
print(f"Loaded {len(lib.songs)} songs in library.")

for song in lib.songs[:5]:
    fpath = song.get("path")
    exists = os.path.exists(fpath) if fpath else False
    print(f"Song: {song.get('title')} | Path: {fpath} | Exists: {exists}")
