import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bgm_master import BGMMasterApp

print("Initializing BGMMasterApp...")
app = BGMMasterApp()
app.withdraw() # hide GUI window

print(f"App initialized. Audio player engine - use_pygame: {app.audio_player.use_pygame}, use_wmp: {app.audio_player.use_wmp}")
print(f"Playlist len: {len(app.playlist)}")

if app.playlist:
    song = app.playlist[0]
    print(f"\n--- NORMAL MODE TEST ---")
    print(f"Selecting song: {song['title']}")
    print(f"Calling _on_select_and_play_row...")
    app._on_select_and_play_row(song)

    print(f"audio_player.current_file: {app.audio_player.current_file}")
    print(f"audio_player.is_playing: {app.audio_player.is_playing}")
    print(f"audio_player.is_paused: {app.audio_player.is_paused}")
    print(f"audio_player.volume: {app.audio_player.volume}")
    print(f"current_song: {app.current_song.get('title') if app.current_song else None}")

    time.sleep(2)

    print("\nSimulating user clicking Play/Pause button...")
    app._on_toggle_play_pause()
    print(f"After Play/Pause toggle - is_playing: {app.audio_player.is_playing}, is_paused: {app.audio_player.is_paused}")

    time.sleep(1)

    print("\nSimulating user clicking Play/Pause button again (Resume)...")
    app._on_toggle_play_pause()
    print(f"After Play/Pause toggle - is_playing: {app.audio_player.is_playing}, is_paused: {app.audio_player.is_paused}")

app.audio_player.stop()
if hasattr(app.audio_player, "wmp") and app.audio_player.wmp:
    app.audio_player.wmp.close()
app.destroy()
print("App test finished.")
