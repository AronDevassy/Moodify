import sys
import os
import time
import ctypes

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bgm_master import MusicLibraryManager

lib = MusicLibraryManager()
valid_songs = [s for s in lib.songs if os.path.exists(s.get("path", ""))]

for song in valid_songs[:5]:
    song_path = os.path.normpath(os.path.abspath(song["path"]))
    print(f"\nTesting song: {song['title']}")
    print(f"Path: {song_path}")
    
    # Send open with quotes on normpath directly
    buf = ctypes.create_unicode_buffer(512)
    alias = "test_song_alias"
    ctypes.windll.winmm.mciSendStringW(f'close {alias}', None, 0, 0)
    
    cmd_open = f'open "{song_path}" type mpegvideo alias {alias}'
    err_open = ctypes.windll.winmm.mciSendStringW(cmd_open, None, 0, 0)
    print(f"Open result: {err_open}")
    if err_open != 0:
        err_buf = ctypes.create_unicode_buffer(512)
        ctypes.windll.winmm.mciGetErrorStringW(err_open, err_buf, 511)
        print(f"Open Error msg: {err_buf.value}")
    else:
        # Get length
        len_buf = ctypes.create_unicode_buffer(512)
        ctypes.windll.winmm.mciSendStringW(f'status {alias} length', len_buf, 511, 0)
        print(f"Length ms: {len_buf.value}")
        
        # Play 1 sec
        ctypes.windll.winmm.mciSendStringW(f'play {alias} from 0', None, 0, 0)
        time.sleep(1)
        
        pos_buf = ctypes.create_unicode_buffer(512)
        ctypes.windll.winmm.mciSendStringW(f'status {alias} position', pos_buf, 511, 0)
        print(f"Position ms: {pos_buf.value}")
        
        ctypes.windll.winmm.mciSendStringW(f'stop {alias}', None, 0, 0)
        ctypes.windll.winmm.mciSendStringW(f'close {alias}', None, 0, 0)
