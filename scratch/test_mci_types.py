import sys
import os
import ctypes

path = r"C:\Users\ASUS\Music\ENGLISH\02 Maroon 5 - Payphone.mp3"
alias = "test_payphone"

print(f"Testing open methods for: {path}")
print(f"Exists: {os.path.exists(path)}")

cmds = [
    f'open "{path}" type mpegvideo alias {alias}',
    f'open "{path}" alias {alias}',
    f'open "{path}" type mpegvideo2 alias {alias}',
    f'open "{path}" type waveaudio alias {alias}',
]

for cmd in cmds:
    ctypes.windll.winmm.mciSendStringW(f'close {alias}', None, 0, 0)
    err = ctypes.windll.winmm.mciSendStringW(cmd, None, 0, 0)
    print(f"CMD: '{cmd}' -> Err: {err}")
    if err != 0:
        buf = ctypes.create_unicode_buffer(512)
        ctypes.windll.winmm.mciGetErrorStringW(err, buf, 511)
        print(f"  Error message: {buf.value}")
    else:
        print("  SUCCESS!")
        ctypes.windll.winmm.mciSendStringW(f'close {alias}', None, 0, 0)
