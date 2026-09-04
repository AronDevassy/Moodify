import ctypes
import os
import wave
import uuid
import sys

def get_audio_duration(file_path: str) -> float:
    if not file_path or not os.path.exists(file_path):
        return 0.0
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".wav":
        try:
            with wave.open(file_path, "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                if rate > 0:
                    return float(frames) / float(rate)
        except Exception:
            pass

    if sys.platform == "win32":
        try:
            norm = os.path.normpath(os.path.abspath(file_path))
            buf = ctypes.create_unicode_buffer(512)
            ctypes.windll.kernel32.GetShortPathNameW(norm, buf, 512)
            short_path = buf.value if buf.value else norm

            alias = f"dur_{uuid.uuid4().hex[:8]}"
            device_type = "waveaudio" if ext == ".wav" else "mpegvideo"
            err = ctypes.windll.winmm.mciSendStringW(f'open "{short_path}" type {device_type} alias {alias}', None, 0, 0)
            if err != 0 and device_type == "mpegvideo":
                err = ctypes.windll.winmm.mciSendStringW(f'open "{short_path}" alias {alias}', None, 0, 0)

            if err == 0:
                buf_len = ctypes.create_unicode_buffer(512)
                ctypes.windll.winmm.mciSendStringW(f'status {alias} length', buf_len, 511, 0)
                ctypes.windll.winmm.mciSendStringW(f'close {alias}', None, 0, 0)
                val = buf_len.value.strip()
                if val.isdigit():
                    dur = float(val) / 1000.0
                    if dur > 0:
                        return dur
        except Exception as e:
            print(f"[AudioDuration] MCI error: {e}")

    return 0.0

wav_path = os.path.abspath('scratch_test.wav')
with wave.open(wav_path, 'wb') as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(44100)
    wf.writeframes(b'\x00\x00' * 88200) # 2 seconds

dur = get_audio_duration(wav_path)
print("Detected duration for 2-sec WAV:", dur)

if os.path.exists(wav_path):
    os.remove(wav_path)
