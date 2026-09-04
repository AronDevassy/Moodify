import sys
import os
import time
import subprocess

p1 = r"C:\Users\ASUS\Music\TAMIL NEW\Adaavadi Video Song  LIK  Pradeep Ranganathan  Anirudh Ravichander  Krithi Shetty.mp3"
p2 = r"C:\Users\ASUS\Music\ENGLISH\02 Maroon 5 - Payphone.mp3"

print("Testing PowerShell System.Windows.Media.MediaPlayer...")

for p in [p1, p2]:
    print(f"\nTesting file: {os.path.basename(p)}")
    cmd = f'Add-Type -AssemblyName presentationCore; $p = New-Object System.Windows.Media.MediaPlayer; $p.Open("{p}"); $p.Play(); Start-Sleep -s 3'
    res = subprocess.run(["powershell", "-Command", cmd], capture_output=True, text=True)
    print(f"Exit code: {res.returncode}")
    print(f"Stdout: {res.stdout}")
    print(f"Stderr: {res.stderr}")
