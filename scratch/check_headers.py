import sys
import os

p1 = r"C:\Users\ASUS\Music\TAMIL NEW\Adaavadi Video Song  LIK  Pradeep Ranganathan  Anirudh Ravichander  Krithi Shetty.mp3"
p2 = r"C:\Users\ASUS\Music\ENGLISH\02 Maroon 5 - Payphone.mp3"

for p in [p1, p2]:
    print(f"\nFile: {os.path.basename(p)}")
    if os.path.exists(p):
        with open(p, "rb") as f:
            header = f.read(32)
            print(f"Header hex: {header.hex()}")
            print(f"Header text: {header}")
