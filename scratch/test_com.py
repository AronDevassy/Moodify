import sys

try:
    import win32com.client
    print("win32com is available!")
    wmp = win32com.client.Dispatch("WMPlayer.OCX")
    print("WMPlayer.OCX COM object created successfully!")
    print(f"Version: {wmp.versionInfo}")
except Exception as e:
    print(f"win32com test failed: {e}")
