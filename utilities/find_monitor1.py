import pyautogui
import time

print("=== Monitor Position Finder ===")
print()
print("Move your mouse to Monitor 1 (the blue one on the right)")
print("I'll show you the coordinates in real-time...")
print()
print("Press Ctrl+C when done")
print()

try:
    while True:
        x, y = pyautogui.position()
        print(f"Mouse position: X={x:5d}, Y={y:5d}", end='\r')
        time.sleep(0.1)
except KeyboardInterrupt:
    print()
    x, y = pyautogui.position()
    print()
    print(f"Final position: X={x}, Y={y}")
    print()
    print("Now move mouse to:")
    print("1. Top-left corner of Monitor 1")
    print("2. Bottom-right corner of Monitor 1")
    print()
    print("Tell me both coordinates!")
