import subprocess
import time
import pyautogui
import pygetwindow as gw
import os
import numpy as np
import cv2
import mss

# Monitor 1 exact dimensions
MONITOR_1_X_START = 1920
MONITOR_1_Y_START = 0
MONITOR_1_X_END = 3625
MONITOR_1_Y_END = 1015

MONITOR_WIDTH = MONITOR_1_X_END - MONITOR_1_X_START
MONITOR_HEIGHT = MONITOR_1_Y_END - MONITOR_1_Y_START

WINDOW_WIDTH = int(MONITOR_WIDTH * 0.8)
WINDOW_HEIGHT = int(MONITOR_HEIGHT * 0.8)

WINDOW_X = MONITOR_1_X_START + (MONITOR_WIDTH - WINDOW_WIDTH) // 2
WINDOW_Y = MONITOR_1_Y_START + (MONITOR_HEIGHT - WINDOW_HEIGHT) // 2

chrome_path = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"

# Close all existing Chrome windows
print("Closing all existing Chrome windows...")
os.system("taskkill /F /IM chrome.exe >nul 2>&1")
time.sleep(2)

print("Opening Chrome window...")

subprocess.Popen([
    chrome_path,
    "--new-window"
])

time.sleep(2.5)

pyautogui.press('tab')
time.sleep(0.3)
pyautogui.press('enter')
time.sleep(0.5)

chrome_windows = gw.getWindowsWithTitle('Chrome')
if chrome_windows:
    win = chrome_windows[0]
    
    win.resizeTo(WINDOW_WIDTH, WINDOW_HEIGHT)
    time.sleep(0.2)
    win.moveTo(WINDOW_X, WINDOW_Y)
    time.sleep(0.5)
    
    center_x = WINDOW_X + WINDOW_WIDTH // 2
    center_y = WINDOW_Y + WINDOW_HEIGHT // 2
    pyautogui.click(center_x, center_y)
    time.sleep(0.3)
    
    pyautogui.hotkey('ctrl', 'l')
    time.sleep(0.3)
    pyautogui.typewrite('chrome://dino', interval=0.05)
    time.sleep(0.3)
    pyautogui.press('enter')
    time.sleep(1)
    
    pyautogui.press('space')
    time.sleep(0.5)
    
    print()
    print("✅ Dino game started!")
    print()
    
    # Starting position - adjust these
    capture_left = 2300
    capture_top = 200
    capture_width = 600
    capture_height = 150
    
    print("=== INTERACTIVE CAPTURE POSITION FINDER ===")
    print()
    print("Controls:")
    print("  i/k: Move UP/DOWN")
    print("    i : Move UP")
    print("    k : Move DOWN")
    print("  j/l: Move LEFT/RIGHT")
    print("    j : Move LEFT")
    print("    l : Move RIGHT")
    print()
    print("  w/s: Adjust width")
    print("    w : Increase width")
    print("    s : Decrease width")
    print()
    print("  e/d: Adjust height")
    print("    e : Increase height")
    print("    d : Decrease height")
    print()
    print("  SPACE : Make dino JUMP")
    print("  q : QUIT")
    print()
    
    sct = mss.mss()
    step = 10  # pixels to move per keypress
    
    try:
        while True:
            capture_region = {
                "top": capture_top,
                "left": capture_left,
                "width": capture_width,
                "height": capture_height,
            }
            
            # Capture
            screenshot = sct.grab(capture_region)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            
            # Display info on frame
            cv2.putText(frame, f"Top-Left: ({capture_left}, {capture_top})", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Size: {capture_width}x{capture_height}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Use i/k/j/l to adjust", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('Find Capture Region - Use i/k/j/l keys', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):  # Space for jump
                pyautogui.press('space')
                print("JUMP!")
            elif key == ord('i'):  # Up
                capture_top -= step
                print(f"Move UP: top={capture_top}")
            elif key == ord('k'):  # Down
                capture_top += step
                print(f"Move DOWN: top={capture_top}")
            elif key == ord('j'):  # Left
                capture_left -= step
                print(f"Move LEFT: left={capture_left}")
            elif key == ord('l'):  # Right
                capture_left += step
                print(f"Move RIGHT: left={capture_left}")
            elif key == ord('w'):  # Increase width
                capture_width += step
                print(f"Width: {capture_width}")
            elif key == ord('s'):  # Decrease width
                capture_width -= step
                print(f"Width: {capture_width}")
            elif key == ord('e'):  # Increase height
                capture_height += step
                print(f"Height: {capture_height}")
            elif key == ord('d'):  # Decrease height
                capture_height -= step
                print(f"Height: {capture_height}")
    
    except KeyboardInterrupt:
        pass
    
    finally:
        cv2.destroyAllWindows()
        print()
        print("=== FINAL COORDINATES ===")
        print(f"CAPTURE_LEFT = {capture_left}")
        print(f"CAPTURE_TOP = {capture_top}")
        print(f"CAPTURE_WIDTH = {capture_width}")
        print(f"CAPTURE_HEIGHT = {capture_height}")
        
else:
    print("❌ Failed to find Chrome window")
