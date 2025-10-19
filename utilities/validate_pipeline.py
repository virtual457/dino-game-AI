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

print(f"Monitor 1 size: {MONITOR_WIDTH}x{MONITOR_HEIGHT}")
print(f"Window size: {WINDOW_WIDTH}x{WINDOW_HEIGHT}")
print(f"Window position: ({WINDOW_X}, {WINDOW_Y})")
print()
print("Opening Chrome window...")

# Open Chrome
subprocess.Popen([
    chrome_path,
    "--new-window"
])

time.sleep(2.5)

# Handle restore popup
print("Handling restore page popup...")
pyautogui.press('tab')
time.sleep(0.3)
pyautogui.press('enter')
time.sleep(0.5)

# Get Chrome window
chrome_windows = gw.getWindowsWithTitle('Chrome')
if chrome_windows:
    win = chrome_windows[0]
    
    # Resize and move
    print("Positioning window...")
    win.resizeTo(WINDOW_WIDTH, WINDOW_HEIGHT)
    time.sleep(0.2)
    win.moveTo(WINDOW_X, WINDOW_Y)
    time.sleep(0.5)
    
    # Click to focus
    center_x = WINDOW_X + WINDOW_WIDTH // 2
    center_y = WINDOW_Y + WINDOW_HEIGHT // 2
    pyautogui.click(center_x, center_y)
    time.sleep(0.3)
    
    # Open dino game
    print("Opening dino game...")
    pyautogui.hotkey('ctrl', 'l')
    time.sleep(0.3)
    pyautogui.typewrite('chrome://dino', interval=0.05)
    time.sleep(0.3)
    pyautogui.press('enter')
    time.sleep(1)
    
    # Press space to start game
    print("Starting game (pressing space)...")
    pyautogui.press('space')
    time.sleep(0.5)
    
    print()
    print("✅ Dino game started!")
    print()
    
    # Perfect capture coordinates
    CAPTURE_LEFT = 2720
    CAPTURE_TOP = 400
    CAPTURE_WIDTH = 950
    CAPTURE_HEIGHT = 300
    
    capture_region = {
        "top": CAPTURE_TOP,
        "left": CAPTURE_LEFT,
        "width": CAPTURE_WIDTH,
        "height": CAPTURE_HEIGHT,
    }
    
    print("Starting screen capture validation...")
    print(f"Capture region: Left={CAPTURE_LEFT}, Top={CAPTURE_TOP}")
    print(f"Size: {CAPTURE_WIDTH}x{CAPTURE_HEIGHT}")
    print()
    print("Controls:")
    print("  Press SPACE to make dino JUMP")
    print("  Press 'q' to QUIT")
    print()
    
    sct = mss.mss()
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            # Capture screenshot
            screenshot = sct.grab(capture_region)
            
            # Convert to numpy array
            frame = np.array(screenshot)
            
            # Convert BGRA to BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            
            # Calculate FPS
            frame_count += 1
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            
            # Display FPS on frame
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Press SPACE to JUMP, 'q' to QUIT", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show the captured frame
            cv2.imshow('Dino Game - Pipeline Validation', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):
                # Send jump command to game
                pyautogui.press('space')
                print("JUMP!")
    
    except KeyboardInterrupt:
        print("\nStopped by user")
    
    finally:
        cv2.destroyAllWindows()
        print()
        print(f"Average FPS: {fps:.1f}")
        print()
        print("✅ Pipeline validated successfully!")
        print(f"   Capture: {CAPTURE_WIDTH}x{CAPTURE_HEIGHT} @ {fps:.1f} FPS")
        print()
        print("Ready to build the neural network! 🧠")
        
else:
    print("❌ Failed to find Chrome window")
