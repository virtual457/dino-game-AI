import pyautogui
import mss
import numpy as np
import cv2
import time

print("=== Display Scaling Detection ===")
print()
print("Move your mouse to the TOP-LEFT corner of the dino game area")
print("(where you want capture to start)")
print()
input("Press Enter when mouse is at TOP-LEFT corner...")

# Get mouse position
top_left_x, top_left_y = pyautogui.position()
print(f"Mouse at TOP-LEFT: ({top_left_x}, {top_left_y})")
print()

print("Now move your mouse to the BOTTOM-RIGHT corner of the dino game area")
print("(where you want capture to end)")
print()
input("Press Enter when mouse is at BOTTOM-RIGHT corner...")

# Get mouse position
bottom_right_x, bottom_right_y = pyautogui.position()
print(f"Mouse at BOTTOM-RIGHT: ({bottom_right_x}, {bottom_right_y})")
print()

# Calculate dimensions
width = bottom_right_x - top_left_x
height = bottom_right_y - top_left_y

print(f"Calculated capture region (from mouse positions):")
print(f"  Top-left: ({top_left_x}, {top_left_y})")
print(f"  Bottom-right: ({bottom_right_x}, {bottom_right_y})")
print(f"  Width: {width}, Height: {height}")
print()

# Now test actual capture with these coordinates
capture_region = {
    "top": top_left_y,
    "left": top_left_x,
    "width": width,
    "height": height
}

print("Testing capture with these coordinates...")
print("A window will open showing what's being captured.")
print("Press 'q' to close.")
print()

sct = mss.mss()

try:
    while True:
        screenshot = sct.grab(capture_region)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        
        # Add coordinates to frame
        cv2.putText(frame, f"Top-left: ({top_left_x}, {top_left_y})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Size: {width}x{height}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Capture Test - Press q to close', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    pass
finally:
    cv2.destroyAllWindows()
    
print()
print("Use these coordinates in validate_pipeline.py:")
print(f"CAPTURE_LEFT = {top_left_x}")
print(f"CAPTURE_TOP = {top_left_y}")
print(f"CAPTURE_RIGHT = {bottom_right_x}")
print(f"CAPTURE_BOTTOM = {bottom_right_y}")
