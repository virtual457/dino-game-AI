import mss
import numpy as np
import cv2
import time

# Window coordinates (from open_single_dino.py)
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

# Define capture region (the game area inside Chrome window)
# Skip Chrome's top bar (~100px) and focus on game area
GAME_AREA_TOP_OFFSET = 100
GAME_AREA_HEIGHT = 200  # Dino game is only ~150px tall

capture_region = {
    "top": WINDOW_Y + GAME_AREA_TOP_OFFSET,
    "left": WINDOW_X,
    "width": WINDOW_WIDTH,
    "height": GAME_AREA_HEIGHT,
}

print("Starting screen capture...")
print(f"Capture region: {capture_region}")
print("Press 'q' to quit")
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
        
        # Show the captured frame
        cv2.imshow('Dino Game Capture', frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nStopped by user")

finally:
    cv2.destroyAllWindows()
    print(f"\nAverage FPS: {fps:.1f}")
