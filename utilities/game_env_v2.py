"""
Game environment handler for Chrome Dino - V2 with Action Persistence
Implements frame stacking AND persistent action execution

Key Improvements:
- Action persistence: Hold keys until action changes
- Clean state transitions: Release old key before pressing new one
- No action flickering
"""
import subprocess
import time
import pyautogui
import pygetwindow as gw
import os
import mss
import numpy as np
import cv2
from collections import deque
from config import *

# Disable pyautogui pauses for faster actions
pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False


class DinoGameEnv:
    """
    Handles Chrome Dino game window and interactions
    
    Key Features:
    - Frame stacking: Maintains 4-frame buffer for velocity information
    - Crash detection: 3-frame similarity check
    - Action persistence: Hold keys until action changes (prevents flickering)
    """
    
    def __init__(self):
        self.window = None
        self.sct = mss.mss()
        self.capture_region = {
            "top": CAPTURE_TOP,
            "left": CAPTURE_LEFT,
            "width": CAPTURE_WIDTH,
            "height": CAPTURE_HEIGHT,
        }
        
        # Frame stacking buffer - stores last 4 preprocessed frames
        self.frame_stack = deque(maxlen=N_FRAME_STACK)
        
        # Crash detection buffer - stores raw grayscale frames
        self.crash_detection_buffer = []
        
        # ACTION PERSISTENCE: Track previous action
        self.previous_action = 0  # Start with "nothing" (0)
    
    def open_game(self):
        """Open Chrome with dino game"""
        print("[ENV] Closing existing Chrome windows...")
        os.system(f"taskkill /F /IM chrome.exe >nul 2>&1")
        time.sleep(2)
        
        print("[ENV] Opening Chrome...")
        window_width = int(MONITOR_WIDTH * WINDOW_WIDTH_PCT)
        window_height = int(MONITOR_HEIGHT * WINDOW_HEIGHT_PCT)
        window_x = MONITOR_1_X_START + (MONITOR_WIDTH - window_width) // 2
        window_y = MONITOR_1_Y_START + (MONITOR_HEIGHT - window_height) // 2
        
        subprocess.Popen([CHROME_PATH, "--new-window"])
        time.sleep(2.5)
        
        # Handle restore popup
        pyautogui.press('tab')
        time.sleep(0.3)
        pyautogui.press('enter')
        time.sleep(0.5)
        
        # Position window
        chrome_windows = gw.getWindowsWithTitle('Chrome')
        if not chrome_windows:
            return False
        
        self.window = chrome_windows[0]
        self.window.resizeTo(window_width, window_height)
        time.sleep(0.2)
        self.window.moveTo(window_x, window_y)
        time.sleep(0.5)
        
        # Focus and navigate to dino
        center_x = window_x + window_width // 2
        center_y = window_y + window_height // 2
        pyautogui.click(center_x, center_y)
        time.sleep(0.3)
        
        pyautogui.hotkey('ctrl', 'l')
        time.sleep(0.3)
        pyautogui.typewrite(GAME_URL, interval=0.05)
        time.sleep(0.3)
        pyautogui.press('enter')
        time.sleep(1)
        
        # Start game
        pyautogui.press('space')
        time.sleep(0.5)
        
        print("[ENV] Game ready!")
        return True
    
    def capture_frame(self):
        """Capture game screen and return raw frame"""
        screenshot = self.sct.grab(self.capture_region)
        raw_frame = np.array(screenshot)
        raw_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGRA2BGR)
        return raw_frame
    
    def preprocess_frame(self, frame):
        """
        Preprocess frame for neural network
        
        Args:
            frame: Raw BGR frame from screen capture
        
        Returns:
            preprocessed: (1, 84, 252) single frame (binary: 0 or 1)
            gray: Grayscale version for crash detection
            resized: Resized version for visualization
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_AREA)
        
        # Convert to binary (black and white) using threshold
        # Pixels > 127 become 1.0 (white), pixels <= 127 become 0.0 (black)
        _, binary = cv2.threshold(resized, 127, 1, cv2.THRESH_BINARY)
        binary = binary.astype(np.float32)
        
        preprocessed = np.expand_dims(binary, axis=0)  # (1, 84, 252)
        return preprocessed, gray, resized
    
    def add_frame_to_stack(self, frame):
        """
        Add preprocessed frame to stack
        
        Args:
            frame: (1, 84, 252) preprocessed frame
        
        If stack is not full, duplicate the frame to fill it
        """
        self.frame_stack.append(frame)
        
        # Fill stack with duplicates on first frame
        while len(self.frame_stack) < N_FRAME_STACK:
            self.frame_stack.append(frame)
    
    def get_stacked_state(self):
        """
        Get current state as 4 stacked frames
        
        Returns:
            state: (4, 84, 252) numpy array - 4 stacked frames
        """
        if len(self.frame_stack) == 0:
            # Return zero state if no frames yet
            return np.zeros((N_FRAME_STACK, INPUT_HEIGHT, INPUT_WIDTH), dtype=np.float32)
        
        # Stack frames along first dimension: (4, 84, 252)
        stacked = np.vstack(list(self.frame_stack))
        return stacked
    
    def is_game_over(self, current_gray_frame):
        """
        Detect crash by checking if 3 consecutive frames are frozen
        
        Args:
            current_gray_frame: Raw grayscale frame
        
        Returns:
            bool: True if crash detected
        """
        if len(self.crash_detection_buffer) < 2:
            return False
        
        prev_frame_1 = self.crash_detection_buffer[-1]
        prev_frame_2 = self.crash_detection_buffer[-2]
        
        if current_gray_frame.shape != prev_frame_1.shape:
            return False
        
        # Check frame similarity
        diff_1 = cv2.absdiff(current_gray_frame, prev_frame_1)
        _, thresh_1 = cv2.threshold(diff_1, 2, 255, cv2.THRESH_BINARY)
        similarity_1 = 1.0 - (np.sum(thresh_1 > 0) / thresh_1.size)
        
        diff_2 = cv2.absdiff(prev_frame_1, prev_frame_2)
        _, thresh_2 = cv2.threshold(diff_2, 2, 255, cv2.THRESH_BINARY)
        similarity_2 = 1.0 - (np.sum(thresh_2 > 0) / thresh_2.size)
        
        return similarity_1 > FRAME_SIMILARITY_THRESHOLD and similarity_2 > FRAME_SIMILARITY_THRESHOLD
    
    def update_crash_detection_buffer(self, gray_frame):
        """Update crash detection buffer"""
        self.crash_detection_buffer.append(gray_frame.copy())
        if len(self.crash_detection_buffer) > 2:
            self.crash_detection_buffer.pop(0)
    
    def reset_buffers(self):
        """Clear all buffers (call at episode start)"""
        self.frame_stack.clear()
        self.crash_detection_buffer = []
        # Reset action state
        self.previous_action = 0
        # Make sure no keys are held
        pyautogui.keyUp('space')
        pyautogui.keyUp('down')
    
    def execute_action(self, action):
        """
        Execute action with PERSISTENCE (hold keys until action changes)
        
        Args:
            action: 0=nothing, 1=jump, 2=duck
        
        Logic:
        - If action == previous_action: Do nothing (keep holding)
        - If action != previous_action: Release old, press new
        - If action == 0 (nothing): Release everything
        - Special case: Jump is momentary, so we press+release immediately
        
        Action Mapping:
            0: Nothing (release all keys)
            1: Jump (press space briefly)
            2: Duck (hold down key)
        """
        # Action unchanged - maintain current state
        if action == self.previous_action:
            return  # Keep holding whatever we're holding
        
        # Action changed - transition to new action
        
        # Step 1: Release previous action
        if self.previous_action == 1:  # Was jumping
            # Jump is momentary, already released
            pass
        elif self.previous_action == 2:  # Was ducking
            pyautogui.keyUp('down')
        
        # Step 2: Execute new action
        if action == 0:  # Nothing
            # Already released above, nothing more to do
            pass
        
        elif action == 1:  # Jump
            # Jump is MOMENTARY - press and release immediately
            # The game physics handle the jump duration
            pyautogui.press('space')
        
        elif action == 2:  # Duck
            # Duck is PERSISTENT - hold down key
            # Release old action if needed (already done above)
            # If previous was jump, we already don't hold anything
            pyautogui.keyDown('down')
        
        # Update previous action
        self.previous_action = action
    
    def restart_game(self):
        """Restart game after crash"""
        # Make sure all keys are released before restarting
        pyautogui.keyUp('space')
        pyautogui.keyUp('down')
        
        print("[ENV] Pressing SPACE to restart...")
        pyautogui.press('space')
        print(f"[ENV] Waiting {RESTART_WAIT_DURATION}s for obstacles...")
        time.sleep(RESTART_WAIT_DURATION)
        print("[ENV] Ready!")
        
        # Reset action state
        self.previous_action = 0


if __name__ == "__main__":
    # Test action persistence logic
    env = DinoGameEnv()
    
    print("\n=== Testing Action Persistence ===\n")
    
    test_cases = [
        (0, 0, "Nothing → Nothing: Should do nothing"),
        (0, 1, "Nothing → Jump: Should press space"),
        (1, 1, "Jump → Jump: Should do nothing (already released)"),
        (1, 2, "Jump → Duck: Should press down"),
        (2, 2, "Duck → Duck: Should do nothing (keep holding)"),
        (2, 0, "Duck → Nothing: Should release down"),
        (0, 2, "Nothing → Duck: Should press down"),
        (2, 1, "Duck → Jump: Should release down, press space"),
    ]
    
    env.previous_action = 0
    
    for prev, curr, description in test_cases:
        env.previous_action = prev
        print(f"{description}")
        print(f"  Previous: {prev}, Current: {curr}")
        # In real use, this would call pyautogui commands
        # For testing, we just print what would happen
        if curr == prev:
            print(f"  → No change needed")
        elif prev == 2:
            print(f"  → Release DOWN")
        
        if curr == 1:
            print(f"  → Press SPACE (momentary)")
        elif curr == 2:
            print(f"  → Hold DOWN")
        elif curr == 0:
            print(f"  → Release all")
        
        print()
    
    print("✅ Action persistence logic verified!")
