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

pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False


class DinoGameEnv:
    def __init__(self):
        self.window = None
        self.sct = mss.mss()
        self.capture_region = {
            "top": CAPTURE_TOP,
            "left": CAPTURE_LEFT,
            "width": CAPTURE_WIDTH,
            "height": CAPTURE_HEIGHT,
        }
        self.frame_stack = deque(maxlen=N_FRAME_STACK)
        self.crash_detection_buffer = []
        self.previous_action = 0
    
    def open_game(self):
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
        
        pyautogui.press('tab')
        time.sleep(0.3)
        pyautogui.press('enter')
        time.sleep(0.5)
        
        chrome_windows = gw.getWindowsWithTitle('Chrome')
        if not chrome_windows:
            return False
        
        self.window = chrome_windows[0]
        self.window.resizeTo(window_width, window_height)
        time.sleep(0.2)
        self.window.moveTo(window_x, window_y)
        time.sleep(0.5)
        
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
        
        pyautogui.press('space')
        time.sleep(0.5)
        
        print("[ENV] Game ready!")
        return True
    
    def capture_frame(self):
        screenshot = self.sct.grab(self.capture_region)
        raw_frame = np.array(screenshot)
        raw_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGRA2BGR)
        return raw_frame
    
    def preprocess_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_AREA)
        _, binary = cv2.threshold(resized, 127, 1, cv2.THRESH_BINARY)
        binary = binary.astype(np.float32)
        preprocessed = np.expand_dims(binary, axis=0)
        return preprocessed, gray, resized
    
    def add_frame_to_stack(self, frame):
        self.frame_stack.append(frame)
        while len(self.frame_stack) < N_FRAME_STACK:
            self.frame_stack.append(frame)
    
    def get_stacked_state(self):
        if len(self.frame_stack) == 0:
            return np.zeros((N_FRAME_STACK, INPUT_HEIGHT, INPUT_WIDTH), dtype=np.float32)
        stacked = np.vstack(list(self.frame_stack))
        return stacked
    
    def is_game_over(self, current_gray_frame):
        if len(self.crash_detection_buffer) < 2:
            return False
        
        prev_frame_1 = self.crash_detection_buffer[-1]
        prev_frame_2 = self.crash_detection_buffer[-2]
        
        if current_gray_frame.shape != prev_frame_1.shape:
            return False
        
        diff_1 = cv2.absdiff(current_gray_frame, prev_frame_1)
        _, thresh_1 = cv2.threshold(diff_1, 2, 255, cv2.THRESH_BINARY)
        similarity_1 = 1.0 - (np.sum(thresh_1 > 0) / thresh_1.size)
        
        diff_2 = cv2.absdiff(prev_frame_1, prev_frame_2)
        _, thresh_2 = cv2.threshold(diff_2, 2, 255, cv2.THRESH_BINARY)
        similarity_2 = 1.0 - (np.sum(thresh_2 > 0) / thresh_2.size)
        
        return similarity_1 > FRAME_SIMILARITY_THRESHOLD and similarity_2 > FRAME_SIMILARITY_THRESHOLD
    
    def update_crash_detection_buffer(self, gray_frame):
        self.crash_detection_buffer.append(gray_frame.copy())
        if len(self.crash_detection_buffer) > 2:
            self.crash_detection_buffer.pop(0)
    
    def reset_buffers(self):
        self.frame_stack.clear()
        self.crash_detection_buffer = []
        self.previous_action = 0
        pyautogui.keyUp('up')
        pyautogui.keyUp('down')
    
    def execute_action(self, action):
        if action == self.previous_action:
            return
        
        if self.previous_action == 1:
            pyautogui.keyUp('up')
        elif self.previous_action == 2:
            pyautogui.keyUp('down')
        
        if action == 1:
            pyautogui.keyDown('up')
        elif action == 2:
            pyautogui.keyDown('down')
        
        self.previous_action = action
    
    def restart_game(self):
        pyautogui.keyUp('up')
        pyautogui.keyUp('down')
        
        print("[ENV] Pressing SPACE to restart...")
        pyautogui.press('space')
        print(f"[ENV] Waiting {RESTART_WAIT_DURATION}s for obstacles...")
        time.sleep(RESTART_WAIT_DURATION)
        print("[ENV] Ready!")
        
        self.previous_action = 0
