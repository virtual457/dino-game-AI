import subprocess
import time
import pyautogui
import pygetwindow as gw
import os
import numpy as np
import cv2
import mss
import torch
from agent import DQNAgent

# Game constants
CAPTURE_LEFT = 2720
CAPTURE_TOP = 400
CAPTURE_WIDTH = 950
CAPTURE_HEIGHT = 300

# Training constants
TARGET_FPS = 30
FRAME_TIME = 1.0 / TARGET_FPS  # 0.0333 seconds per frame

# Monitor 1 dimensions
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


def preprocess_frame(frame):
    """
    Preprocess game frame for neural network
    
    Args:
        frame: Raw screenshot (H, W, 3) BGR image
    
    Returns:
        Preprocessed frame (1, 84, 84) grayscale normalized
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Resize to 84x84
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    
    # Normalize to [0, 1]
    normalized = resized.astype(np.float32) / 255.0
    
    # Add channel dimension
    processed = np.expand_dims(normalized, axis=0)
    
    return processed


def get_reward(prev_frame, curr_frame, game_over):
    """
    Calculate reward for current frame
    
    Args:
        prev_frame: Previous frame
        curr_frame: Current frame
        game_over: Whether game is over
    
    Returns:
        reward: Float reward value
    """
    if game_over:
        return -10.0  # Large penalty for dying
    else:
        return 0.1  # Small reward for staying alive


def is_game_over(frame):
    """
    Detect if game is over by checking for "GAME OVER" text or restart button
    
    Simple heuristic: Check if there's significant change in specific region
    You may need to adjust this based on visual inspection
    
    Args:
        frame: Current game frame
    
    Returns:
        bool: True if game over detected
    """
    # For now, return False - we'll implement proper detection later
    # You can add logic to detect the "GAME OVER" text or restart button
    return False


def restart_game():
    """Restart the game by pressing space"""
    pyautogui.press('space')
    time.sleep(0.5)


def open_game():
    """Open Chrome with dino game"""
    print("Closing existing Chrome windows...")
    os.system("taskkill /F /IM chrome.exe >nul 2>&1")
    time.sleep(2)
    
    print("Opening Chrome...")
    subprocess.Popen([chrome_path, "--new-window"])
    time.sleep(2.5)
    
    # Handle restore popup
    pyautogui.press('tab')
    time.sleep(0.3)
    pyautogui.press('enter')
    time.sleep(0.5)
    
    # Position window
    chrome_windows = gw.getWindowsWithTitle('Chrome')
    if chrome_windows:
        win = chrome_windows[0]
        win.resizeTo(WINDOW_WIDTH, WINDOW_HEIGHT)
        time.sleep(0.2)
        win.moveTo(WINDOW_X, WINDOW_Y)
        time.sleep(0.5)
        
        # Focus and open dino
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
        
        # Start game
        pyautogui.press('space')
        time.sleep(0.5)
        
        print("✅ Game ready!")
        return True
    return False


def train(episodes=1000, max_steps_per_episode=10000):
    """
    Train the DQN agent
    
    Args:
        episodes: Number of episodes to train
        max_steps_per_episode: Maximum steps per episode
    """
    # Initialize agent
    agent = DQNAgent(
        learning_rate=0.0001,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_capacity=50000,
        batch_size=32
    )
    
    # Open game
    if not open_game():
        print("Failed to open game!")
        return
    
    # Setup screen capture
    capture_region = {
        "top": CAPTURE_TOP,
        "left": CAPTURE_LEFT,
        "width": CAPTURE_WIDTH,
        "height": CAPTURE_HEIGHT,
    }
    
    sct = mss.mss()
    
    print(f"\n{'='*60}")
    print(f"Starting Training at {TARGET_FPS} FPS")
    print(f"{'='*60}\n")
    
    # Training loop
    for episode in range(episodes):
        episode_reward = 0
        episode_steps = 0
        episode_loss = []
        episode_start_time = time.time()
        
        # Get initial state
        screenshot = sct.grab(capture_region)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        state = preprocess_frame(frame)
        
        for step in range(max_steps_per_episode):
            step_start_time = time.time()
            
            # Agent selects action
            action = agent.select_action(state, training=True)
            
            # Execute action
            if action == 1:  # Jump
                pyautogui.press('space')
            
            # Wait to maintain FPS
            elapsed = time.time() - step_start_time
            if elapsed < FRAME_TIME:
                time.sleep(FRAME_TIME - elapsed)
            
            # Get next state
            screenshot = sct.grab(capture_region)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            next_state = preprocess_frame(frame)
            
            # Check if game over (simplified - always False for now)
            done = is_game_over(frame)
            
            # Calculate reward
            reward = get_reward(state, next_state, done)
            episode_reward += reward
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            
            # Learn
            loss = agent.learn()
            if loss is not None:
                episode_loss.append(loss)
            
            # Move to next state
            state = next_state
            episode_steps += 1
            
            # Break if done
            if done:
                restart_game()
                break
        
        # Episode summary
        episode_time = time.time() - episode_start_time
        avg_loss = np.mean(episode_loss) if episode_loss else 0.0
        actual_fps = episode_steps / episode_time
        
        print(f"Episode {episode+1}/{episodes}")
        print(f"  Steps: {episode_steps}")
        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Avg Loss: {avg_loss:.4f}")
        print(f"  Epsilon: {agent.epsilon:.3f}")
        print(f"  FPS: {actual_fps:.1f}")
        print(f"  Time: {episode_time:.1f}s")
        print()
        
        # Save model periodically
        if (episode + 1) % 10 == 0:
            agent.save(f"checkpoints/dino_dqn_episode_{episode+1}.pth")
    
    print("\n✅ Training complete!")
    agent.save("checkpoints/dino_dqn_final.pth")


if __name__ == "__main__":
    # Create checkpoints directory
    os.makedirs("checkpoints", exist_ok=True)
    
    try:
        train(episodes=100, max_steps_per_episode=10000)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()
