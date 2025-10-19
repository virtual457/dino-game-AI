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
from collections import deque

# Set matplotlib to use non-GUI backend to avoid threading issues
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Disable pyautogui pauses for faster actions
pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False

# Game constants
CAPTURE_LEFT = 2720
CAPTURE_TOP = 400
CAPTURE_WIDTH = 950
CAPTURE_HEIGHT = 300

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
    """Preprocess game frame for neural network"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    processed = np.expand_dims(normalized, axis=0)
    return processed, gray, resized


def is_game_over(current_frame, frame_history):
    """Detect crash by checking if 3 consecutive frames are frozen"""
    if len(frame_history) < 2:
        return False
    
    prev_frame_1 = frame_history[-1]
    prev_frame_2 = frame_history[-2]
    
    if current_frame.shape != prev_frame_1.shape or current_frame.shape != prev_frame_2.shape:
        return False
    
    # Check frame similarity
    diff_1 = cv2.absdiff(current_frame, prev_frame_1)
    _, thresh_1 = cv2.threshold(diff_1, 2, 255, cv2.THRESH_BINARY)
    similarity_1 = 1.0 - (np.sum(thresh_1 > 0) / thresh_1.size)
    
    diff_2 = cv2.absdiff(prev_frame_1, prev_frame_2)
    _, thresh_2 = cv2.threshold(diff_2, 2, 255, cv2.THRESH_BINARY)
    similarity_2 = 1.0 - (np.sum(thresh_2 > 0) / thresh_2.size)
    
    # If all 3 frames are >99.9% identical, game is frozen
    return similarity_1 > 0.999 and similarity_2 > 0.999


def calculate_reward(action, game_over, action_history, jump_count, duck_count, total_actions):
    """Calculate reward with dynamic penalty for movements"""
    if game_over:
        return -10.0, 0.0
    
    reward = 0.5  # Base reward for staying alive
    
    return reward, 0.0


def open_game():
    """Open Chrome with dino game"""
    print("Closing existing Chrome windows...")
    os.system("taskkill /F /IM chrome.exe >nul 2>&1")
    time.sleep(2)
    
    print("Opening Chrome...")
    subprocess.Popen([chrome_path, "--new-window"])
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
        
        print("✅ Game ready!")
        return True
    return False


def train():
    """Main training loop"""
    print("\n" + "="*70)
    print("DQN Training - 3 Actions: Nothing, Jump, Duck")
    print("="*70)
    
    # Create checkpoints directory
    os.makedirs("checkpoints", exist_ok=True)
    
    # Initialize agent with 3 actions
    agent = DQNAgent(
        n_actions=3,
        learning_rate=0.0001,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        batch_size=32,
        buffer_capacity=15000  # ~30 minutes of gameplay at 10 FPS
    )
    
    # Try to load existing model
    checkpoint_path = "checkpoints/debug_model.pth"
    if os.path.exists(checkpoint_path):
        print(f"\n💾 Found existing model: {checkpoint_path}")
        agent.load(checkpoint_path)
        print("✅ Resuming training from checkpoint!")
    else:
        print("\n🆕 No existing model found. Starting fresh!")
    
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
    
    print("\nControls:")
    print("  Press 'q' to quit (saves model)")
    print("  Press 's' to save model manually")
    print("\nAuto-saves every 500 steps")
    print("Running at 10 FPS")
    print("\nStarting in 2 seconds...")
    time.sleep(2)
    
    # FPS control
    TARGET_FPS = 10
    FRAME_TIME = 1.0 / TARGET_FPS
    
    # Episode tracking for plotting
    episode_number = 0
    episode_rewards = deque(maxlen=1000)  # Store last 1000 episode rewards
    episode_rewards_history = []  # Store ALL episode rewards
    episode_avg_rewards = []  # Store rolling average for each episode
    
    # Episode tracking
    step_count = 0
    total_reward = 0
    total_loss = []
    prev_state = None
    last_save_step = 0
    save_interval = 500
    frame_history = []
    episode_start_time = time.time()
    
    # Action tracking
    jump_count = 0
    duck_count = 0
    total_actions = 0
    action_history = []  # Rolling 500-action window
    
    # Frame visualization tracking
    last_jump_frames = []
    last_duck_frames = []
    last_nothing_frames = []
    
    try:
        while True:
            step_start = time.time()
            
            # Capture and preprocess
            screenshot = sct.grab(capture_region)
            raw_frame = np.array(screenshot)
            raw_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGRA2BGR)
            state, gray_frame, resized_frame = preprocess_frame(raw_frame)
            
            # Get network prediction
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                q_values = agent.policy_net(state_tensor)
                q_values_np = q_values.cpu().numpy()[0]
                del state_tensor, q_values
                if agent.device.type == 'cuda':
                    torch.cuda.empty_cache()
            
            # Select action
            action = agent.select_action(state, training=True)
            action_names = ["NOTHING", "JUMP", "DUCK"]
            action_name = action_names[action]
            
            # Check if game over
            game_over = is_game_over(gray_frame, frame_history)
            if game_over and len(frame_history) >= 2:
                print(f"\n🔴 CRASH DETECTED at frame {step_count}!")
            
            # Update frame history
            frame_history.append(gray_frame.copy())
            if len(frame_history) > 2:
                frame_history.pop(0)
            
            # Calculate reward using reward function
            reward, current_penalty = calculate_reward(
                action, game_over, action_history,
                jump_count, duck_count, total_actions
            )
            total_reward += reward
            
            # Track actions
            total_actions += 1
            action_history.append(action)
            if len(action_history) > 500:
                action_history.pop(0)
            
            # Execute action and store frames
            if action == 1:  # Jump
                pyautogui.press('space')
                jump_count += 1
                last_jump_frames.append(resized_frame.copy())
                if len(last_jump_frames) > 3:
                    last_jump_frames.pop(0)
            elif action == 2:  # Duck
                pyautogui.keyDown('down')
                time.sleep(0.2)  # Hold for 70ms
                pyautogui.keyUp('down')
                duck_count += 1
                last_duck_frames.append(resized_frame.copy())
                if len(last_duck_frames) > 3:
                    last_duck_frames.pop(0)
            else:  # Nothing
                last_nothing_frames.append(resized_frame.copy())
                if len(last_nothing_frames) > 3:
                    last_nothing_frames.pop(0)
            
            # Training
            # Only store experiences while game is running
            # Skip frozen frames after crash
            if prev_state is not None:
                # Store the transition
                agent.store_transition(prev_state, action, reward, state, game_over)
                
                # Train the network
                loss = agent.learn()
                if loss is not None:
                    total_loss.append(loss)
            
            # Update prev_state only if game is still running
            # This prevents using frozen frames as "prev_state" for next iteration
            if not game_over:
                prev_state = state.copy()
            else:
                prev_state = None  # Reset so we don't use frozen frame
            
            # === VISUALIZATION ===
            vis_height = 700
            vis_width = 800
            visualization = np.zeros((vis_height, vis_width, 3), dtype=np.uint8)
            
            # Headers
            cv2.putText(visualization, "LAST 3 JUMP FRAMES", (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(visualization, "LAST 3 DUCK FRAMES", (10, 255), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(visualization, "LAST 3 NOTHING FRAMES", (10, 485), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)
            
            frame_size = 150
            
            # Jump frames (top)
            for i, frame in enumerate(last_jump_frames[-3:]):
                x_pos = 10 + (i * 160)
                frame_display = cv2.resize(frame, (frame_size, frame_size))
                frame_3ch = cv2.cvtColor(frame_display, cv2.COLOR_GRAY2BGR)
                cv2.rectangle(frame_3ch, (0, 0), (frame_size-1, frame_size-1), (0, 255, 0), 3)
                visualization[40:190, x_pos:x_pos+frame_size] = frame_3ch
            
            # Duck frames (middle)
            for i, frame in enumerate(last_duck_frames[-3:]):
                x_pos = 10 + (i * 160)
                frame_display = cv2.resize(frame, (frame_size, frame_size))
                frame_3ch = cv2.cvtColor(frame_display, cv2.COLOR_GRAY2BGR)
                cv2.rectangle(frame_3ch, (0, 0), (frame_size-1, frame_size-1), (255, 255, 0), 3)
                visualization[270:420, x_pos:x_pos+frame_size] = frame_3ch
            
            # Nothing frames (bottom)
            for i, frame in enumerate(last_nothing_frames[-3:]):
                x_pos = 10 + (i * 160)
                frame_display = cv2.resize(frame, (frame_size, frame_size))
                frame_3ch = cv2.cvtColor(frame_display, cv2.COLOR_GRAY2BGR)
                cv2.rectangle(frame_3ch, (0, 0), (frame_size-1, frame_size-1), (255, 100, 0), 3)
                visualization[500:650, x_pos:x_pos+frame_size] = frame_3ch
            
            # Stats (right side)
            stats_x = 520
            cv2.putText(visualization, f"Step: {step_count}", (stats_x, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(visualization, f"Reward: {total_reward:.1f}", (stats_x, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(visualization, f"Epsilon: {agent.epsilon:.3f}", (stats_x, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Episode actions
            jump_pct = (jump_count / total_actions * 100) if total_actions > 0 else 0
            duck_pct = (duck_count / total_actions * 100) if total_actions > 0 else 0
            nothing_pct = 100 - jump_pct - duck_pct
            
            cv2.putText(visualization, "Episode:", (stats_x, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(visualization, f"Jump: {jump_count} ({jump_pct:.1f}%)", (stats_x, 180), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(visualization, f"Duck: {duck_count} ({duck_pct:.1f}%)", (stats_x, 205), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(visualization, f"Nothing: ({nothing_pct:.1f}%)", (stats_x, 230), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 1)
            
            # Rolling 500 stats
            if len(action_history) >= 100:
                movements_500 = sum(1 for a in action_history if a in [1, 2])
                movement_pct_500 = (movements_500 / len(action_history)) * 100
            else:
                movement_pct_500 = jump_pct + duck_pct
            
            cv2.putText(visualization, "Last 500:", (stats_x, 280), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
            cv2.putText(visualization, f"Moves: {movement_pct_500:.1f}%", (stats_x, 310), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(visualization, f"Size: {len(action_history)}/500", (stats_x, 335), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            
            # Current penalty
            penalty_color = (0, 255, 255) if current_penalty == 0 else (0, 255, 0) if current_penalty < 1 else (0, 165, 255) if current_penalty < 3 else (0, 0, 255)
            cv2.putText(visualization, f"Penalty: -{current_penalty:.2f}", (stats_x, 370), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, penalty_color, 2)
            
            # Training status
            buffer_size = len(agent.replay_buffer)
            if buffer_size < 32:
                cv2.putText(visualization, f"Collecting: {buffer_size}/32", (stats_x, 420), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
            else:
                avg_loss = np.mean(total_loss[-100:]) if total_loss else 0.0
                cv2.putText(visualization, "Training!", (stats_x, 420), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(visualization, f"Loss: {avg_loss:.4f}", (stats_x, 450), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Game state
            if game_over:
                cv2.putText(visualization, "GAME OVER", (stats_x, 520), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                cv2.putText(visualization, "Running", (stats_x, 520), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            cv2.imshow('DQN Training - 3 Actions', visualization)
            
            # FPS control
            total_elapsed = time.time() - step_start
            if total_elapsed < FRAME_TIME:
                time.sleep(FRAME_TIME - total_elapsed)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n\n💾 Saving model before exit...")
                agent.save(checkpoint_path)
                break
            elif key == ord('s'):
                print("\n💾 Manual save...")
                agent.save(checkpoint_path)
                print("✅ Model saved!")
            
            step_count += 1
            
            # Auto-save every 500 steps
            if step_count - last_save_step >= save_interval:
                print(f"\n💾 Auto-saving at step {step_count}...")
                agent.save(checkpoint_path)
                last_save_step = step_count
                print("✅ Model saved!\n")
            
            # Restart if game over
            if game_over:
                print("\n⏸️  Pausing 3 seconds after crash...")
                time.sleep(3)
                
                # Increment episode
                episode_number += 1
                
                # Store episode reward
                episode_rewards.append(total_reward)
                episode_rewards_history.append(total_reward)
                
                # Calculate rolling average
                avg_reward_last_1000 = np.mean(episode_rewards)
                episode_avg_rewards.append(avg_reward_last_1000)
                
                # Episode summary
                episode_duration = time.time() - episode_start_time
                avg_loss = np.mean(total_loss) if total_loss else 0.0
                nothing_count = total_actions - jump_count - duck_count
                
                # Rolling stats
                if len(action_history) >= 100:
                    movements_500 = sum(1 for a in action_history if a in [1, 2])
                    movement_pct_500 = (movements_500 / len(action_history)) * 100
                else:
                    movement_pct_500 = ((jump_count + duck_count) / total_actions * 100) if total_actions > 0 else 0.0
                
                print(f"\n{'='*70}")
                print(f"💥 EPISODE {episode_number} ENDED")
                print(f"{'='*70}")
                print(f"  Steps survived:    {step_count}")
                print(f"  Episode reward:    {total_reward:.2f}")
                print(f"  Avg last 1000:     {avg_reward_last_1000:.2f} (from {len(episode_rewards)} episodes)")
                print(f"  Average loss:      {avg_loss:.4f}")
                print(f"  Epsilon:           {agent.epsilon:.3f}")
                print(f"  Buffer size:       {len(agent.replay_buffer):,}/15,000")
                print(f"  Duration:          {episode_duration:.1f}s")
                print(f"  ---")
                print(f"  Episode actions:")
                print(f"    Total:    {total_actions}")
                print(f"    Jumps:    {jump_count} ({jump_pct:.1f}%)")
                print(f"    Ducks:    {duck_count} ({duck_pct:.1f}%)")
                print(f"    Nothing:  {nothing_count} ({nothing_pct:.1f}%)")
                print(f"  ---")
                print(f"  Rolling 500-action stats:")
                print(f"    History:  {len(action_history)}/500")
                print(f"    Moves:    {movement_pct_500:.1f}%")
                if agent.device.type == 'cuda':
                    gpu_memory = torch.cuda.memory_allocated() / 1024**2
                    print(f"  ---")
                    print(f"  GPU:      {gpu_memory:.1f}MB")
                print(f"{'='*70}\n")
                
                # Plot progress every 10 episodes
                if episode_number % 10 == 0 and len(episode_rewards_history) > 1:
                    print("📊 Generating progress plot...")
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    # Plot individual episode rewards
                    ax.plot(episode_rewards_history, alpha=0.3, color='blue', label='Episode Reward')
                    
                    # Plot rolling average
                    ax.plot(episode_avg_rewards, color='red', linewidth=2, label='Avg Last 1000 Episodes')
                    
                    ax.set_xlabel('Episode')
                    ax.set_ylabel('Total Reward')
                    ax.set_title(f'Training Progress - Episode {episode_number}')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    # Save to same file (overwrites)
                    plot_path = "checkpoints/training_progress.png"
                    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    print(f"✅ Plot updated: {plot_path}\n")
                
                # Restart
                print("➡️  Pressing SPACE to restart...")
                pyautogui.press('space')
                print("⏳ Waiting 4 seconds for obstacles...")
                time.sleep(4)
                print("✅ Ready!\n")
                
                # Reset episode (keep action_history!)
                step_count = 0
                total_reward = 0
                total_loss = []
                prev_state = None
                frame_history = []
                episode_start_time = time.time()
                jump_count = 0
                duck_count = 0
                total_actions = 0
                last_jump_frames = []
                last_duck_frames = []
                last_nothing_frames = []
    
    except KeyboardInterrupt:
        print("\n\nStopped by user")
        print("\n💾 Saving model...")
        agent.save(checkpoint_path)
    finally:
        cv2.destroyAllWindows()
        
        if agent.device.type == 'cuda':
            print("\n🧹 Cleaning GPU...")
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        print("\n✅ Complete!")
        print(f"   Buffer: {len(agent.replay_buffer)}")
        print(f"   Saved: {checkpoint_path}")


if __name__ == "__main__":
    train()
