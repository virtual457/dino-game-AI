"""
Debug Dashboard - Comprehensive DQN Analysis Tool

Shows in real-time:
1. What images are being fed to the network (4 stacked frames)
2. Q-values for each action
3. Action selection process
4. Reward feedback
5. Performance metrics
6. Frame timing analysis
"""

import os
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import torch
from collections import deque

from config import *
from game_env import DinoGameEnv
from agent import DDQNAgent


class DebugDashboard:
    """Real-time debugging dashboard for DQN agent"""
    
    def __init__(self):
        self.env = DinoGameEnv()
        self.agent = None
        
        # Performance tracking
        self.frame_times = deque(maxlen=100)
        self.q_value_history = deque(maxlen=100)
        self.action_history = deque(maxlen=1000)
        self.reward_history = deque(maxlen=100)
        
        # Setup visualization
        plt.ion()
        self.fig = plt.figure(figsize=(20, 12))
        self.setup_plots()
        
        # Frame counter
        self.frame_count = 0
        
        # Episode stats
        self.episode_frames = 0
        self.episode_reward = 0
        self.episode_start = time.time()
    
    def setup_plots(self):
        """Setup matplotlib subplots"""
        gs = GridSpec(4, 6, figure=self.fig, hspace=0.3, wspace=0.4)
        
        # Row 1: Frame stack visualization (4 frames)
        self.ax_frames = [
            self.fig.add_subplot(gs[0, 0]),
            self.fig.add_subplot(gs[0, 1]),
            self.fig.add_subplot(gs[0, 2]),
            self.fig.add_subplot(gs[0, 3])
        ]
        for i, ax in enumerate(self.ax_frames):
            ax.set_title(f'Frame t-{3-i}', fontsize=10)
            ax.axis('off')
        
        # Row 1: Current raw frame
        self.ax_current = self.fig.add_subplot(gs[0, 4:])
        self.ax_current.set_title('Current Raw Frame', fontsize=12, fontweight='bold')
        self.ax_current.axis('off')
        
        # Row 2: Q-values and action selection
        self.ax_qvalues = self.fig.add_subplot(gs[1, :3])
        self.ax_qvalues.set_title('Q-Values by Action', fontsize=12, fontweight='bold')
        self.ax_qvalues.set_xlabel('Action')
        self.ax_qvalues.set_ylabel('Q-Value')
        
        self.ax_action_dist = self.fig.add_subplot(gs[1, 3:])
        self.ax_action_dist.set_title('Action Distribution (Last 1000)', fontsize=12, fontweight='bold')
        self.ax_action_dist.set_xlabel('Action')
        self.ax_action_dist.set_ylabel('Count')
        
        # Row 3: Performance metrics
        self.ax_qhist = self.fig.add_subplot(gs[2, :3])
        self.ax_qhist.set_title('Q-Value History', fontsize=12)
        self.ax_qhist.set_xlabel('Frame')
        self.ax_qhist.set_ylabel('Max Q-Value')
        
        self.ax_rewards = self.fig.add_subplot(gs[2, 3:])
        self.ax_rewards.set_title('Reward History', fontsize=12)
        self.ax_rewards.set_xlabel('Frame')
        self.ax_rewards.set_ylabel('Reward')
        
        # Row 4: Timing and stats
        self.ax_timing = self.fig.add_subplot(gs[3, :3])
        self.ax_timing.set_title('Frame Processing Time', fontsize=12)
        self.ax_timing.set_xlabel('Frame')
        self.ax_timing.set_ylabel('Time (ms)')
        
        self.ax_stats = self.fig.add_subplot(gs[3, 3:])
        self.ax_stats.axis('off')
        self.ax_stats.set_title('Episode Statistics', fontsize=12, fontweight='bold')
    
    def initialize_agent(self):
        """Load the trained agent"""
        self.agent = DDQNAgent(
            state_shape=STATE_SHAPE,
            n_actions=N_ACTIONS,
            learning_rate=LEARNING_RATE,
            gamma=GAMMA,
            epsilon_start=EPSILON_START,
            epsilon_end=EPSILON_END,
            epsilon_decay_rate=EPSILON_DECAY_RATE,
            batch_size=BATCH_SIZE,
            buffer_capacity=BUFFER_CAPACITY,
            target_update_freq=TARGET_UPDATE_FREQ
        )
        
        # Load checkpoint
        if os.path.exists(CHECKPOINT_PATH):
            print(f"\n✅ Loading model: {CHECKPOINT_PATH}")
            self.agent.load(CHECKPOINT_PATH)
            print(f"   Epsilon: {self.agent.epsilon:.4f}")
            print(f"   Total steps: {self.agent.total_steps:,}\n")
        else:
            print(f"\n⚠️  No checkpoint found at {CHECKPOINT_PATH}")
            print("   Using untrained agent\n")
    
    def get_q_values_and_action(self, state):
        """Get Q-values and selected action with detailed breakdown"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.agent.device)
            q_values = self.agent.policy_net(state_tensor)
            q_values_np = q_values.cpu().numpy()[0]
            
            # Determine action (epsilon-greedy)
            is_random = False
            if np.random.random() < self.agent.epsilon:
                action = np.random.randint(0, N_ACTIONS)
                is_random = True
            else:
                action = q_values_np.argmax()
            
            return q_values_np, action, is_random
    
    def update_visualization(self, state, q_values, action, is_random, reward, raw_frame, timing_ms):
        """Update all dashboard plots"""
        # Track history
        self.q_value_history.append(q_values.max())
        self.action_history.append(action)
        self.reward_history.append(reward)
        self.frame_times.append(timing_ms)
        self.episode_frames += 1
        self.episode_reward += reward
        
        # Clear all axes
        for ax in self.ax_frames:
            ax.clear()
            ax.axis('off')
        self.ax_current.clear()
        self.ax_current.axis('off')
        self.ax_qvalues.clear()
        self.ax_action_dist.clear()
        self.ax_qhist.clear()
        self.ax_rewards.clear()
        self.ax_timing.clear()
        self.ax_stats.clear()
        self.ax_stats.axis('off')
        
        # 1. Show frame stack (4 frames)
        for i, ax in enumerate(self.ax_frames):
            frame_img = (state[i] * 255).astype(np.uint8)
            ax.imshow(frame_img, cmap='gray')
            ax.set_title(f'Frame t-{3-i}', fontsize=10)
            ax.axis('off')
        
        # 2. Show current raw frame
        self.ax_current.imshow(cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB))
        self.ax_current.set_title('Current Raw Frame', fontsize=12, fontweight='bold')
        self.ax_current.axis('off')
        
        # 3. Q-values bar chart
        action_names = ['NOTHING', 'JUMP', 'DUCK']
        colors = ['gray' if i != action else ('green' if not is_random else 'orange') for i in range(N_ACTIONS)]
        bars = self.ax_qvalues.bar(action_names, q_values, color=colors, alpha=0.7)
        self.ax_qvalues.set_title('Q-Values by Action', fontsize=12, fontweight='bold')
        self.ax_qvalues.set_ylabel('Q-Value')
        self.ax_qvalues.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        self.ax_qvalues.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars, q_values):
            height = bar.get_height()
            self.ax_qvalues.text(bar.get_x() + bar.get_width()/2., height,
                               f'{val:.3f}',
                               ha='center', va='bottom' if height > 0 else 'top',
                               fontsize=9, fontweight='bold')
        
        # Highlight selected action
        action_text = f"Action: {action_names[action]}"
        if is_random:
            action_text += " (RANDOM)"
            self.ax_qvalues.text(0.5, 0.95, action_text, 
                               transform=self.ax_qvalues.transAxes,
                               ha='center', va='top', fontsize=11, 
                               bbox=dict(boxstyle='round', facecolor='orange', alpha=0.8),
                               fontweight='bold')
        else:
            action_text += " (GREEDY)"
            self.ax_qvalues.text(0.5, 0.95, action_text,
                               transform=self.ax_qvalues.transAxes,
                               ha='center', va='top', fontsize=11,
                               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
                               fontweight='bold')
        
        # 4. Action distribution
        if len(self.action_history) > 0:
            action_counts = [self.action_history.count(i) for i in range(N_ACTIONS)]
            self.ax_action_dist.bar(action_names, action_counts, color=['gray', 'blue', 'red'], alpha=0.7)
            self.ax_action_dist.set_title('Action Distribution (Last 1000)', fontsize=12, fontweight='bold')
            self.ax_action_dist.set_ylabel('Count')
            self.ax_action_dist.grid(True, alpha=0.3)
            
            # Add percentages
            total = sum(action_counts)
            for i, (name, count) in enumerate(zip(action_names, action_counts)):
                pct = (count / total * 100) if total > 0 else 0
                self.ax_action_dist.text(i, count, f'{pct:.1f}%',
                                        ha='center', va='bottom', fontsize=9)
        
        # 5. Q-value history
        if len(self.q_value_history) > 0:
            self.ax_qhist.plot(list(self.q_value_history), color='blue', linewidth=2)
            self.ax_qhist.set_title('Q-Value History (Max)', fontsize=12)
            self.ax_qhist.set_xlabel('Recent Frames')
            self.ax_qhist.set_ylabel('Max Q-Value')
            self.ax_qhist.grid(True, alpha=0.3)
            self.ax_qhist.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        
        # 6. Reward history
        if len(self.reward_history) > 0:
            colors_reward = ['green' if r > 0 else 'red' if r < 0 else 'gray' for r in self.reward_history]
            self.ax_rewards.bar(range(len(self.reward_history)), list(self.reward_history), 
                              color=colors_reward, alpha=0.7)
            self.ax_rewards.set_title('Reward History', fontsize=12)
            self.ax_rewards.set_xlabel('Recent Frames')
            self.ax_rewards.set_ylabel('Reward')
            self.ax_rewards.grid(True, alpha=0.3)
            self.ax_rewards.axhline(y=0, color='k', linestyle='-', linewidth=1)
        
        # 7. Timing analysis
        if len(self.frame_times) > 0:
            self.ax_timing.plot(list(self.frame_times), color='purple', linewidth=2)
            self.ax_timing.set_title('Frame Processing Time', fontsize=12)
            self.ax_timing.set_xlabel('Recent Frames')
            self.ax_timing.set_ylabel('Time (ms)')
            self.ax_timing.grid(True, alpha=0.3)
            
            # Add target FPS line
            target_time = (1000 / TARGET_FPS) if TARGET_FPS else 100
            self.ax_timing.axhline(y=target_time, color='red', linestyle='--', 
                                  label=f'Target: {target_time:.1f}ms ({TARGET_FPS} FPS)')
            self.ax_timing.legend()
        
        # 8. Episode statistics
        elapsed = time.time() - self.episode_start
        fps = self.episode_frames / elapsed if elapsed > 0 else 0
        avg_timing = np.mean(self.frame_times) if len(self.frame_times) > 0 else 0
        
        stats_text = f"""
EPISODE STATISTICS

Frame: {self.frame_count}
Episode Frames: {self.episode_frames}
Episode Reward: {self.episode_reward:.2f}
Episode Duration: {elapsed:.1f}s
Current FPS: {fps:.1f}

AGENT STATE
Epsilon: {self.agent.epsilon:.4f}
Total Steps: {self.agent.total_steps:,}
Buffer Size: {len(self.agent.replay_buffer):,}

PERFORMANCE
Avg Frame Time: {avg_timing:.2f}ms
Target Frame Time: {(1000/TARGET_FPS) if TARGET_FPS else 100:.2f}ms
Max Q-Value: {q_values.max():.3f}
Min Q-Value: {q_values.min():.3f}
        """
        
        self.ax_stats.text(0.1, 0.95, stats_text, transform=self.ax_stats.transAxes,
                          fontsize=10, verticalalignment='top', fontfamily='monospace',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Update display
        plt.pause(0.001)
    
    def print_frame_info(self, frame_num, q_values, action, is_random, reward, timing_ms):
        """Print detailed frame information to console"""
        action_names = ['NOTHING', 'JUMP', 'DUCK']
        action_str = action_names[action]
        
        if is_random:
            selection = "🎲 RANDOM"
        else:
            selection = "🎯 GREEDY"
        
        print(f"\n{'='*70}")
        print(f"FRAME {frame_num}")
        print(f"{'='*70}")
        print(f"Q-Values:")
        print(f"  NOTHING: {q_values[0]:7.4f}")
        print(f"  JUMP:    {q_values[1]:7.4f}")
        print(f"  DUCK:    {q_values[2]:7.4f}")
        print(f"\nAction Selected: {action_str} ({selection})")
        print(f"Reward: {reward:+.2f}")
        print(f"Processing Time: {timing_ms:.2f}ms")
        print(f"Epsilon: {self.agent.epsilon:.4f}")
        print(f"{'='*70}")
    
    def run_debug_session(self, frames_per_episode=1000):
        """Run a debugging session"""
        print("\n" + "="*70)
        print("DEBUG DASHBOARD - DQN Analysis")
        print("="*70)
        print("\nInitializing agent...")
        self.initialize_agent()
        
        print("\nOpening game...")
        if not self.env.open_game():
            print("❌ Failed to open game!")
            return
        
        print("\n✅ Game opened!")
        print("\nControls:")
        print("  - Press Ctrl+C to stop")
        print("  - Watch the dashboard for real-time analysis")
        print("\nStarting in 3 seconds...")
        time.sleep(3)
        
        # Initialize frame stack
        print("\n[INIT] Building initial frame stack...")
        for i in range(N_FRAME_STACK):
            raw_frame = self.env.capture_frame()
            preprocessed_frame, gray_frame, _ = self.env.preprocess_frame(raw_frame)
            self.env.add_frame_to_stack(preprocessed_frame)
            self.env.update_crash_detection_buffer(gray_frame)
            time.sleep(0.05)
        print("[INIT] Frame stack ready!\n")
        
        self.episode_start = time.time()
        
        try:
            while self.episode_frames < frames_per_episode:
                frame_start = time.time()
                
                # Get state
                state = self.env.get_stacked_state()
                
                # Get Q-values and action
                q_values, action, is_random = self.get_q_values_and_action(state)
                
                # Execute action
                self.env.execute_action(action)
                
                # Capture next frame
                raw_frame = self.env.capture_frame()
                preprocessed_frame, gray_frame, resized = self.env.preprocess_frame(raw_frame)
                self.env.add_frame_to_stack(preprocessed_frame)
                
                # Check game over
                game_over = self.env.is_game_over(gray_frame)
                self.env.update_crash_detection_buffer(gray_frame)
                
                # Calculate reward
                if game_over:
                    reward = REWARD_DEATH
                    print("\n💥 CRASH DETECTED!")
                else:
                    reward = REWARD_ALIVE
                
                # Calculate timing
                timing_ms = (time.time() - frame_start) * 1000
                
                # Update visualization
                self.update_visualization(state, q_values, action, is_random, reward, raw_frame, timing_ms)
                
                # Print frame info every 10 frames
                if self.frame_count % 10 == 0:
                    self.print_frame_info(self.frame_count, q_values, action, is_random, reward, timing_ms)
                
                self.frame_count += 1
                
                # Handle game over
                if game_over:
                    print(f"\n{'='*70}")
                    print(f"EPISODE COMPLETE")
                    print(f"{'='*70}")
                    print(f"Frames: {self.episode_frames}")
                    print(f"Total Reward: {self.episode_reward:.2f}")
                    print(f"Duration: {time.time() - self.episode_start:.2f}s")
                    print(f"{'='*70}\n")
                    
                    print("Restarting in 3 seconds...")
                    time.sleep(3)
                    self.env.restart_game()
                    time.sleep(2)
                    
                    # Reset episode stats
                    self.episode_frames = 0
                    self.episode_reward = 0
                    self.episode_start = time.time()
                    
                    # Rebuild frame stack
                    self.env.reset_buffers()
                    for i in range(N_FRAME_STACK):
                        raw_frame = self.env.capture_frame()
                        preprocessed_frame, gray_frame, _ = self.env.preprocess_frame(raw_frame)
                        self.env.add_frame_to_stack(preprocessed_frame)
                        self.env.update_crash_detection_buffer(gray_frame)
                        time.sleep(0.05)
                
                # FPS limiting if enabled
                if TARGET_FPS:
                    elapsed = time.time() - frame_start
                    target_time = 1.0 / TARGET_FPS
                    if elapsed < target_time:
                        time.sleep(target_time - elapsed)
        
        except KeyboardInterrupt:
            print("\n\n⏹️  Debug session stopped by user")
        
        finally:
            plt.ioff()
            plt.close('all')
            print("\n✅ Debug dashboard closed")


def main():
    """Main entry point"""
    dashboard = DebugDashboard()
    dashboard.run_debug_session(frames_per_episode=10000)


if __name__ == "__main__":
    main()
