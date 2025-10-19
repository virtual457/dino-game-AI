"""
Training coordinator for DDQN Dino Game Agent

10 FPS MODE (100ms per frame):
- Each frame takes exactly 0.1s (100ms)
- Consistent timing for reproducible training
- Tracks frames that exceeded target time (slow frames)
"""
import time
import os
import numpy as np
import torch
from config import *
from game_env import DinoGameEnv
from agent import DDQNAgent
from rewards import calculate_reward
from episode_tracker import EpisodeTracker
from episode_visualizer import EpisodeVisualizer


class Trainer:
    """Coordinates DDQN training with FPS lock from config"""
    
    def __init__(self):
        self.env = DinoGameEnv()
        self.agent = None
        self.episode_tracker = EpisodeTracker()
        self.episode_visualizer = EpisodeVisualizer()
        
        # Episode state
        self.step_count = 0
        self.total_reward = 0
        self.losses = []
        self.last_save_step = 0
        self.episode_start_time = None
        self.episode_real_start_time = None  # For FPS calculation
        
        # Episode buffer - stores ALL transitions until episode ends
        self.episode_experiences = []
        self.frame_timestamps = []
        self.step_times = []  # Track processing time per step
        self.slow_frames = 0  # Count frames that exceeded target time
    
    def initialize_agent(self):
        """Initialize DDQN agent"""
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
        
        # Try to load checkpoint
        if os.path.exists(CHECKPOINT_PATH):
            print(f"\n[LOAD] Found existing model: {CHECKPOINT_PATH}")
            try:
                self.agent.load(CHECKPOINT_PATH)
                print("[OK] Resuming training!")
            except Exception as e:
                print(f"[WARN] Error loading: {e}")
                print("[RESET] Starting fresh...")
                backup_path = CHECKPOINT_PATH + ".corrupted"
                if os.path.exists(CHECKPOINT_PATH):
                    os.rename(CHECKPOINT_PATH, backup_path)
        else:
            print("\n[NEW] Starting fresh training!")
    
    def run_training_step(self):
        """Execute one training step with proper timing"""
        step_start = time.time()
        
        # 1. Get current state (4 stacked frames)
        state = self.env.get_stacked_state()
        
        # 2. Select action (network inference)
        action = self.agent.select_action(state, training=True)
        
        # 3. Execute action (press/release key)
        self.env.execute_action(action)
        
        # Check if we exceeded target time BEFORE waiting
        elapsed_before_wait = time.time() - step_start
        target_time_ms = FRAME_TIME * 1000  # Convert to ms
        
        # 4. FPS limiter: Wait to let game react BEFORE capturing
        elapsed = time.time() - step_start
        if elapsed < FRAME_TIME:
            time.sleep(FRAME_TIME - elapsed)
        else:
            # Frame took longer than target - count as slow frame
            self.slow_frames += 1
        
        # 5. Capture NEXT frame (game has now reacted to our action)
        raw_frame = self.env.capture_frame()
        preprocessed_frame, gray_frame, resized_frame = self.env.preprocess_frame(raw_frame)
        
        # Add to frame stack
        self.env.add_frame_to_stack(preprocessed_frame)
        next_state = self.env.get_stacked_state()
        
        # 6. Check crash in NEXT state
        game_over = self.env.is_game_over(gray_frame)
        if game_over and len(self.env.crash_detection_buffer) >= 2:
            print(f"\nCRASH detected at step {self.step_count}!")
        
        self.env.update_crash_detection_buffer(gray_frame)
        
        # 7. Calculate base reward (NO death penalty yet)
        if game_over:
            reward = 0.0  # Placeholder - frozen frames will be discarded anyway
        else:
            reward = calculate_reward(action, game_over=False)
        
        # Store transition in EPISODE buffer
        timestamp = time.time() - self.episode_start_time
        self.episode_experiences.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': game_over,
            'frame_num': self.step_count,
            'timestamp': timestamp
        })
        
        # Track performance (total time for this step)
        step_time = (time.time() - step_start) * 1000
        self.step_times.append(step_time)
        
        self.step_count += 1
        
        # Progress with ACTUAL FPS (every 10 frames)
        if self.step_count % 10 == 0:
            total_elapsed = time.time() - self.episode_real_start_time
            fps = self.step_count / total_elapsed if total_elapsed > 0 else 0
            avg_time = np.mean(self.step_times[-10:])
            slow_pct = (self.slow_frames / self.step_count * 100) if self.step_count > 0 else 0
            print(f"Step {self.step_count} | FPS: {fps:.1f} | "
                  f"{avg_time:.1f}ms/frame | Slow: {slow_pct:.1f}% | "
                  f"e: {self.agent.epsilon:.3f}",
                  end='\r')
        
        return game_over
    
    def _apply_reward_shaping(self):
        """Apply reward shaping at episode end - ESCALATING PENALTIES FOR LAST 10 FRAMES"""
        if len(self.episode_experiences) < 3:
            print("[WARN] Episode too short for reward shaping")
            return 0, 0
        
        original_count = len(self.episode_experiences)
        
        # Remove the last 3 frames (frozen frames AFTER crash)
        frozen_count = 3
        valid_experiences = self.episode_experiences[:-frozen_count]
        
        if len(valid_experiences) == 0:
            print("[WARN] All frames were frozen - episode too short")
            self.episode_experiences = []
            return 0, frozen_count
        
        # Apply escalating penalties to last 10 frames (or fewer if episode is short)
        num_penalty_frames = min(CRASH_PENALTY_FRAMES, len(valid_experiences))
        
        # Generate penalties: -10, -9, -8, -7, -6, -5, -4, -3, -2, -1
        penalties = []
        for i in range(num_penalty_frames):
            # i=0 (last frame): -10, i=1 (N-1): -9, i=2 (N-2): -8, ..., i=9 (N-9): -1
            penalty = CRASH_PENALTY_START + i
            penalties.append(penalty)
        
        # Apply penalties to the last num_penalty_frames
        for i, penalty in enumerate(penalties):
            idx = -(i + 1)  # -1 (last), -2 (second last), etc.
            if abs(idx) <= len(valid_experiences):
                valid_experiences[idx]['reward'] = penalty
                if i == 0:  # Mark last frame as terminal
                    valid_experiences[idx]['done'] = True
        
        self.episode_experiences = valid_experiences
        
        print(f"\n[SHAPING] Original: {original_count} frames")
        print(f"[SHAPING] Removed {frozen_count} frozen frames (n+1, n+2, n+3)")
        print(f"[SHAPING] Applied escalating penalties to last {num_penalty_frames} frames:")
        for i, penalty in enumerate(penalties[:min(5, len(penalties))]):
            idx = -(i + 1)
            if abs(idx) <= len(valid_experiences):
                frame_num = valid_experiences[idx]['frame_num']
                print(f"         Frame {frame_num}: {penalty}")
        if num_penalty_frames > 5:
            print(f"         ... ({num_penalty_frames - 5} more frames with penalties -5 to -1)")
        print(f"[SHAPING] Valid experiences: {len(self.episode_experiences)}")
        
        return len(valid_experiences), frozen_count
    
    def _add_to_replay_and_train(self):
        """Add all shaped experiences to replay buffer and train"""
        if len(self.episode_experiences) == 0:
            print("[WARN] No experiences to add to replay buffer")
            return
        
        added_count = 0
        training_steps = 0
        
        for exp in self.episode_experiences:
            self.agent.store_transition(
                exp['state'],
                exp['action'],
                exp['reward'],
                exp['next_state'],
                exp['done']
            )
            added_count += 1
        
        if len(self.agent.replay_buffer) >= BATCH_SIZE:
            for _ in range(added_count):
                loss = self.agent.learn()
                if loss is not None:
                    self.losses.append(loss)
                    training_steps += 1
        
        # Calculate and print training statistics
        avg_loss = np.mean(self.losses) if self.losses else 0.0
        min_loss = np.min(self.losses) if self.losses else 0.0
        max_loss = np.max(self.losses) if self.losses else 0.0
        
        print(f"[BUFFER] Added {added_count} experiences to replay buffer")
        print(f"[BUFFER] Total buffer size: {len(self.agent.replay_buffer):,}/{BUFFER_CAPACITY:,}")
        print(f"[TRAIN] Performed {training_steps} training steps on this episode")
        print(f"[TRAIN] Loss - Avg: {avg_loss:.4f}, Min: {min_loss:.4f}, Max: {max_loss:.4f}")
    
    def handle_episode_end(self):
        """Handle episode end with reward shaping and buffer addition"""
        valid_count, discarded_count = self._apply_reward_shaping()
        self._add_to_replay_and_train()
        
        self.total_reward = sum(exp['reward'] for exp in self.episode_experiences)
        
        time.sleep(CRASH_PAUSE_DURATION)
        
        # Calculate stats
        duration = time.time() - self.episode_real_start_time
        fps = self.step_count / duration if duration > 0 else 0
        avg_time = np.mean(self.step_times)
        slow_frame_pct = (self.slow_frames / self.step_count * 100) if self.step_count > 0 else 0
        
        avg_loss_ep = np.mean(self.losses) if self.losses else 0.0
        
        # Calculate target network updates
        target_updates = self.agent.learn_step_counter // TARGET_UPDATE_FREQ
        
        print(f"\n\n{'='*70}")
        print(f"EPISODE {self.episode_tracker.episode_number + 1} COMPLETE")
        print(f"{'='*70}")
        print(f"Frames:      {self.step_count}")
        print(f"Duration:    {duration:.2f}s")
        print(f"FPS:         {fps:.1f}")
        print(f"Avg Time:    {avg_time:.2f}ms/frame")
        print(f"Min Time:    {np.min(self.step_times):.2f}ms")
        print(f"Max Time:    {np.max(self.step_times):.2f}ms")
        print(f"Slow Frames: {self.slow_frames}/{self.step_count} ({slow_frame_pct:.1f}%)")
        print(f"Reward:      {self.total_reward:.2f}")
        print(f"Avg Loss:    {avg_loss_ep:.4f}")
        print(f"Epsilon:     {self.agent.epsilon:.4f}")
        print(f"Target Updates: {target_updates} (every {TARGET_UPDATE_FREQ} steps)")
        print(f"Added:       {valid_count} experiences")
        print(f"Discarded:   {discarded_count} frozen frames")
        print(f"{'='*70}\n")
        
        # Track slow frame percentage in episode tracker
        avg_reward = self.episode_tracker.record_episode(self.total_reward, slow_frame_pct)
        avg_loss = np.mean(self.losses) if self.losses else 0.0
        
        # Count actions
        action_counts = {0: 0, 1: 0, 2: 0}
        for exp in self.episode_experiences:
            action_counts[exp['action']] += 1
        
        total = sum(action_counts.values())
        action_pcts = {k: (v/total*100) if total > 0 else 0 for k, v in action_counts.items()}
        
        # Prepare frames for display
        first_10 = self.episode_experiences[:10] if len(self.episode_experiences) >= 10 else self.episode_experiences
        last_10 = self.episode_experiences[-10:] if len(self.episode_experiences) >= 10 else self.episode_experiences
        
        first_10_display = [{
            'frame_num': exp['frame_num'],
            'image': (exp['state'][3] * 255).astype(np.uint8),
            'action': exp['action'],
            'reward': exp['reward'],
            'timestamp': exp['timestamp']
        } for exp in first_10]
        
        last_10_display = [{
            'frame_num': exp['frame_num'],
            'image': (exp['state'][3] * 255).astype(np.uint8),
            'action': exp['action'],
            'reward': exp['reward'],
            'timestamp': exp['timestamp']
        } for exp in last_10]
        
        # Identify crash frame
        crash_frame_idx = -1
        for i in range(len(last_10_display)-1, -1, -1):
            if last_10_display[i]['reward'] == REWARD_DEATH:
                crash_frame_idx = i
                break
        
        # Show visualization
        self.episode_visualizer.show_episode_summary({
            'episode_num': self.episode_tracker.episode_number,
            'steps': self.step_count,
            'total_reward': self.total_reward,
            'avg_reward_1000': avg_reward,
            'avg_frame_time': avg_time,
            'avg_loss': avg_loss,
            'epsilon': self.agent.epsilon,
            'buffer_size': len(self.agent.replay_buffer),
            'jumps': action_counts[1],
            'ducks': action_counts[2],
            'nothing': action_counts[0],
            'jump_pct': action_pcts[1],
            'duck_pct': action_pcts[2],
            'nothing_pct': action_pcts[0],
            'movement_pct_500': 0.0,
            'experiences_added': valid_count,
            'experiences_discarded': discarded_count,
            'first_10_frames': first_10_display,
            'last_10_frames': last_10_display,
            'crash_frame_idx': crash_frame_idx
        })
        
        # Plot if needed
        if self.episode_tracker.should_plot():
            self.episode_tracker.generate_plot()
        
        # Restart
        self.env.restart_game()
        self.reset_episode()
    
    def reset_episode(self):
        """Reset episode state"""
        self.step_count = 0
        self.total_reward = 0
        self.losses = []
        self.slow_frames = 0  # Reset slow frame counter
        self.env.reset_buffers()
        self.episode_start_time = time.time()
        self.episode_real_start_time = time.time()
        self.episode_experiences = []
        self.frame_timestamps = []
        self.step_times = []
        
        # Initialize frame stack with 4 real frames before training starts
        print("[INIT] Capturing initial 4 frames for frame stack...")
        for i in range(N_FRAME_STACK):
            raw_frame = self.env.capture_frame()
            preprocessed_frame, gray_frame, _ = self.env.preprocess_frame(raw_frame)
            self.env.add_frame_to_stack(preprocessed_frame)
            self.env.update_crash_detection_buffer(gray_frame)
            time.sleep(0.05)
        print("[INIT] Frame stack initialized with 4 real frames!")
    
    def auto_save(self):
        """Auto-save if needed"""
        if self.agent.total_steps - self.last_save_step >= SAVE_INTERVAL:
            print(f"\n[SAVE] Auto-saving at step {self.agent.total_steps}...")
            self.agent.save(CHECKPOINT_PATH)
            self.last_save_step = self.agent.total_steps
            print("[OK] Saved!\n")
    
    def cleanup(self):
        """Cleanup resources"""
        self.episode_visualizer.close()
        
        if self.agent and self.agent.device.type == 'cuda':
            print("\n[CLEANUP] Cleaning GPU...")
            torch.cuda.empty_cache()
        
        print("\n[COMPLETE] Training finished!")
        print(f"   Buffer: {len(self.agent.replay_buffer):,}")
        print(f"   Steps: {self.agent.total_steps:,}")
        print(f"   Epsilon: {self.agent.epsilon:.4f}")
