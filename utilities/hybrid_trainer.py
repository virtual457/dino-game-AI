"""
Hybrid Online/Offline Training Strategy with Buffer Persistence

NEW STRATEGY:
1. Collect 1000 frames with end-of-episode training (max(frames_survived, 32) steps)
2. Offline train until MSE < 1.0 or max 40000 steps (batches of 100)
3. Copy weights to target network
4. Repeat cycle
5. Ctrl+C saves everything before exit

Note: End-of-episode training samples randomly from the entire replay buffer
"""
import os
import pickle
import signal
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import *
from game_env import DinoGameEnv
from agent import DDQNAgent
from rewards import calculate_reward
from episode_tracker import EpisodeTracker
from episode_visualizer import EpisodeVisualizer


class HybridTrainer:
    """Combines online end-of-episode training and offline batch training"""
    
    def __init__(self):
        self.env = DinoGameEnv()
        self.agent = None
        self.episode_tracker = EpisodeTracker()
        self.episode_visualizer = EpisodeVisualizer()
        
        # Training state
        self.total_frames_collected = 0
        self.total_training_steps = 0
        self.cycle_number = 0
        self.should_exit = False
        
        # Episode tracking for visualization
        self.episode_rewards = []  # Track rewards per episode
        self.episode_lengths = []  # Track frames per episode
        self.episode_losses = []   # Track average loss per episode
        
        # Latest episode data for visualization
        self.latest_episode_experiences = []  # Store last complete episode
        self.current_episode_experiences = []  # Current episode being played
        
        # Buffer persistence
        self.buffer_save_path = os.path.join(CHECKPOINT_DIR, "replay_buffer.pkl")
        
        # Setup signal handler for Ctrl+C
        signal.signal(signal.SIGINT, self.signal_handler)
        
    def signal_handler(self, sig, frame):
        """Handle Ctrl+C gracefully"""
        print("\n\n" + "="*70)
        print("[CTRL+C] Interrupt detected! Saving before exit...")
        print("="*70)
        self.should_exit = True
        self.save_checkpoint_and_buffer()
        self.cleanup()
        print("\n[EXIT] Training saved and closed gracefully!")
        sys.exit(0)
    
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
                print("[OK] Model loaded!")
            except Exception as e:
                print(f"[WARN] Error loading: {e}")
        
        # Try to load buffer
        if os.path.exists(self.buffer_save_path):
            print(f"\n[LOAD] Found existing buffer: {self.buffer_save_path}")
            try:
                self.load_buffer()
                print(f"[OK] Buffer loaded! Size: {len(self.agent.replay_buffer):,}")
            except Exception as e:
                print(f"[WARN] Error loading buffer: {e}")
    
    def initial_offline_training(self, num_steps=15):
        """Perform initial offline training before starting online training
        
        Args:
            num_steps: Number of training steps to perform (default 15)
        """
        if len(self.agent.replay_buffer) < BATCH_SIZE:
            print(f"\n[SKIP] Initial offline training skipped (buffer size {len(self.agent.replay_buffer)} < {BATCH_SIZE})")
            return
        
        print(f"\n{'='*70}")
        print(f"INITIAL OFFLINE TRAINING")
        print(f"Training steps: {num_steps}")
        print(f"Buffer size: {len(self.agent.replay_buffer):,}/{BUFFER_CAPACITY:,}")
        print(f"{'='*70}\n")
        
        losses = []
        for step in range(num_steps):
            loss = self.agent.learn()
            if loss is not None:
                losses.append(loss)
                self.total_training_steps += 1
            
            if (step + 1) % 5 == 0 or step == num_steps - 1:
                avg_loss = np.mean(losses) if len(losses) > 0 else 0.0
                print(f"  Step {step+1:>2}/{num_steps} | Avg MSE: {avg_loss:.4f}")
        
        if len(losses) > 0:
            final_mse = np.mean(losses)
            print(f"\n[OK] Initial offline training complete! Avg MSE: {final_mse:.4f}")
        else:
            print(f"\n[WARN] No losses recorded during initial training")
        
        print(f"{'='*70}\n")
    
    def save_buffer(self):
        """Save replay buffer to disk"""
        print(f"[SAVE] Saving buffer to {self.buffer_save_path}...")
        try:
            buffer_data = {
                'buffer': list(self.agent.replay_buffer.buffer),  # Convert deque to list
                'size': len(self.agent.replay_buffer)
            }
            with open(self.buffer_save_path, 'wb') as f:
                pickle.dump(buffer_data, f)
            print(f"[OK] Buffer saved! Size: {len(self.agent.replay_buffer):,}")
        except Exception as e:
            print(f"[ERROR] Failed to save buffer: {e}")
    
    def load_buffer(self):
        """Load replay buffer from disk"""
        from collections import deque
        with open(self.buffer_save_path, 'rb') as f:
            buffer_data = pickle.load(f)
        
        # Convert list back to deque with same maxlen
        capacity = self.agent.replay_buffer.buffer.maxlen
        self.agent.replay_buffer.buffer = deque(buffer_data['buffer'], maxlen=capacity)
    
    def save_checkpoint_and_buffer(self):
        """Save both model and buffer"""
        print(f"\n[SAVE] Saving checkpoint and buffer...")
        self.agent.save(CHECKPOINT_PATH)
        self.save_buffer()
        print(f"[OK] Checkpoint and buffer saved!")
    
    def train_end_of_episode(self, frames_survived):
        """
        Train at end of episode: max(frames_survived, 32) steps
        Each step samples a random batch from the replay buffer
        
        Args:
            frames_survived: Number of frames in the episode
        """
        if len(self.agent.replay_buffer) < BATCH_SIZE:
            return []
        
        # Train for max(n, 32) steps where n = frames_survived
        num_training_steps = max(frames_survived, 32)
        print(f"\n[EOE TRAIN] Training {num_training_steps:,} steps (max({frames_survived} frames, 32))...")
        
        losses = []
        for step in range(num_training_steps):
            # Each learn() call already samples randomly from replay buffer
            loss = self.agent.learn()
            if loss is not None:
                losses.append(loss)
                self.total_training_steps += 1
            
            # Progress every 100 steps for shorter episodes
            if (step + 1) % 100 == 0:
                avg_loss = np.mean(losses[-100:]) if len(losses) >= 100 else np.mean(losses)
                print(f"  Step {step+1:>6}/{num_training_steps:,} | MSE: {avg_loss:.4f}", end='\r')
        
        if len(losses) > 0:
            avg_loss = np.mean(losses)
            print(f"\n[OK] End-of-episode training complete! Avg MSE: {avg_loss:.4f}")
        
        return losses
    
    def collect_frames_with_eoe_training(self, num_frames):
        """
        Collect frames and train at end of each episode
        
        Args:
            num_frames: Number of frames to collect
        """
        print(f"\n{'='*70}")
        print(f"COLLECTION PHASE: Collecting {num_frames} frames")
        print(f"Training: End-of-episode (frames_survived × 32 samples)")
        print(f"Policy: Network with epsilon={self.agent.epsilon:.4f}")
        print(f"Buffer: {len(self.agent.replay_buffer):,}/{BUFFER_CAPACITY:,} ({len(self.agent.replay_buffer)/BUFFER_CAPACITY*100:.1f}% full)")
        print(f"{'='*70}\n")
        
        frames_collected = 0
        episodes_played = 0
        episode_frame_count = 0
        episode_reward = 0.0
        self.current_episode_experiences = []  # Reset current episode
        
        # Initialize frame stack
        self.env.reset_buffers()
        for i in range(N_FRAME_STACK):
            raw_frame = self.env.capture_frame()
            preprocessed_frame, gray_frame, _ = self.env.preprocess_frame(raw_frame)
            self.env.add_frame_to_stack(preprocessed_frame)
            self.env.update_crash_detection_buffer(gray_frame)
        
        while frames_collected < num_frames and not self.should_exit:
            # Get state and action
            state = self.env.get_stacked_state()
            action = self.agent.select_action(state, training=True)
            
            # Execute action
            self.env.execute_action(action)
            
            # Wait for frame time
            import time
            time.sleep(FRAME_TIME)
            
            # Capture next frame
            raw_frame = self.env.capture_frame()
            preprocessed_frame, gray_frame, resized_frame = self.env.preprocess_frame(raw_frame)
            self.env.add_frame_to_stack(preprocessed_frame)
            next_state = self.env.get_stacked_state()
            
            # Check crash
            game_over = self.env.is_game_over(gray_frame)
            self.env.update_crash_detection_buffer(gray_frame)
            
            # Calculate reward
            if game_over:
                reward = REWARD_DEATH
            else:
                reward = calculate_reward(action, game_over=False)
            
            # Store experience
            self.agent.store_transition(state, action, reward, next_state, game_over)
            
            # Also store for visualization
            self.current_episode_experiences.append({
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state,
                'done': game_over,
                'frame_num': episode_frame_count,
                'timestamp': frames_collected / (TARGET_FPS if TARGET_FPS else 10)
            })
            
            frames_collected += 1
            episode_frame_count += 1
            episode_reward += reward
            self.total_frames_collected += 1
            
            # Progress
            if frames_collected % 100 == 0:
                print(f"Collected: {frames_collected}/{num_frames} frames | "
                      f"Buffer: {len(self.agent.replay_buffer):,}/{BUFFER_CAPACITY:,} | "
                      f"Episodes: {episodes_played} | "
                      f"ε: {self.agent.epsilon:.3f}", end='\r')
            
            # Handle crash
            if game_over:
                episodes_played += 1
                
                # Record episode stats (raw rewards - what's in buffer)
                self.episode_lengths.append(episode_frame_count)
                self.episode_rewards.append(episode_reward)
                
                # Save latest episode for visualization (raw rewards - what's in buffer)
                self.latest_episode_experiences = self.current_episode_experiences.copy()
                
                print(f"\n[EPISODE {episodes_played}] {episode_frame_count} frames, reward: {episode_reward:.2f}")
                
                # END-OF-EPISODE TRAINING
                eoe_losses = self.train_end_of_episode(episode_frame_count)
                if len(eoe_losses) > 0:
                    self.episode_losses.append(np.mean(eoe_losses))
                
                # Visualize this episode
                self.visualize_latest_episode()
                
                # Update training plot every 10 episodes
                if episodes_played % 10 == 0:
                    self.update_training_plot()
                
                # Reset episode counters
                episode_frame_count = 0
                episode_reward = 0.0
                self.current_episode_experiences = []
                
                import time
                time.sleep(CRASH_PAUSE_DURATION)
                self.env.restart_game()
                
                # Re-initialize frame stack
                self.env.reset_buffers()
                for i in range(N_FRAME_STACK):
                    raw_frame = self.env.capture_frame()
                    preprocessed_frame, gray_frame, _ = self.env.preprocess_frame(raw_frame)
                    self.env.add_frame_to_stack(preprocessed_frame)
                    self.env.update_crash_detection_buffer(gray_frame)
        
        print(f"\n\n[DONE] Collected {frames_collected} frames in {episodes_played} episodes")
        print(f"[INFO] Buffer size: {len(self.agent.replay_buffer):,}/{BUFFER_CAPACITY:,}")
        print(f"[INFO] Buffer fill: {len(self.agent.replay_buffer)/BUFFER_CAPACITY*100:.1f}%")
        
        # Print episode statistics
        if len(self.episode_lengths) > 0:
            recent = min(10, len(self.episode_lengths))
            avg_length = np.mean(self.episode_lengths[-recent:])
            avg_reward = np.mean(self.episode_rewards[-recent:])
            print(f"[STATS] Last {recent} episodes: Avg {avg_length:.1f} frames, Avg reward {avg_reward:.2f}")
    
    def offline_train_until_converged(self, target_mse=1.0, batch_size=100, max_steps=40000):
        """
        Train offline until MSE falls below threshold
        
        Args:
            target_mse: Target MSE to achieve (default 1.0)
            batch_size: Number of training steps per batch (default 100)
            max_steps: Maximum training steps before stopping (default 40000)
        """
        print(f"\n{'='*70}")
        print(f"OFFLINE TRAINING PHASE")
        print(f"Target MSE: {target_mse} | Batch size: {batch_size} | Max steps: {max_steps:,}")
        print(f"Buffer: {len(self.agent.replay_buffer):,}/{BUFFER_CAPACITY:,} ({len(self.agent.replay_buffer)/BUFFER_CAPACITY*100:.1f}% full)")
        print(f"{'='*70}\n")
        
        if len(self.agent.replay_buffer) < BATCH_SIZE:
            print(f"[SKIP] Not enough experiences ({len(self.agent.replay_buffer)} < {BATCH_SIZE})")
            return
        
        training_step = 0
        recent_losses = []
        
        while training_step < max_steps and not self.should_exit:
            # Train for batch_size steps
            batch_losses = []
            for _ in range(batch_size):
                loss = self.agent.learn()
                if loss is not None:
                    batch_losses.append(loss)
                    recent_losses.append(loss)
                    training_step += 1
                    self.total_training_steps += 1
            
            # Keep only last 1000 losses for averaging
            if len(recent_losses) > 1000:
                recent_losses = recent_losses[-1000:]
            
            # Calculate average loss
            if len(batch_losses) > 0:
                avg_batch_loss = np.mean(batch_losses)
                avg_recent_loss = np.mean(recent_losses[-100:]) if len(recent_losses) >= 100 else np.mean(recent_losses)
                
                print(f"Step {training_step:>6}/{max_steps:,} | "
                      f"Batch MSE: {avg_batch_loss:.4f} | "
                      f"Recent MSE: {avg_recent_loss:.4f} | "
                      f"ε: {self.agent.epsilon:.4f}", end='\r')
                
                # Print full line every 1000 steps
                if training_step % 1000 == 0:
                    print()  # New line for clarity
                
                # Check convergence
                if avg_recent_loss < target_mse and len(recent_losses) >= 100:
                    print(f"\n[CONVERGED] Reached target MSE {target_mse:.4f}!")
                    print(f"[INFO] Total training steps: {training_step:,}")
                    break
        
        if training_step >= max_steps:
            print(f"\n[MAX STEPS] Reached maximum {max_steps:,} steps")
            if len(recent_losses) >= 100:
                final_mse = np.mean(recent_losses[-100:])
                print(f"[INFO] Final MSE: {final_mse:.4f}")
        
        print(f"[INFO] Buffer size after training: {len(self.agent.replay_buffer):,}/{BUFFER_CAPACITY:,}")
    
    def copy_to_target_network(self):
        """Manually copy policy network weights to target network"""
        print(f"\n[TARGET] Copying policy network → target network...")
        self.agent.target_net.load_state_dict(self.agent.policy_net.state_dict())
        print(f"[OK] Weights copied!")
    
    def run_training_cycle(self):
        """Execute one complete training cycle"""
        self.cycle_number += 1
        
        print(f"\n\n{'#'*70}")
        print(f"{'#'*70}")
        print(f"##  TRAINING CYCLE {self.cycle_number}")
        print(f"##  Total frames collected: {self.total_frames_collected:,}")
        print(f"##  Total training steps: {self.total_training_steps:,}")
        print(f"##  Buffer size: {len(self.agent.replay_buffer):,}/{BUFFER_CAPACITY:,}")
        print(f"##  Current epsilon: {self.agent.epsilon:.4f}")
        print(f"{'#'*70}")
        print(f"{'#'*70}\n")
        
        # Phase 1: Collect 1000 frames with end-of-episode training
        self.collect_frames_with_eoe_training(1000)
        
        if self.should_exit:
            return
        
        # Phase 2: Offline train until MSE < 1.0 or max 40000 steps
        self.offline_train_until_converged(target_mse=1.0, batch_size=100, max_steps=40000)
        
        if self.should_exit:
            return
        
        # Phase 3: Copy weights to target network
        self.copy_to_target_network()
        
        # Save everything after each cycle
        self.save_checkpoint_and_buffer()
        
        # Generate final training plot for this cycle
        self.update_training_plot()
        
        print(f"\n{'='*70}")
        print(f"CYCLE {self.cycle_number} COMPLETE")
        
        # Print summary
        if len(self.episode_lengths) > 0:
            print(f"\nCycle {self.cycle_number} Summary:")
            print(f"  Total episodes: {len(self.episode_lengths)}")
            print(f"  Avg episode length: {np.mean(self.episode_lengths):.1f} frames")
            print(f"  Max episode length: {max(self.episode_lengths)} frames")
            print(f"  Avg reward (last 10): {np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else np.mean(self.episode_rewards):.2f}")
            print(f"  Buffer fill: {len(self.agent.replay_buffer):,}/{BUFFER_CAPACITY:,} ({len(self.agent.replay_buffer)/BUFFER_CAPACITY*100:.1f}%)")
            print(f"  Current epsilon: {self.agent.epsilon:.4f}")
        
        print(f"{'='*70}\n")
    
    def update_training_plot(self):
        """Generate training progress plot similar to main trainer"""
        if len(self.episode_rewards) <= 1:
            return
        
        print(f"\n[PLOT] Generating training progress plot...")
        
        # Calculate rolling average (last 100 episodes or all if less)
        window_size = min(100, len(self.episode_rewards))
        avg_rewards = []
        for i in range(len(self.episode_rewards)):
            start_idx = max(0, i - window_size + 1)
            avg_rewards.append(np.mean(self.episode_rewards[start_idx:i+1]))
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot episode rewards
        episodes = list(range(1, len(self.episode_rewards) + 1))
        ax.plot(episodes, self.episode_rewards, alpha=0.3, color='blue', label='Episode Reward')
        ax.plot(episodes, avg_rewards, color='red', linewidth=2, label=f'Avg Last {window_size} Episodes')
        
        # Formatting
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Total Reward', fontsize=12)
        ax.set_title(f'Hybrid Training Progress - Cycle {self.cycle_number}, Episode {len(self.episode_rewards)}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add stats text box
        if len(self.episode_rewards) >= 10:
            recent_avg = np.mean(self.episode_rewards[-10:])
            stats_text = f"Last 10 Episodes Avg: {recent_avg:.2f}\n"
            stats_text += f"Best Episode: {max(self.episode_lengths)} frames\n"
            stats_text += f"Current Epsilon: {self.agent.epsilon:.4f}"
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Save plot
        plot_path = os.path.join(CHECKPOINT_DIR, "training_progress.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        print(f"[OK] Plot saved: {plot_path}")
    
    def visualize_latest_episode(self):
        """Visualize the latest complete episode"""
        if len(self.latest_episode_experiences) == 0:
            return
        
        # Count actions
        action_counts = {0: 0, 1: 0, 2: 0}
        for exp in self.latest_episode_experiences:
            action_counts[exp['action']] += 1
        
        total_actions = sum(action_counts.values())
        action_pcts = {k: (v/total_actions*100) if total_actions > 0 else 0 
                      for k, v in action_counts.items()}
        
        # Calculate episode stats
        episode_length = len(self.latest_episode_experiences)
        episode_reward = sum(exp['reward'] for exp in self.latest_episode_experiences)
        
        # Prepare frames for display (first 10 and last 10)
        first_10 = self.latest_episode_experiences[:10] if episode_length >= 10 else self.latest_episode_experiences
        last_10 = self.latest_episode_experiences[-10:] if episode_length >= 10 else self.latest_episode_experiences
        
        first_10_display = [{
            'frame_num': exp['frame_num'],
            'image': (exp['state'][3] * 255).astype(np.uint8),  # Last frame of stack
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
            if last_10_display[i]['reward'] == REWARD_DEATH or last_10_display[i]['reward'] < 0:
                crash_frame_idx = i
                break
        
        # Calculate average loss
        avg_loss = np.mean(self.episode_losses[-10:]) if len(self.episode_losses) >= 10 else \
                   (np.mean(self.episode_losses) if len(self.episode_losses) > 0 else 0.0)
        
        # Show visualization
        self.episode_visualizer.show_episode_summary({
            'episode_num': len(self.episode_lengths),
            'steps': episode_length,
            'total_reward': episode_reward,
            'avg_reward_1000': np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards),
            'avg_frame_time': 1000.0 / (TARGET_FPS if TARGET_FPS else 10),  # ms per frame
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
            'experiences_added': episode_length,
            'experiences_discarded': 0,
            'first_10_frames': first_10_display,
            'last_10_frames': last_10_display,
            'crash_frame_idx': crash_frame_idx
        })
    
    def cleanup(self):
        """Cleanup resources"""
        self.episode_visualizer.close()
        
        if self.agent and self.agent.device.type == 'cuda':
            print("\n[CLEANUP] Cleaning GPU...")
            torch.cuda.empty_cache()


def main():
    """Main training loop"""
    print("\n" + "="*70)
    print(" " * 15 + "HYBRID ONLINE/OFFLINE TRAINING - V2")
    print("="*70)
    print("\nNEW Strategy:")
    print("  1. Collect 1000 frames")
    print("  2. Train at end of each episode: max(n, 32) steps")
    print("     - Short episodes (n < 32): Train for 32 steps")
    print("     - Long episodes (n >= 32): Train for n steps")
    print("     - Each step samples randomly from entire replay buffer")
    print("  3. After 1000 frames: Offline train until MSE < 1.0 (max 40k steps)")
    print("  4. Copy weights to target network")
    print("  5. Repeat cycle")
    print("  6. Press Ctrl+C anytime to save and exit gracefully")
    print("="*70 + "\n")
    
    trainer = HybridTrainer()
    
    # Initialize
    print("[INIT] Opening game...")
    if not trainer.env.open_game():
        print("[ERROR] Failed to open game!")
        return
    
    print("[INIT] Initializing agent...")
    trainer.initialize_agent()
    
    # Perform initial offline training (15 steps) before starting online training
    # This helps stabilize the network before gameplay begins
    print("\n[INIT] Running initial offline training...")
    trainer.initial_offline_training(num_steps=15)
    
    # Training loop - runs until Ctrl+C
    try:
        while not trainer.should_exit:
            trainer.run_training_cycle()
    
    except KeyboardInterrupt:
        print("\n\n[STOP] Training interrupted by user (Ctrl+C)")
        trainer.save_checkpoint_and_buffer()
    
    finally:
        trainer.cleanup()
        print("\n[COMPLETE] Training finished!")


if __name__ == "__main__":
    main()
