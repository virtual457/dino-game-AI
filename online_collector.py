import os
import sys
import argparse
import pickle
import time
import numpy as np
import torch
from collections import deque

from config import *
from game_env import DinoGameEnv
from agent import DDQNAgent
from rewards import calculate_reward
from episode_visualizer import EpisodeVisualizer


class OnlineCollector:
    
    def __init__(self, num_episodes=10, max_frames_per_episode=10000, 
                 epsilon=0.1, visualize=True):
        self.num_episodes = num_episodes
        self.max_frames_per_episode = max_frames_per_episode
        self.epsilon = epsilon
        self.visualize = visualize
        
        self.env = None
        self.agent = None
        self.visualizer = EpisodeVisualizer() if visualize else None
        
        self.checkpoint_path = CHECKPOINT_PATH
        self.buffer_path = os.path.join(CHECKPOINT_DIR, "replay_buffer.pkl")
        
        self.episode_stats = []
        self.total_frames_collected = 0
        
    def load_agent(self):
        print("\n" + "="*80)
        print("LOADING AGENT")
        print("="*80)
        
        self.agent = DDQNAgent(
            state_shape=STATE_SHAPE,
            n_actions=N_ACTIONS,
            learning_rate=LEARNING_RATE,
            gamma=GAMMA,
            epsilon_start=self.epsilon,
            epsilon_end=self.epsilon,
            epsilon_decay_rate=1.0,
            batch_size=BATCH_SIZE,
            buffer_capacity=BUFFER_CAPACITY,
            target_update_freq=TARGET_UPDATE_FREQ
        )
        
        self.agent.epsilon = self.epsilon
        
        if os.path.exists(self.checkpoint_path):
            print(f"\n[LOAD] Loading model from: {self.checkpoint_path}")
            try:
                self.agent.load(self.checkpoint_path)
                print("[OK] Model loaded successfully!")
            except Exception as e:
                print(f"[WARN] Error loading model: {e}")
                print("[INFO] Using random policy")
        else:
            print(f"[WARN] No checkpoint found at {self.checkpoint_path}")
            print("[INFO] Using random policy for exploration")
        
        if os.path.exists(self.buffer_path):
            print(f"\n[LOAD] Loading existing buffer from: {self.buffer_path}")
            try:
                self.load_buffer()
                print(f"[OK] Buffer loaded! Size: {len(self.agent.replay_buffer):,}")
            except Exception as e:
                print(f"[WARN] Could not load buffer: {e}")
                print("[INFO] Starting with empty buffer")
        else:
            print(f"\n[INFO] No existing buffer found")
            print(f"[INFO] Starting with empty buffer")
        
        print(f"\nCollection configuration:")
        print(f"  Epsilon (exploration): {self.epsilon:.3f}")
        print(f"  Buffer capacity: {BUFFER_CAPACITY:,}")
        print(f"  Current buffer size: {len(self.agent.replay_buffer):,}")
    
    def load_buffer(self):
        with open(self.buffer_path, 'rb') as f:
            buffer_data = pickle.load(f)
        
        self.agent.replay_buffer.buffer = deque(
            buffer_data['buffer'],
            maxlen=self.agent.replay_buffer.buffer.maxlen
        )
    
    def save_buffer(self):
        print(f"\n[SAVE] Saving buffer to {self.buffer_path}...")
        try:
            buffer_data = {
                'buffer': list(self.agent.replay_buffer.buffer),
                'size': len(self.agent.replay_buffer)
            }
            with open(self.buffer_path, 'wb') as f:
                pickle.dump(buffer_data, f)
            print(f"[OK] Buffer saved! Size: {len(self.agent.replay_buffer):,}")
        except Exception as e:
            print(f"[ERROR] Failed to save buffer: {e}")
    
    def open_game(self):
        print("\n" + "="*80)
        print("INITIALIZING GAME ENVIRONMENT")
        print("="*80)
        
        self.env = DinoGameEnv()
        
        print(f"\n[GAME] Opening Chrome Dino game...")
        if not self.env.open_game():
            print("[ERROR] Failed to open game!")
            return False
        
        print("[OK] Game opened successfully!")
        print(f"\nGame configuration:")
        print(f"  Capture region: {CAPTURE_WIDTH}×{CAPTURE_HEIGHT}")
        print(f"  Frame rate: {TARGET_FPS} FPS")
        print(f"  Frame stack: {N_FRAME_STACK} frames")
        
        return True
    
    def collect_episode(self, episode_num):
        print(f"\n{'='*80}")
        print(f"EPISODE {episode_num}/{self.num_episodes}")
        print(f"{'='*80}")
        print(f"Buffer: {len(self.agent.replay_buffer):,}/{BUFFER_CAPACITY:,} "
              f"({len(self.agent.replay_buffer)/BUFFER_CAPACITY*100:.1f}% full)")
        print(f"Epsilon: {self.epsilon:.3f}")
        print("="*80 + "\n")
        
        self.env.reset_buffers()
        for i in range(N_FRAME_STACK):
            raw_frame = self.env.capture_frame()
            preprocessed_frame, gray_frame, _ = self.env.preprocess_frame(raw_frame)
            self.env.add_frame_to_stack(preprocessed_frame)
            self.env.update_crash_detection_buffer(gray_frame)
        
        frame_count = 0
        episode_reward = 0.0
        episode_experiences = []
        action_counts = {0: 0, 1: 0, 2: 0}
        start_time = time.time()
        
        while frame_count < self.max_frames_per_episode:
            state = self.env.get_stacked_state()
            action = self.agent.select_action(state, training=False)
            action_counts[action] += 1
            
            self.env.execute_action(action)
            
            time.sleep(FRAME_TIME)
            
            raw_frame = self.env.capture_frame()
            preprocessed_frame, gray_frame, resized_frame = self.env.preprocess_frame(raw_frame)
            self.env.add_frame_to_stack(preprocessed_frame)
            next_state = self.env.get_stacked_state()
            
            game_over = self.env.is_game_over(gray_frame)
            self.env.update_crash_detection_buffer(gray_frame)
            
            if game_over:
                reward = REWARD_DEATH
            else:
                reward = calculate_reward(action, game_over=False)
            
            episode_experiences.append({
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state,
                'done': game_over,
                'frame_num': frame_count,
                'timestamp': frame_count / (TARGET_FPS if TARGET_FPS else 10)
            })
            
            frame_count += 1
            episode_reward += reward
            self.total_frames_collected += 1
            
            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                print(f"Frame {frame_count:>5} | "
                      f"Reward: {episode_reward:>8.2f} | "
                      f"FPS: {fps:>5.1f} | "
                      f"Buffer: {len(self.agent.replay_buffer):>6,}", end='\r')
            
            if game_over:
                break
        
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        
        print(f"\n\n[EPISODE COMPLETE]")
        print(f"  Frames collected: {frame_count}")
        print(f"  Raw reward: {episode_reward:.2f}")
        
        experiences_before_cleaning = len(episode_experiences)
        clean_experiences = episode_experiences
        experiences_removed = 0
        
        if len(episode_experiences) >= 3 and game_over:
            clean_experiences = episode_experiences[:-3]
            
            if len(clean_experiences) > 0:
                num_penalty_frames = min(8, len(clean_experiences))
                
                for i in range(num_penalty_frames):
                    idx = -(i + 1)
                    clean_experiences[idx]['reward'] = -10.0
                    if i == 0:
                        clean_experiences[idx]['done'] = True
                
                experiences_removed = experiences_before_cleaning - len(clean_experiences)
                
                frame_count = len(clean_experiences)
                episode_reward = sum(exp['reward'] for exp in clean_experiences)
                
                print(f"  Frozen frames removed: {experiences_removed}")
                print(f"  Penalty frames: {num_penalty_frames} (all -10.0)")
                print(f"  Clean frames to store: {frame_count}")
                print(f"  Adjusted reward: {episode_reward:.2f}")
            else:
                print(f"  WARNING: Episode too short after removing frozen frames!")
                clean_experiences = []
        else:
            print(f"  All frames valid: {frame_count} (no crash or too short)")
        
        for exp in clean_experiences:
            self.agent.store_transition(
                exp['state'],
                exp['action'],
                exp['reward'],
                exp['next_state'],
                exp['done']
            )
        
        print(f"  Duration: {elapsed:.1f}s")
        print(f"  FPS: {fps:.1f}")
        print(f"  Actions: Nothing={action_counts[0]}, Jump={action_counts[1]}, Duck={action_counts[2]}")
        print(f"  Buffer size: {len(self.agent.replay_buffer):,}/{BUFFER_CAPACITY:,}")
        
        if self.visualize and self.visualizer and len(clean_experiences) > 0:
            self.visualize_episode(episode_num, clean_experiences, action_counts)
        
        if episode_num < self.num_episodes:
            print(f"\n[RESTART] Restarting game for next episode...")
            time.sleep(CRASH_PAUSE_DURATION)
            self.env.restart_game()
        
        return {
            'episode_num': episode_num,
            'frames': frame_count,
            'reward': episode_reward,
            'duration': elapsed,
            'fps': fps,
            'actions': action_counts
        }
    
    def visualize_episode(self, episode_num, experiences, action_counts):
        if len(experiences) == 0:
            return
        
        episode_length = len(experiences)
        episode_reward = sum(exp['reward'] for exp in experiences)
        
        first_10 = experiences[:10] if episode_length >= 10 else experiences
        last_10 = experiences[-10:] if episode_length >= 10 else experiences
        
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
        
        crash_frame_idx = -1
        for i in range(len(last_10_display)-1, -1, -1):
            if last_10_display[i]['reward'] < 0:
                crash_frame_idx = i
                break
        
        total_actions = sum(action_counts.values())
        action_pcts = {k: (v/total_actions*100) if total_actions > 0 else 0 
                      for k, v in action_counts.items()}
        
        self.visualizer.show_episode_summary({
            'episode_num': episode_num,
            'steps': episode_length,
            'total_reward': episode_reward,
            'avg_reward_1000': episode_reward / episode_length if episode_length > 0 else 0,
            'avg_frame_time': 1000.0 / (TARGET_FPS if TARGET_FPS else 10),
            'avg_loss': 0.0,
            'epsilon': self.epsilon,
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
    
    def run(self):
        print("\n" + "="*80)
        print("STARTING ONLINE COLLECTION")
        print("="*80)
        print(f"\nCollection plan:")
        print(f"  Episodes: {self.num_episodes}")
        print(f"  Max frames per episode: {self.max_frames_per_episode:,}")
        print(f"  Epsilon: {self.epsilon:.3f}")
        print(f"  Visualization: {'Enabled' if self.visualize else 'Disabled'}")
        print("="*80 + "\n")
        
        start_time = time.time()
        
        try:
            for episode in range(1, self.num_episodes + 1):
                stats = self.collect_episode(episode)
                self.episode_stats.append(stats)
            
            print("\n\n" + "="*80)
            print("COLLECTION COMPLETE")
            print("="*80)
            
            total_time = time.time() - start_time
            total_frames = sum(s['frames'] for s in self.episode_stats)
            avg_frames = np.mean([s['frames'] for s in self.episode_stats])
            max_frames = max([s['frames'] for s in self.episode_stats])
            avg_reward = np.mean([s['reward'] for s in self.episode_stats])
            
            print(f"\nCollection statistics:")
            print(f"  Episodes completed: {len(self.episode_stats)}")
            print(f"  Total frames collected: {total_frames:,}")
            print(f"  Average frames per episode: {avg_frames:.1f}")
            print(f"  Best episode: {max_frames} frames")
            print(f"  Average reward: {avg_reward:.2f}")
            print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
            print(f"  Average FPS: {total_frames/total_time:.1f}")
            print(f"\nBuffer statistics:")
            print(f"  Final buffer size: {len(self.agent.replay_buffer):,}/{BUFFER_CAPACITY:,}")
            print(f"  Buffer fill: {len(self.agent.replay_buffer)/BUFFER_CAPACITY*100:.1f}%")
            print(f"  Frames added this session: {total_frames:,}")
            
        except KeyboardInterrupt:
            print("\n\n[INTERRUPTED] Collection interrupted by user (Ctrl+C)")
            print(f"[INFO] Completed {len(self.episode_stats)} episodes")
        
        finally:
            self.save_buffer()
            
            if self.visualizer:
                self.visualizer.close()


def main():
    parser = argparse.ArgumentParser(
        description='Online Collector - Collect gameplay experiences using trained model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--num-episodes', type=int, default=10,
                       help='Number of episodes to collect')
    parser.add_argument('--max-frames', type=int, default=10000,
                       help='Maximum frames per episode')
    parser.add_argument('--epsilon', type=float, default=0.1,
                       help='Exploration rate (0=greedy, 1=random)')
    parser.add_argument('--no-visualize', action='store_true',
                       help='Disable episode visualization')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("ONLINE COLLECTOR - Collect Gameplay Experiences")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Episodes: {args.num_episodes}")
    print(f"  Max frames per episode: {args.max_frames:,}")
    print(f"  Epsilon: {args.epsilon:.3f}")
    print(f"  Visualization: {'Disabled' if args.no_visualize else 'Enabled'}")
    print("="*80 + "\n")
    
    collector = OnlineCollector(
        num_episodes=args.num_episodes,
        max_frames_per_episode=args.max_frames,
        epsilon=args.epsilon,
        visualize=not args.no_visualize
    )
    
    collector.load_agent()
    
    if not collector.open_game():
        print("\n[ERROR] Failed to open game")
        sys.exit(1)
    
    print("\n[START] Beginning collection...")
    print("Press Ctrl+C at any time to stop and save buffer\n")
    
    collector.run()
    
    print("\n[COMPLETE] Collection finished!")
    print(f"[INFO] Buffer saved to: {collector.buffer_path}")
    print("\nNext steps:")
    print("  1. Train on collected data:")
    print("     python offline_trainer.py --max-steps 10000")
    print("  2. Collect more episodes:")
    print("     python online_collector.py --num-episodes 10")
    print("  3. Or run hybrid trainer for combined training:")
    print("     python hybrid_trainer.py")


if __name__ == "__main__":
    main()