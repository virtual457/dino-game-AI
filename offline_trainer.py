import os
import sys
import argparse
import pickle
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import deque

from config import *
from agent import DDQNAgent


class OfflineTrainer:
    
    def __init__(self, target_mse=0.1, max_steps=10000, batch_size=100, save_interval=1000):
        self.target_mse = target_mse
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.save_interval = save_interval
        
        self.total_training_steps = 0
        self.training_losses = []
        
        self.checkpoint_path = CHECKPOINT_PATH
        self.buffer_path = os.path.join(CHECKPOINT_DIR, "replay_buffer.pkl")
        
        self.agent = None
        
        self.states_tensor = None
        self.actions_tensor = None
        self.rewards_tensor = None
        self.next_states_tensor = None
        self.dones_tensor = None
        self.mid_point = None
        
    def load_agent(self):
        print("\n" + "="*80)
        print("LOADING AGENT")
        print("="*80)
        
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
        
        if os.path.exists(self.checkpoint_path):
            print(f"\n[LOAD] Loading model from: {self.checkpoint_path}")
            try:
                self.agent.load(self.checkpoint_path)
                print("[OK] Model loaded successfully!")
            except Exception as e:
                print(f"[WARN] Error loading model: {e}")
                print("[INFO] Starting with fresh model weights")
        else:
            print(f"[INFO] No checkpoint found at {self.checkpoint_path}")
            print("[INFO] Starting with fresh model weights")
        
        print(f"\nAgent configuration:")
        print(f"  Learning rate: {LEARNING_RATE}")
        print(f"  Gamma: {GAMMA}")
        print(f"  Batch size: {BATCH_SIZE}")
        print(f"  Buffer capacity: {BUFFER_CAPACITY:,}")
        
    def prepare_balanced_buffer(self):
        print(f"\n{'='*70}")
        print(f"PRE-CONVERTING BUFFER TO TENSORS (for speed)")
        print(f"{'='*70}\n")
        
        if len(self.agent.replay_buffer) < BATCH_SIZE:
            print(f"[SKIP] Buffer too small ({len(self.agent.replay_buffer)} < {BATCH_SIZE})")
            return False
        
        all_experiences = list(self.agent.replay_buffer.buffer)
        buffer_size = len(all_experiences)
        
        print(f"Buffer size: {buffer_size:,} experiences")
        print(f"Converting to tensors on CPU...")
        
        states, actions, rewards, next_states, dones = zip(*all_experiences)
        
        states_np = np.array(states, dtype=np.float32)
        actions_np = np.array(actions, dtype=np.int64)
        rewards_np = np.array(rewards, dtype=np.float32)
        next_states_np = np.array(next_states, dtype=np.float32)
        dones_np = np.array(dones, dtype=np.float32)
        
        self.states_tensor = torch.FloatTensor(states_np)
        self.actions_tensor = torch.LongTensor(actions_np)
        self.rewards_tensor = torch.FloatTensor(rewards_np)
        self.next_states_tensor = torch.FloatTensor(next_states_np)
        self.dones_tensor = torch.FloatTensor(dones_np)
        
        total_memory = (
            self.states_tensor.element_size() * self.states_tensor.nelement() +
            self.next_states_tensor.element_size() * self.next_states_tensor.nelement() +
            self.actions_tensor.element_size() * self.actions_tensor.nelement() +
            self.rewards_tensor.element_size() * self.rewards_tensor.nelement() +
            self.dones_tensor.element_size() * self.dones_tensor.nelement()
        )
        
        print(f"[OK] Tensors created!")
        print(f"  Memory used: {total_memory / 1024**3:.2f} GB")
        print(f"  States: {tuple(self.states_tensor.shape)}")
        print(f"  Actions: {tuple(self.actions_tensor.shape)}")
        print(f"  Rewards: {tuple(self.rewards_tensor.shape)}")
        
        print(f"\nSorting by reward...")
        sorted_indices = torch.argsort(self.rewards_tensor)
        
        self.states_tensor = self.states_tensor[sorted_indices]
        self.actions_tensor = self.actions_tensor[sorted_indices]
        self.rewards_tensor = self.rewards_tensor[sorted_indices]
        self.next_states_tensor = self.next_states_tensor[sorted_indices]
        self.dones_tensor = self.dones_tensor[sorted_indices]
        
        rewards_np = self.rewards_tensor.numpy()
        split_idx = np.searchsorted(rewards_np, 5.0)
        
        self.mid_point = split_idx
        
        negative_count = split_idx
        positive_count = buffer_size - split_idx
        
        print(f"[OK] Buffer sorted by reward!")
        print(f"\nSplit point found at index {split_idx}:")
        print(f"  Negative rewards (< +5): {negative_count:,} experiences (indices 0-{split_idx-1})")
        print(f"  Positive rewards (≥ +5): {positive_count:,} experiences (indices {split_idx}-{buffer_size-1})")
        
        if negative_count > 0:
            neg_rewards = rewards_np[:split_idx]
            print(f"\n  Negative section: min={neg_rewards.min():.1f}, max={neg_rewards.max():.1f}, avg={neg_rewards.mean():.2f}")
        else:
            print(f"\n  Negative section: EMPTY")
        
        if positive_count > 0:
            pos_rewards = rewards_np[split_idx:]
            print(f"  Positive section: min={pos_rewards.min():.1f}, max={pos_rewards.max():.1f}, avg={pos_rewards.mean():.2f}")
        else:
            print(f"  Positive section: EMPTY")
        
        print(f"\nDuring training, each 32-frame batch will sample:")
        print(f"  16 frames from negative section (crash penalties)")
        print(f"  16 frames from positive section (+5 alive)")
        print(f"  ⚡ Fast tensor indexing (~1ms vs ~80ms before!)")
        print(f"\n{'='*70}\n")
        
        return True
    
    def sample_balanced_batch(self, batch_size=32):
        import random
        
        n_from_negative = 16
        n_from_positive = 16
        
        n_from_negative = min(n_from_negative, self.mid_point)
        n_from_positive = min(n_from_positive, len(self.rewards_tensor) - self.mid_point)
        
        if n_from_negative > 0:
            negative_indices = random.sample(range(self.mid_point), n_from_negative)
        else:
            negative_indices = []
        
        if n_from_positive > 0:
            positive_indices = random.sample(range(self.mid_point, len(self.rewards_tensor)), n_from_positive)
        else:
            positive_indices = []
        
        all_indices = negative_indices + positive_indices
        random.shuffle(all_indices)
        indices_tensor = torch.LongTensor(all_indices)
        
        states = self.states_tensor[indices_tensor].to(self.agent.device)
        actions = self.actions_tensor[indices_tensor].to(self.agent.device)
        rewards = self.rewards_tensor[indices_tensor].to(self.agent.device)
        next_states = self.next_states_tensor[indices_tensor].to(self.agent.device)
        dones = self.dones_tensor[indices_tensor].to(self.agent.device)
        
        reward_values = self.rewards_tensor[indices_tensor].numpy()
        
        return states, actions, rewards, next_states, dones, reward_values
    
    def load_buffer(self):
        print("\n" + "="*80)
        print("LOADING REPLAY BUFFER")
        print("="*80)
        
        if not os.path.exists(self.buffer_path):
            print(f"\n[ERROR] Buffer not found at: {self.buffer_path}")
            print("[ERROR] Cannot train without buffer!")
            print("\nPlease run the online collector first to generate experiences:")
            print("  python online_collector.py --num-episodes 10")
            return False
        
        print(f"\n[LOAD] Loading buffer from: {self.buffer_path}")
        try:
            with open(self.buffer_path, 'rb') as f:
                buffer_data = pickle.load(f)
            
            self.agent.replay_buffer.buffer = deque(
                buffer_data['buffer'], 
                maxlen=self.agent.replay_buffer.buffer.maxlen
            )
            
            buffer_size = len(self.agent.replay_buffer)
            print(f"[OK] Buffer loaded successfully!")
            print(f"\nBuffer statistics:")
            print(f"  Size: {buffer_size:,} experiences")
            print(f"  Capacity: {BUFFER_CAPACITY:,}")
            print(f"  Fill: {buffer_size/BUFFER_CAPACITY*100:.1f}%")
            
            if buffer_size < BATCH_SIZE:
                print(f"\n[ERROR] Buffer too small! ({buffer_size} < {BATCH_SIZE})")
                print(f"[ERROR] Need at least {BATCH_SIZE} experiences to train")
                return False
            
            if not self.prepare_balanced_buffer():
                print(f"\n[ERROR] Failed to prepare balanced buffer")
                return False
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to load buffer: {e}")
            return False
    
    def train(self):
        print("\n" + "="*80)
        print("OFFLINE TRAINING")
        print("="*80)
        print(f"\nTraining configuration:")
        print(f"  Target MSE: {self.target_mse}")
        print(f"  Max steps: {self.max_steps:,}")
        print(f"  Batch size: {self.batch_size} steps")
        print(f"  Save interval: {self.save_interval:,} steps")
        print(f"\nBuffer: {len(self.agent.replay_buffer):,}/{BUFFER_CAPACITY:,} experiences")
        print("="*80 + "\n")
        
        training_step = 0
        recent_losses = []
        recent_rewards = []
        start_time = time.time()
        
        try:
            while training_step < self.max_steps:
                batch_losses = []
                batch_rewards_list = []
                
                for _ in range(self.batch_size):
                    states, actions, rewards, next_states, dones, reward_values = self.sample_balanced_batch(32)
                    
                    batch_rewards_list.extend(reward_values)
                    
                    current_q = self.agent.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
                    
                    with torch.no_grad():
                        next_actions = self.agent.policy_net(next_states).argmax(dim=1)
                        next_q = self.agent.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
                        target_q = rewards + (1 - dones) * self.agent.gamma * next_q
                    
                    loss = nn.MSELoss()(current_q, target_q)
                    
                    self.agent.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.agent.policy_net.parameters(), 10)
                    self.agent.optimizer.step()
                    batch_losses.append(loss.item())
                    recent_losses.append(loss.item())
                    self.training_losses.append(loss.item())
                    recent_rewards.extend(reward_values)
                    
                    training_step += 1
                    self.total_training_steps += 1
                    self.agent.learn_step_counter += 1
                    
                    if self.agent.learn_step_counter % self.agent.target_update_freq == 0:
                        self.agent.target_net.load_state_dict(self.agent.policy_net.state_dict())
                        print(f"\n[TARGET] Network updated at step {self.agent.learn_step_counter}")
                    
                    if training_step >= self.max_steps:
                        break
                
                if training_step % 100 == 0 and training_step > 0:
                    self.plot_training_progress()
                
                if len(recent_losses) > 1000:
                    recent_losses = recent_losses[-1000:]
                if len(recent_rewards) > 1000:
                    recent_rewards = recent_rewards[-1000:]
                
                if len(batch_losses) > 0:
                    avg_batch_loss = np.mean(batch_losses)
                    avg_recent_loss = np.mean(recent_losses[-100:]) if len(recent_losses) >= 100 else np.mean(recent_losses)
                    
                    if batch_rewards_list:
                        last_batch = batch_rewards_list[-32:]
                        n_positive = sum(1 for r in last_batch if r > 0)
                        n_negative = sum(1 for r in last_batch if r < 0)
                        
                        reward_str = f"Pos:{n_positive} Neg:{n_negative}"
                    else:
                        reward_str = "N/A"
                    
                    elapsed = time.time() - start_time
                    steps_per_sec = training_step / elapsed if elapsed > 0 else 0
                    
                    print(f"Step {training_step:>6,}/{self.max_steps:,} | "
                          f"MSE: {avg_recent_loss:.4f} | "
                          f"Speed: {steps_per_sec:.1f}/s | "
                          f"Rewards[{reward_str}]")
                    
                    if training_step % 1000 == 0:
                        self.save_checkpoint()
                        
                        try:
                            import importlib
                            import config as cfg
                            importlib.reload(cfg)
                            if hasattr(cfg, 'TARGET_MSE_OFFLINE'):
                                old_mse = self.target_mse
                                self.target_mse = cfg.TARGET_MSE_OFFLINE
                                if old_mse != self.target_mse:
                                    print(f"\n[CONFIG] Target MSE updated: {old_mse:.4f} → {self.target_mse:.4f}")
                        except Exception as e:
                            pass
                    
                    if avg_recent_loss < self.target_mse and len(recent_losses) >= 100:
                        print(f"\n\n[CONVERGED] Reached target MSE {self.target_mse:.4f}!")
                        print(f"[INFO] Final MSE: {avg_recent_loss:.4f}")
                        print(f"[INFO] Total steps: {training_step:,}")
                        
                        print(f"\n[TARGET] Copying policy network → target network...")
                        self.agent.target_net.load_state_dict(self.agent.policy_net.state_dict())
                        print(f"[OK] Target network updated with converged weights!")
                        
                        break
            
            print("\n\n" + "="*80)
            if training_step >= self.max_steps:
                print(f"[MAX STEPS] Completed {self.max_steps:,} training steps")
            else:
                print(f"[COMPLETE] Training converged after {training_step:,} steps")
            
            if len(recent_losses) >= 100:
                final_mse = np.mean(recent_losses[-100:])
                print(f"[INFO] Final MSE (last 100 steps): {final_mse:.4f}")
            
            elapsed = time.time() - start_time
            print(f"[INFO] Training time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
            print(f"[INFO] Average speed: {training_step/elapsed:.1f} steps/second")
            print("="*80 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n[INTERRUPTED] Training interrupted by user (Ctrl+C)")
            print(f"[INFO] Completed {training_step:,} steps")
        
        finally:
            print("\n[SAVE] Saving final checkpoint...")
            self.save_checkpoint()
            self.plot_training_progress()
            
            if self.agent.device.type == 'cuda':
                torch.cuda.empty_cache()
    
    def save_checkpoint(self):
        try:
            self.agent.save(self.checkpoint_path)
            print(f"\n[OK] Checkpoint saved: {self.checkpoint_path}")
        except Exception as e:
            print(f"\n[ERROR] Failed to save checkpoint: {e}")
    
    def plot_training_progress(self):
        if len(self.training_losses) < 10:
            return
        
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            steps = list(range(1, len(self.training_losses) + 1))
            ax.plot(steps, self.training_losses, alpha=0.3, color='blue', label='MSE Loss')
            
            window_size = min(100, len(self.training_losses) // 10)
            if window_size > 1:
                moving_avg = []
                for i in range(len(self.training_losses)):
                    start_idx = max(0, i - window_size + 1)
                    moving_avg.append(np.mean(self.training_losses[start_idx:i+1]))
                ax.plot(steps, moving_avg, color='red', linewidth=2, 
                       label=f'Moving Average (window={window_size})')
            
            ax.axhline(y=self.target_mse, color='green', linestyle='--', 
                      linewidth=2, label=f'Target MSE = {self.target_mse}')
            
            ax.set_xlabel('Training Step', fontsize=12)
            ax.set_ylabel('MSE Loss', fontsize=12)
            ax.set_title('Offline Training Progress', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
            
            if len(self.training_losses) >= 100:
                recent_mse = np.mean(self.training_losses[-100:])
                stats_text = f"Steps: {len(self.training_losses):,}\n"
                stats_text += f"Recent MSE (last 100): {recent_mse:.4f}\n"
                stats_text += f"Target MSE: {self.target_mse:.4f}"
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                       fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plot_path = os.path.join(CHECKPOINT_DIR, "offline_training_progress.png")
            plt.tight_layout()
            plt.savefig(plot_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            print(f"[OK] Training plot saved: {plot_path}")
            
        except Exception as e:
            print(f"[WARN] Failed to save plot: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Offline Trainer - Train DQN model from replay buffer',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--target-mse', type=float, default=1.0,
                       help='Target MSE to achieve before stopping')
    parser.add_argument('--max-steps', type=int, default=10000,
                       help='Maximum training steps')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Training steps per batch')
    parser.add_argument('--save-every', type=int, default=1000,
                       help='Save checkpoint every N steps')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("OFFLINE TRAINER - Train from Replay Buffer")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Target MSE: {args.target_mse}")
    print(f"  Max steps: {args.max_steps:,}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Save interval: {args.save_every:,}")
    print("="*80 + "\n")
    
    trainer = OfflineTrainer(
        target_mse=args.target_mse,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        save_interval=args.save_every
    )
    
    trainer.load_agent()
    
    if not trainer.load_buffer():
        print("\n[ERROR] Cannot proceed without replay buffer")
        print("\nNext steps:")
        print("  1. Run online collector to generate experiences:")
        print("     python online_collector.py --num-episodes 10")
        print("  2. Then run this offline trainer again")
        sys.exit(1)
    
    print("\n[START] Beginning offline training...")
    print("Press Ctrl+C at any time to stop and save progress\n")
    
    trainer.train()
    
    print("\n[COMPLETE] Offline training finished!")
    print(f"[INFO] Model saved to: {CHECKPOINT_PATH}")
    print(f"[INFO] Buffer remains at: {trainer.buffer_path}")
    print("\nNext steps:")
    print("  1. Run online collector to gather more experiences:")
    print("     python online_collector.py --num-episodes 10")
    print("  2. Run offline trainer again to continue learning:")
    print("     python offline_trainer.py --max-steps 10000")
    print("  3. Or run hybrid trainer for combined training:")
    print("     python hybrid_trainer.py")


if __name__ == "__main__":
    main()