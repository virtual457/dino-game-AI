"""
Episode tracking and plotting
Tracks rewards and slow frame percentage
"""
from collections import deque
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import EPISODE_HISTORY_SIZE, PLOT_PATH, PLOT_INTERVAL


class EpisodeTracker:
    """Tracks episode statistics and generates plots"""
    
    def __init__(self):
        self.episode_number = 0
        self.episode_rewards = deque(maxlen=EPISODE_HISTORY_SIZE)
        self.episode_rewards_history = []
        self.episode_avg_rewards = []
        
        # Track slow frames
        self.slow_frame_pcts = []
        self.slow_frame_pcts_avg = []
    
    def record_episode(self, total_reward, slow_frame_pct=0.0):
        """Record episode completion with slow frame percentage"""
        self.episode_number += 1
        self.episode_rewards.append(total_reward)
        self.episode_rewards_history.append(total_reward)
        
        # Calculate rolling average reward
        avg_reward = np.mean(self.episode_rewards)
        self.episode_avg_rewards.append(avg_reward)
        
        # Track slow frame percentage
        self.slow_frame_pcts.append(slow_frame_pct)
        
        # Calculate rolling average of slow frames (last 100 episodes)
        recent_slow = self.slow_frame_pcts[-min(100, len(self.slow_frame_pcts)):]
        avg_slow = np.mean(recent_slow) if recent_slow else 0.0
        self.slow_frame_pcts_avg.append(avg_slow)
        
        return avg_reward
    
    def should_plot(self):
        """Check if we should generate a plot"""
        return self.episode_number % PLOT_INTERVAL == 0 and len(self.episode_rewards_history) > 1
    
    def generate_plot(self):
        """Generate and save training progress plot with slow frames"""
        print("📊 Generating progress plot...")
        
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # TOP PLOT: Rewards
        ax1.plot(self.episode_rewards_history, alpha=0.3, color='blue', label='Episode Reward')
        ax1.plot(self.episode_avg_rewards, color='red', linewidth=2, 
                label=f'Avg Last {EPISODE_HISTORY_SIZE} Episodes')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.set_title(f'Training Progress - Episode {self.episode_number}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # BOTTOM PLOT: Slow Frame Percentage
        ax2.plot(self.slow_frame_pcts, alpha=0.3, color='orange', label='Slow Frame %')
        ax2.plot(self.slow_frame_pcts_avg, color='darkred', linewidth=2, 
                label='Avg Last 100 Episodes')
        ax2.axhline(y=10, color='green', linestyle='--', alpha=0.5, label='10% Target')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Slow Frame %')
        ax2.set_title('Performance: Frames Exceeding Target Time (100ms)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, max(100, max(self.slow_frame_pcts) if self.slow_frame_pcts else 100))
        
        # Save plot
        plt.tight_layout()
        plt.savefig(PLOT_PATH, dpi=100, bbox_inches='tight')
        plt.close(fig)
        print(f"✅ Plot updated: {PLOT_PATH}\n")
    
    def get_current_avg(self):
        """Get current rolling average reward"""
        return np.mean(self.episode_rewards) if len(self.episode_rewards) > 0 else 0.0
