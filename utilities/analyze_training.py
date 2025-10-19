"""
Training Analysis Tool

Analyzes saved model and identifies issues:
1. Q-value distribution analysis
2. Action preference patterns
3. Network weight analysis
4. Policy visualization
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

from config import *
from agent import DDQNAgent


class TrainingAnalyzer:
    """Analyzes trained DQN model for debugging"""
    
    def __init__(self):
        self.agent = None
    
    def load_agent(self, checkpoint_path):
        """Load trained agent"""
        if not os.path.exists(checkpoint_path):
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            return False
        
        print(f"\n{'='*70}")
        print("LOADING TRAINED MODEL")
        print(f"{'='*70}")
        
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
        
        try:
            self.agent.load(checkpoint_path)
            print(f"\n✅ Model loaded successfully!")
            print(f"   Total steps: {self.agent.total_steps:,}")
            print(f"   Epsilon: {self.agent.epsilon:.4f}")
            print(f"   Buffer size: {len(self.agent.replay_buffer):,}")
            return True
        except Exception as e:
            print(f"\n❌ Error loading model: {e}")
            return False
    
    def analyze_network_weights(self):
        """Analyze network weights for issues"""
        print(f"\n{'='*70}")
        print("NETWORK WEIGHT ANALYSIS")
        print(f"{'='*70}\n")
        
        for name, param in self.agent.policy_net.named_parameters():
            if param.requires_grad:
                data = param.data.cpu().numpy()
                print(f"{name}:")
                print(f"  Shape: {param.shape}")
                print(f"  Mean: {data.mean():.6f}")
                print(f"  Std: {data.std():.6f}")
                print(f"  Min: {data.min():.6f}")
                print(f"  Max: {data.max():.6f}")
                
                # Check for dead neurons (all zeros)
                if len(data.shape) > 1:
                    dead_neurons = np.all(np.abs(data) < 1e-6, axis=tuple(range(1, len(data.shape))))
                    dead_count = np.sum(dead_neurons)
                    if dead_count > 0:
                        print(f"  ⚠️  Dead neurons: {dead_count}/{data.shape[0]} ({dead_count/data.shape[0]*100:.1f}%)")
                
                # Check for exploding weights
                if np.abs(data).max() > 10:
                    print(f"  ⚠️  Large weights detected (max abs: {np.abs(data).max():.2f})")
                
                print()
    
    def test_q_values_on_random_states(self, n_samples=1000):
        """Test Q-values on random states"""
        print(f"\n{'='*70}")
        print(f"Q-VALUE DISTRIBUTION ANALYSIS ({n_samples} random states)")
        print(f"{'='*70}\n")
        
        q_values_all = []
        action_preferences = {0: 0, 1: 0, 2: 0}
        
        for i in range(n_samples):
            # Generate random state
            state = np.random.rand(4, 84, 252).astype(np.float32)
            
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.agent.device)
                q_values = self.agent.policy_net(state_tensor).cpu().numpy()[0]
                q_values_all.append(q_values)
                
                # Track which action has highest Q-value
                best_action = np.argmax(q_values)
                action_preferences[best_action] += 1
        
        q_values_all = np.array(q_values_all)
        
        print("Q-Value Statistics:")
        print(f"  Overall Mean: {q_values_all.mean():.4f}")
        print(f"  Overall Std: {q_values_all.std():.4f}")
        print(f"  Overall Min: {q_values_all.min():.4f}")
        print(f"  Overall Max: {q_values_all.max():.4f}")
        print()
        
        action_names = ['NOTHING', 'JUMP', 'DUCK']
        for i, name in enumerate(action_names):
            print(f"{name} Q-values:")
            print(f"  Mean: {q_values_all[:, i].mean():.4f}")
            print(f"  Std: {q_values_all[:, i].std():.4f}")
            print(f"  Min: {q_values_all[:, i].min():.4f}")
            print(f"  Max: {q_values_all[:, i].max():.4f}")
            print()
        
        print("Action Preferences on Random States:")
        for i, name in enumerate(action_names):
            pct = action_preferences[i] / n_samples * 100
            print(f"  {name}: {action_preferences[i]}/{n_samples} ({pct:.1f}%)")
        print()
        
        # Check for action collapse
        dominant_action = max(action_preferences, key=action_preferences.get)
        dominant_pct = action_preferences[dominant_action] / n_samples * 100
        if dominant_pct > 80:
            print(f"⚠️  WARNING: Action collapse detected!")
            print(f"    {action_names[dominant_action]} is preferred {dominant_pct:.1f}% of the time")
            print(f"    This suggests the network has converged to a degenerate policy")
        
        return q_values_all, action_preferences
    
    def visualize_q_distributions(self, q_values_all):
        """Visualize Q-value distributions"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        action_names = ['NOTHING', 'JUMP', 'DUCK']
        
        # 1. Q-value histograms per action
        ax = axes[0, 0]
        for i, name in enumerate(action_names):
            ax.hist(q_values_all[:, i], bins=50, alpha=0.5, label=name)
        ax.set_title('Q-Value Distributions per Action', fontweight='bold')
        ax.set_xlabel('Q-Value')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Max Q-value distribution
        ax = axes[0, 1]
        max_q = q_values_all.max(axis=1)
        ax.hist(max_q, bins=50, color='green', alpha=0.7)
        ax.set_title('Max Q-Value Distribution', fontweight='bold')
        ax.set_xlabel('Max Q-Value')
        ax.set_ylabel('Frequency')
        ax.axvline(max_q.mean(), color='red', linestyle='--', label=f'Mean: {max_q.mean():.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Q-value differences (JUMP vs NOTHING, DUCK vs NOTHING)
        ax = axes[1, 0]
        jump_diff = q_values_all[:, 1] - q_values_all[:, 0]
        duck_diff = q_values_all[:, 2] - q_values_all[:, 0]
        ax.hist(jump_diff, bins=50, alpha=0.5, label='JUMP - NOTHING', color='blue')
        ax.hist(duck_diff, bins=50, alpha=0.5, label='DUCK - NOTHING', color='red')
        ax.set_title('Q-Value Differences vs NOTHING', fontweight='bold')
        ax.set_xlabel('Q-Value Difference')
        ax.set_ylabel('Frequency')
        ax.axvline(0, color='black', linestyle='-', linewidth=2)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Action preference breakdown
        ax = axes[1, 1]
        best_actions = q_values_all.argmax(axis=1)
        action_counts = [np.sum(best_actions == i) for i in range(3)]
        bars = ax.bar(action_names, action_counts, color=['gray', 'blue', 'red'], alpha=0.7)
        ax.set_title('Action Preferences (Best Q-Value)', fontweight='bold')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3)
        
        # Add percentages on bars
        total = sum(action_counts)
        for bar, count in zip(bars, action_counts):
            height = bar.get_height()
            pct = (count / total * 100) if total > 0 else 0
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{count}\n({pct:.1f}%)',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        analysis_path = 'checkpoints/q_value_analysis.png'
        plt.savefig(analysis_path, dpi=150, bbox_inches='tight')
        print(f"✅ Q-value analysis saved: {analysis_path}")
        
        plt.show()
    
    def analyze_replay_buffer(self):
        """Analyze replay buffer contents"""
        print(f"\n{'='*70}")
        print("REPLAY BUFFER ANALYSIS")
        print(f"{'='*70}\n")
        
        if len(self.agent.replay_buffer) == 0:
            print("⚠️  Replay buffer is empty!")
            return
        
        print(f"Buffer Size: {len(self.agent.replay_buffer):,}/{BUFFER_CAPACITY:,}")
        print(f"Fill Percentage: {len(self.agent.replay_buffer)/BUFFER_CAPACITY*100:.1f}%")
        print()
        
        # Sample and analyze
        if len(self.agent.replay_buffer) >= 1000:
            states, actions, rewards, next_states, dones = self.agent.replay_buffer.sample(1000)
            
            print("Action Distribution in Buffer:")
            action_names = ['NOTHING', 'JUMP', 'DUCK']
            for i, name in enumerate(action_names):
                count = np.sum(actions == i)
                pct = count / len(actions) * 100
                print(f"  {name}: {count}/1000 ({pct:.1f}%)")
            print()
            
            print("Reward Distribution:")
            print(f"  Mean: {rewards.mean():.4f}")
            print(f"  Std: {rewards.std():.4f}")
            print(f"  Min: {rewards.min():.4f}")
            print(f"  Max: {rewards.max():.4f}")
            
            # Count specific reward values
            unique_rewards, counts = np.unique(rewards, return_counts=True)
            print(f"\n  Reward Value Breakdown:")
            for val, count in zip(unique_rewards, counts):
                pct = count / len(rewards) * 100
                print(f"    {val:+.2f}: {count}/1000 ({pct:.1f}%)")
            print()
            
            print("Terminal States:")
            terminal_count = np.sum(dones)
            print(f"  Done flags: {terminal_count}/1000 ({terminal_count/10:.1f}%)")
        else:
            print("⚠️  Buffer too small for detailed analysis (< 1000 experiences)")
    
    def generate_full_report(self, checkpoint_path=CHECKPOINT_PATH):
        """Generate comprehensive analysis report"""
        print("\n" + "="*70)
        print("DQN TRAINING ANALYSIS REPORT")
        print("="*70)
        
        # Load model
        if not self.load_agent(checkpoint_path):
            return
        
        # 1. Network weights
        self.analyze_network_weights()
        
        # 2. Replay buffer
        self.analyze_replay_buffer()
        
        # 3. Q-value analysis
        q_values_all, action_preferences = self.test_q_values_on_random_states(n_samples=1000)
        
        # 4. Visualizations
        self.visualize_q_distributions(q_values_all)
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)
        print("\nDiagnosis Summary:")
        print("-" * 70)
        
        # Check for common issues
        issues = []
        
        # Issue 1: Action collapse
        dominant_action = max(action_preferences, key=action_preferences.get)
        dominant_pct = action_preferences[dominant_action] / 1000 * 100
        if dominant_pct > 80:
            action_names = ['NOTHING', 'JUMP', 'DUCK']
            issues.append(f"❌ Action Collapse: {action_names[dominant_action]} preferred {dominant_pct:.1f}% of the time")
        
        # Issue 2: Q-values too high/low
        q_mean = q_values_all.mean()
        if abs(q_mean) > 100:
            issues.append(f"⚠️  Extreme Q-values: Mean Q = {q_mean:.2f}")
        
        # Issue 3: Low buffer fill
        buffer_fill = len(self.agent.replay_buffer) / BUFFER_CAPACITY
        if buffer_fill < 0.1:
            issues.append(f"⚠️  Low buffer utilization: {buffer_fill*100:.1f}% full")
        
        # Issue 4: High epsilon (still exploring heavily)
        if self.agent.epsilon > 0.5:
            issues.append(f"⚠️  High epsilon: {self.agent.epsilon:.4f} (still mostly random)")
        
        if issues:
            print("\nIssues Detected:")
            for issue in issues:
                print(f"  {issue}")
        else:
            print("\n✅ No obvious issues detected!")
        
        print("\n" + "="*70)


def main():
    """Main entry point"""
    analyzer = TrainingAnalyzer()
    analyzer.generate_full_report()


if __name__ == "__main__":
    main()
