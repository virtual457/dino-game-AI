import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


class EpisodeVisualizer:
    
    def __init__(self):
        self.action_names = {0: "NOTHING", 1: "JUMP", 2: "DUCK"}
        self.action_colors = {0: 'gray', 1: 'green', 2: 'orange'}
        self.save_dir = 'checkpoints/episode_images'
        os.makedirs(self.save_dir, exist_ok=True)
    
    def show_episode_summary(self, episode_stats):
        episode_num = episode_stats['episode_num']
        
        all_frames = episode_stats['last_10_frames']
        num_frames = len(all_frames)
        
        if num_frames < 10:
            print(f"⚠️  Episode too short ({num_frames} frames), showing all available frames")
            last_10 = all_frames
        else:
            last_10 = all_frames[-10:]
        
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(4, 5, figure=fig, hspace=0.15, wspace=0.2, 
                     left=0.05, right=0.95, top=0.92, bottom=0.05)
        
        crash_idx = episode_stats.get('crash_frame_idx', -1)
        
        for i in range(5):
            if i < len(last_10):
                frame_data = last_10[i]
                
                ax_frame = fig.add_subplot(gs[0, i])
                offset = len(last_10) - 1 - i
                self._draw_frame(ax_frame, frame_data, f"n-{offset}", is_crash=False)
                
                ax_stats = fig.add_subplot(gs[1, i])
                self._draw_frame_stats(ax_stats, frame_data, is_crash=False)
            else:
                ax_frame = fig.add_subplot(gs[0, i])
                ax_frame.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=16)
                ax_frame.axis('off')
                
                ax_stats = fig.add_subplot(gs[1, i])
                ax_stats.axis('off')
        
        for i in range(5):
            frame_idx = min(5 + i, len(last_10) - 1) if len(last_10) > 5 else i
            
            if frame_idx < len(last_10) and (5 + i) < len(last_10):
                frame_data = last_10[5 + i]
                is_crash = (5 + i == len(last_10) - 1)
                
                ax_frame = fig.add_subplot(gs[2, i])
                if is_crash:
                    label = "n (CRASH)"
                else:
                    offset = len(last_10) - 1 - (5 + i)
                    label = f"n-{offset}"
                self._draw_frame(ax_frame, frame_data, label, is_crash=is_crash)
                
                ax_stats = fig.add_subplot(gs[3, i])
                self._draw_frame_stats(ax_stats, frame_data, is_crash=is_crash)
            else:
                if frame_idx < len(last_10) and len(last_10) <= 5:
                    frame_data = last_10[frame_idx]
                    is_crash = (frame_idx == len(last_10) - 1)
                    
                    ax_frame = fig.add_subplot(gs[2, i])
                    label = "n (CRASH)" if is_crash else f"Frame {frame_data['frame_num']}"
                    self._draw_frame(ax_frame, frame_data, label, is_crash=is_crash)
                    
                    ax_stats = fig.add_subplot(gs[3, i])
                    self._draw_frame_stats(ax_stats, frame_data, is_crash=is_crash)
                else:
                    ax_frame = fig.add_subplot(gs[2, i])
                    ax_frame.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=16)
                    ax_frame.axis('off')
                    
                    ax_stats = fig.add_subplot(gs[3, i])
                    ax_stats.axis('off')
        
        fig.suptitle(f'Episode {episode_num} - {episode_stats["steps"]} Frames - Total Reward: {episode_stats["total_reward"]:.1f}', 
                     fontsize=16, fontweight='bold')
        
        latest_path = os.path.join(self.save_dir, "latest_episode.png")
        plt.savefig(latest_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"✅ Episode visualization saved: {latest_path}")
        
        plt.close(fig)
    
    def _draw_frame(self, ax, frame_data, title, is_crash=False):
        ax.imshow(frame_data['image'], cmap='gray', aspect='auto')
        
        if is_crash:
            border_color = 'red'
            border_width = 5
        else:
            border_color = 'green'
            border_width = 2
        
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(border_width)
        
        title_color = 'red' if is_crash else 'black'
        title_text = f"{title} 💥" if is_crash else title
        ax.set_title(title_text, fontsize=12, fontweight='bold', color=title_color)
        
        ax.axis('off')
    
    def _draw_frame_stats(self, ax, frame_data, is_crash=False):
        ax.axis('off')
        
        action_name = self.action_names.get(frame_data['action'], "???")
        action_color = self.action_colors.get(frame_data['action'], 'gray')
        
        reward = frame_data['reward']
        reward_color = 'green' if reward > 0 else 'red' if reward < 0 else 'gray'
        
        stats_text = f"""
ACTION: {action_name}

REWARD: {reward:+.1f}

TIME: {frame_data['timestamp']:.2f}s
        """
        
        if is_crash:
            bg_color = '#ffcccc'
        else:
            bg_color = 'white'
        
        ax.text(0.5, 0.5, stats_text, 
               transform=ax.transAxes,
               fontsize=11,
               ha='center', 
               va='center',
               fontfamily='monospace',
               fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.8', 
                        facecolor=bg_color, 
                        edgecolor='black',
                        linewidth=2))
    
    def close(self):
        plt.close('all')
        print("✅ Visualizer closed")


def test_visualizer():
    print("Testing Episode Visualizer...")
    
    def create_dummy_frame(frame_num, action, reward, timestamp):
        image = np.random.randint(0, 255, (84, 252), dtype=np.uint8)
        return {
            'frame_num': frame_num,
            'image': image,
            'action': action,
            'reward': reward,
            'timestamp': timestamp
        }
    
    episode_stats = {
        'episode_num': 42,
        'steps': 347,
        'total_reward': 24.7,
        'avg_reward_1000': 18.5,
        'avg_frame_time': 12.3,
        'avg_loss': 0.0234,
        'epsilon': 0.456,
        'buffer_size': 45000,
        'jumps': 89,
        'ducks': 23,
        'nothing': 235,
        'jump_pct': 25.6,
        'duck_pct': 6.6,
        'nothing_pct': 67.7,
        'movement_pct_500': 32.3,
        'experiences_added': 344,
        'experiences_discarded': 3,
        'first_10_frames': [create_dummy_frame(i, i % 3, 0.1, i*0.1) for i in range(10)],
        'last_10_frames': [create_dummy_frame(340+i, (i+1) % 3, -10.0 if i == 9 else 0.1, 34.0+i*0.1) for i in range(10)],
        'crash_frame_idx': 9
    }
    
    viz = EpisodeVisualizer()
    viz.show_episode_summary(episode_stats)
    viz.close()
    
    print("✅ Test complete! Check checkpoints/episode_images/latest_episode.png")


if __name__ == "__main__":
    test_visualizer()