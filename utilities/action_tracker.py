"""
Action tracking and statistics
"""
from collections import deque
from config import ACTION_HISTORY_SIZE


class ActionTracker:
    """Tracks actions taken during training"""
    
    def __init__(self):
        # Episode-level tracking
        self.jump_count = 0
        self.duck_count = 0
        self.total_actions = 0
        
        # Rolling window for penalty calculation
        self.action_history = deque(maxlen=ACTION_HISTORY_SIZE)
        
        # Frame storage for visualization
        self.last_jump_frames = []
        self.last_duck_frames = []
        self.last_nothing_frames = []
    
    def record_action(self, action, frame):
        """
        Record an action and store frame for visualization
        
        Args:
            action: 0=nothing, 1=jump, 2=duck
            frame: Resized frame (84x84) to store
        """
        self.total_actions += 1
        self.action_history.append(action)
        
        if action == 1:  # Jump
            self.jump_count += 1
            self.last_jump_frames.append(frame.copy())
            if len(self.last_jump_frames) > 3:
                self.last_jump_frames.pop(0)
        elif action == 2:  # Duck
            self.duck_count += 1
            self.last_duck_frames.append(frame.copy())
            if len(self.last_duck_frames) > 3:
                self.last_duck_frames.pop(0)
        else:  # Nothing
            self.last_nothing_frames.append(frame.copy())
            if len(self.last_nothing_frames) > 3:
                self.last_nothing_frames.pop(0)
    
    def get_episode_stats(self):
        """Get statistics for current episode"""
        jump_pct = (self.jump_count / self.total_actions * 100) if self.total_actions > 0 else 0
        duck_pct = (self.duck_count / self.total_actions * 100) if self.total_actions > 0 else 0
        nothing_pct = 100 - jump_pct - duck_pct
        nothing_count = self.total_actions - self.jump_count - self.duck_count
        
        return {
            'total': self.total_actions,
            'jumps': self.jump_count,
            'ducks': self.duck_count,
            'nothing': nothing_count,
            'jump_pct': jump_pct,
            'duck_pct': duck_pct,
            'nothing_pct': nothing_pct
        }
    
    def get_rolling_stats(self):
        """Get statistics from rolling 500-action window"""
        if len(self.action_history) >= 100:
            movements = sum(1 for a in self.action_history if a in [1, 2])
            movement_pct = (movements / len(self.action_history)) * 100
        else:
            episode_stats = self.get_episode_stats()
            movement_pct = episode_stats['jump_pct'] + episode_stats['duck_pct']
        
        return {
            'history_size': len(self.action_history),
            'movement_pct': movement_pct
        }
    
    def reset_episode(self):
        """Reset episode-level counters (keep action_history!)"""
        self.jump_count = 0
        self.duck_count = 0
        self.total_actions = 0
        self.last_jump_frames = []
        self.last_duck_frames = []
        self.last_nothing_frames = []
        # action_history persists across episodes!
