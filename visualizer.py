import cv2
import numpy as np
from config import VIS_HEIGHT, VIS_WIDTH, FRAME_DISPLAY_SIZE


class TrainingVisualizer:
    
    def __init__(self):
        self.window_name = 'DQN Training - 3 Actions'
    
    def create_visualization(self, action_tracker, episode_stats, rolling_stats, 
                            current_penalty, step_count, total_reward, epsilon, 
                            buffer_size, avg_loss, game_over):
        visualization = np.zeros((VIS_HEIGHT, VIS_WIDTH, 3), dtype=np.uint8)
        
        cv2.putText(visualization, "LAST 3 JUMP FRAMES", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(visualization, "LAST 3 DUCK FRAMES", (10, 255), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(visualization, "LAST 3 NOTHING FRAMES", (10, 485), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)
        
        self._display_frames(visualization, action_tracker.last_jump_frames, 40, (0, 255, 0))
        self._display_frames(visualization, action_tracker.last_duck_frames, 270, (255, 255, 0))
        self._display_frames(visualization, action_tracker.last_nothing_frames, 500, (255, 100, 0))
        
        self._display_stats(visualization, episode_stats, rolling_stats, current_penalty,
                          step_count, total_reward, epsilon, buffer_size, avg_loss, game_over)
        
        return visualization
    
    def _display_frames(self, visualization, frames, y_offset, border_color):
        for i, frame in enumerate(frames[-3:]):
            x_pos = 10 + (i * (FRAME_DISPLAY_SIZE + 10))
            frame_display = cv2.resize(frame, (FRAME_DISPLAY_SIZE, FRAME_DISPLAY_SIZE))
            frame_3ch = cv2.cvtColor(frame_display, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(frame_3ch, (0, 0), (FRAME_DISPLAY_SIZE-1, FRAME_DISPLAY_SIZE-1), 
                         border_color, 3)
            visualization[y_offset:y_offset+FRAME_DISPLAY_SIZE, 
                         x_pos:x_pos+FRAME_DISPLAY_SIZE] = frame_3ch
    
    def _display_stats(self, visualization, episode_stats, rolling_stats, current_penalty,
                      step_count, total_reward, epsilon, buffer_size, avg_loss, game_over):
        stats_x = 520
        
        cv2.putText(visualization, f"Step: {step_count}", (stats_x, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(visualization, f"Reward: {total_reward:.1f}", (stats_x, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(visualization, f"Epsilon: {epsilon:.3f}", (stats_x, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(visualization, "Episode:", (stats_x, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(visualization, f"Jump: {episode_stats['jumps']} ({episode_stats['jump_pct']:.1f}%)", 
                   (stats_x, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(visualization, f"Duck: {episode_stats['ducks']} ({episode_stats['duck_pct']:.1f}%)", 
                   (stats_x, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(visualization, f"Nothing: ({episode_stats['nothing_pct']:.1f}%)", 
                   (stats_x, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 1)
        
        cv2.putText(visualization, "Last 500:", (stats_x, 280), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
        cv2.putText(visualization, f"Moves: {rolling_stats['movement_pct']:.1f}%", 
                   (stats_x, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(visualization, f"Size: {rolling_stats['history_size']}/500", 
                   (stats_x, 335), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        penalty_color = (0, 255, 255) if current_penalty == 0 else \
                       (0, 255, 0) if current_penalty < 1 else \
                       (0, 165, 255) if current_penalty < 3 else (0, 0, 255)
        cv2.putText(visualization, f"Penalty: -{current_penalty:.2f}", (stats_x, 370), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, penalty_color, 2)
        
        if buffer_size < 32:
            cv2.putText(visualization, f"Collecting: {buffer_size}/32", (stats_x, 420), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
        else:
            cv2.putText(visualization, "Training!", (stats_x, 420), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if avg_loss is not None:
                cv2.putText(visualization, f"Loss: {avg_loss:.4f}", (stats_x, 450), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if game_over:
            cv2.putText(visualization, "GAME OVER", (stats_x, 520), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            cv2.putText(visualization, "Running", (stats_x, 520), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    def show(self, visualization):
        cv2.imshow(self.window_name, visualization)
        return cv2.waitKey(1) & 0xFF
    
    def close(self):
        cv2.destroyAllWindows()