import os
import sys
import time
import numpy as np
import cv2
from PIL import Image
from config import *
from game_env import DinoGameEnv
from agent import DDQNAgent


class GifCreator:
    def __init__(self, num_episodes=10, output_path="demo.gif", fps=10):
        self.num_episodes = num_episodes
        self.output_path = output_path
        self.fps = fps
        self.env = None
        self.agent = None
        self.best_episode_frames = []
        self.best_episode_length = 0
        self.best_episode_num = 0
        
    def load_agent(self):
        print("\n" + "="*80)
        print("LOADING AGENT")
        print("="*80)
        
        self.agent = DDQNAgent(
            state_shape=STATE_SHAPE,
            n_actions=N_ACTIONS,
            learning_rate=LEARNING_RATE,
            gamma=GAMMA,
            epsilon_start=0.0,
            epsilon_end=0.0,
            epsilon_decay_rate=1.0,
            batch_size=BATCH_SIZE,
            buffer_capacity=BUFFER_CAPACITY,
            target_update_freq=TARGET_UPDATE_FREQ
        )
        
        checkpoint_path = CHECKPOINT_PATH
        if os.path.exists(checkpoint_path):
            print(f"\n[LOAD] Loading model from: {checkpoint_path}")
            try:
                self.agent.load(checkpoint_path)
                print("[OK] Model loaded successfully!")
            except Exception as e:
                print(f"[ERROR] Failed to load model: {e}")
                return False
        else:
            print(f"\n[ERROR] No checkpoint found at: {checkpoint_path}")
            return False
        
        return True
    
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
        return True
    
    def play_episode(self, episode_num):
        print(f"\n{'='*80}")
        print(f"EPISODE {episode_num}/{self.num_episodes}")
        print(f"{'='*80}\n")
        
        self.env.reset_buffers()
        for i in range(N_FRAME_STACK):
            raw_frame = self.env.capture_frame()
            preprocessed_frame, gray_frame, _ = self.env.preprocess_frame(raw_frame)
            self.env.add_frame_to_stack(preprocessed_frame)
            self.env.update_crash_detection_buffer(gray_frame)
        
        frame_count = 0
        episode_frames = []
        max_frames = 10000
        
        while frame_count < max_frames:
            state = self.env.get_stacked_state()
            action = self.agent.select_action(state, training=False)
            
            self.env.execute_action(action)
            time.sleep(FRAME_TIME)
            
            raw_frame = self.env.capture_frame()
            preprocessed_frame, gray_frame, resized_frame = self.env.preprocess_frame(raw_frame)
            self.env.add_frame_to_stack(preprocessed_frame)
            
            episode_frames.append(raw_frame)
            
            game_over = self.env.is_game_over(gray_frame)
            self.env.update_crash_detection_buffer(gray_frame)
            
            frame_count += 1
            
            if frame_count % 100 == 0:
                print(f"Frame {frame_count:>5} | Frames captured: {len(episode_frames)}", end='\r')
            
            if game_over:
                break
        
        print(f"\n[EPISODE COMPLETE] Frames survived: {frame_count}")
        
        if frame_count > self.best_episode_length:
            self.best_episode_length = frame_count
            self.best_episode_frames = episode_frames
            self.best_episode_num = episode_num
            print(f"[NEW BEST] This is the longest episode so far!")
        
        if episode_num < self.num_episodes:
            print(f"\n[RESTART] Restarting game for next episode...")
            time.sleep(CRASH_PAUSE_DURATION)
            self.env.restart_game()
        
        return frame_count
    
    def create_gif(self):
        if len(self.best_episode_frames) == 0:
            print("\n[ERROR] No frames captured!")
            return False
        
        print("\n" + "="*80)
        print("CREATING GIF")
        print("="*80)
        print(f"\nBest episode: #{self.best_episode_num}")
        print(f"Total frames: {len(self.best_episode_frames)}")
        print(f"Processing frames...")
        
        pil_frames = []
        downsample_factor = max(1, len(self.best_episode_frames) // 300)
        
        for i, frame in enumerate(self.best_episode_frames[::downsample_factor]):
            if i % 10 == 0:
                print(f"Processing frame {i}/{len(self.best_episode_frames)//downsample_factor}...", end='\r')
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            height, width = frame_rgb.shape[:2]
            new_width = 800
            new_height = int(height * (new_width / width))
            frame_resized = cv2.resize(frame_rgb, (new_width, new_height))
            
            pil_frame = Image.fromarray(frame_resized)
            pil_frames.append(pil_frame)
        
        print(f"\n\n[SAVE] Saving GIF to: {self.output_path}")
        
        duration_ms = int(1000 / self.fps)
        pil_frames[0].save(
            self.output_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=False
        )
        
        file_size_mb = os.path.getsize(self.output_path) / (1024 * 1024)
        print(f"[OK] GIF created successfully!")
        print(f"  Frames in GIF: {len(pil_frames)}")
        print(f"  File size: {file_size_mb:.2f} MB")
        print(f"  Duration: {len(pil_frames) / self.fps:.1f} seconds")
        
        return True
    
    def run(self):
        print("\n" + "="*80)
        print("GIF CREATOR - Find Best Episode and Create Demo GIF")
        print("="*80)
        print(f"\nConfiguration:")
        print(f"  Episodes to evaluate: {self.num_episodes}")
        print(f"  Output file: {self.output_path}")
        print(f"  GIF FPS: {self.fps}")
        print("="*80 + "\n")
        
        episode_lengths = []
        
        try:
            for episode in range(1, self.num_episodes + 1):
                length = self.play_episode(episode)
                episode_lengths.append(length)
            
            print("\n\n" + "="*80)
            print("ALL EPISODES COMPLETE")
            print("="*80)
            
            print(f"\nEpisode lengths:")
            for i, length in enumerate(episode_lengths, 1):
                marker = " ← BEST" if i == self.best_episode_num else ""
                print(f"  Episode {i:>2}: {length:>5} frames{marker}")
            
            print(f"\nBest episode: #{self.best_episode_num} with {self.best_episode_length} frames")
            print(f"Average length: {np.mean(episode_lengths):.1f} frames")
            print(f"Median length: {np.median(episode_lengths):.1f} frames")
            
            if not self.create_gif():
                print("\n[ERROR] Failed to create GIF")
                return False
            
            print("\n" + "="*80)
            print("SUCCESS!")
            print("="*80)
            print(f"\nDemo GIF saved to: {self.output_path}")
            print(f"You can now add this to your README!")
            print("="*80 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n[INTERRUPTED] GIF creation interrupted by user (Ctrl+C)")
            if len(self.best_episode_frames) > 0:
                print(f"[INFO] Creating GIF from best episode so far...")
                self.create_gif()
        
        return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='GIF Creator - Record best Dino game episode',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of episodes to evaluate')
    parser.add_argument('--output', type=str, default='demo.gif',
                       help='Output GIF filename')
    parser.add_argument('--fps', type=int, default=10,
                       help='GIF frames per second')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("GIF CREATOR FOR DINO GAME")
    print("="*80)
    print(f"\nThis script will:")
    print(f"  1. Play {args.episodes} episodes with the trained model")
    print(f"  2. Find the longest survival episode")
    print(f"  3. Create a demo GIF from that episode")
    print(f"\nOutput: {args.output}")
    print("="*80 + "\n")
    
    creator = GifCreator(
        num_episodes=args.episodes,
        output_path=args.output,
        fps=args.fps
    )
    
    if not creator.load_agent():
        print("\n[ERROR] Failed to load agent")
        sys.exit(1)
    
    if not creator.open_game():
        print("\n[ERROR] Failed to open game")
        sys.exit(1)
    
    creator.run()


if __name__ == "__main__":
    main()
