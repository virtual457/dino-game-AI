"""
Main Training Script - Chrome Dino DQN Agent
Redesigned for clarity, better logging, and control
"""

import os
import sys
import time
import torch
from datetime import datetime

from config import *
from trainer import Trainer


class TrainingSession:
    """Manages a complete training session with better logging and control"""
    
    def __init__(self):
        self.trainer = None
        self.session_start = None
        self.total_episodes = 0
        self.best_episode_frames = 0
        self.running = True
        self.pause_requested = False
    
    def setup(self):
        """Initialize training session"""
        self._print_banner()
        self._create_directories()
        self._initialize_trainer()
        self._verify_gpu()
        self._open_game()
        self._print_session_info()
    
    def _print_banner(self):
        """Print training session banner"""
        print("\n" + "="*80)
        print(" " * 25 + "🦖 CHROME DINO DQN AGENT 🦖")
        print("="*80)
        print()
        print("ARCHITECTURE:")
        print(f"  • Algorithm:       Double DQN (prevents Q-value overestimation)")
        print(f"  • State Input:     4 stacked frames @ {STATE_SHAPE[1]}x{STATE_SHAPE[2]} pixels")
        print(f"  • Action Space:    {N_ACTIONS} actions (Nothing, Jump, Duck)")
        print(f"  • Network:         CNN → Flatten → FC(512→256→{N_ACTIONS})")
        print()
        print("TRAINING CONFIGURATION:")
        print(f"  • Target FPS:      {TARGET_FPS:.1f} fps (60ms per frame - FASTER!)")
        print(f"  • Replay Buffer:   {BUFFER_CAPACITY:,} experiences")
        print(f"  • Batch Size:      {BATCH_SIZE}")
        print(f"  • Learning Rate:   {LEARNING_RATE}")
        print(f"  • Gamma:           {GAMMA}")
        print(f"  • Epsilon Decay:   {EPSILON_DECAY_RATE} (AGGRESSIVE - 10x faster)")
        print(f"  • Target Update:   Every {TARGET_UPDATE_FREQ:,} steps (AGGRESSIVE)")
        print()
        print("REWARDS:")
        print(f"  • Survival:        +{REWARD_ALIVE} per frame")
        print(f"  • Death:           {REWARD_DEATH} (ONLY last frame - trust TD learning!)")
        print(f"  • Shaping:         None - Markovian rewards for proper learning")
        print("="*80)
    
    def _create_directories(self):
        """Create necessary directories"""
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        print(f"\n✅ Checkpoint directory: {CHECKPOINT_DIR}")
    
    def _initialize_trainer(self):
        """Initialize trainer and agent"""
        print("\n📦 Initializing trainer and DDQN agent...")
        self.trainer = Trainer()
        self.trainer.initialize_agent()
        print("✅ Trainer initialized")
    
    def _verify_gpu(self):
        """Verify GPU availability"""
        print("\n🖥️  Hardware Check:")
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  ✅ GPU: {device_name}")
            print(f"  ✅ Memory: {memory_gb:.1f} GB")
            print(f"  ✅ Training will use CUDA acceleration")
        else:
            print(f"  ⚠️  GPU: Not available")
            print(f"  ⚠️  Training will use CPU (slower)")
    
    def _open_game(self):
        """Open Chrome Dino game"""
        print("\n🎮 Opening Chrome Dino game...")
        if not self.trainer.env.open_game():
            print("❌ Failed to open game!")
            sys.exit(1)
        print("✅ Game opened successfully")
        time.sleep(1)
    
    def _print_session_info(self):
        """Print session information and controls"""
        print("\n" + "="*80)
        print("TRAINING SESSION READY")
        print("="*80)
        
        # Agent state
        print(f"\n📊 Agent State:")
        print(f"  • Epsilon:         {self.trainer.agent.epsilon:.4f}")
        print(f"  • Total Steps:     {self.trainer.agent.total_steps:,}")
        print(f"  • Buffer Size:     {len(self.trainer.agent.replay_buffer):,}/{BUFFER_CAPACITY:,}")
        
        # Training settings
        print(f"\n⚙️  Training Settings:")
        print(f"  • Auto-save:       Every {SAVE_INTERVAL:,} steps")
        print(f"  • Episode vis:     Every 10 episodes")
        print(f"  • Target FPS:      {TARGET_FPS} (consistent timing)")
        
        # Controls
        print(f"\n🎮 Controls:")
        print(f"  • Ctrl+C:          Stop training and save")
        print(f"  • Close window:    Emergency stop (may not save)")
        
        print(f"\n⏱️  Starting training in 3 seconds...")
        print("="*80 + "\n")
        
        time.sleep(3)
        self.session_start = time.time()
    
    def run(self):
        """Main training loop"""
        try:
            # Initialize first episode
            self.trainer.reset_episode()
            
            # Training loop
            while self.running:
                # Run one training step
                game_over = self.trainer.run_training_step()
                
                # Auto-save check
                self.trainer.auto_save()
                
                # Handle episode end
                if game_over:
                    self.total_episodes += 1
                    self._update_best_score()
                    self.trainer.handle_episode_end()
        
        except KeyboardInterrupt:
            self._handle_stop()
        except Exception as e:
            self._handle_error(e)
        finally:
            self._cleanup()
    
    def _update_best_score(self):
        """Track best episode performance"""
        if self.trainer.step_count > self.best_episode_frames:
            self.best_episode_frames = self.trainer.step_count
            print(f"\n🏆 NEW BEST EPISODE! {self.best_episode_frames} frames")
    
    def _handle_stop(self):
        """Handle graceful shutdown"""
        print("\n\n" + "="*80)
        print("TRAINING STOPPED BY USER")
        print("="*80)
        
        # Save model
        print("\n💾 Saving model...")
        try:
            self.trainer.agent.save(CHECKPOINT_PATH)
            print(f"✅ Model saved: {CHECKPOINT_PATH}")
        except Exception as e:
            print(f"❌ Error saving model: {e}")
        
        # Print session summary
        self._print_summary()
    
    def _handle_error(self, error):
        """Handle unexpected errors"""
        print("\n\n" + "="*80)
        print("TRAINING STOPPED - ERROR")
        print("="*80)
        print(f"\n❌ Error: {error}")
        
        # Try to save
        print("\n💾 Attempting to save model...")
        try:
            self.trainer.agent.save(CHECKPOINT_PATH)
            print(f"✅ Model saved: {CHECKPOINT_PATH}")
        except Exception as save_error:
            print(f"❌ Could not save: {save_error}")
        
        # Print traceback
        import traceback
        print("\nTraceback:")
        traceback.print_exc()
    
    def _cleanup(self):
        """Cleanup resources"""
        print("\n🧹 Cleaning up...")
        if self.trainer:
            self.trainer.cleanup()
        print("✅ Cleanup complete")
    
    def _print_summary(self):
        """Print training session summary"""
        if self.session_start:
            duration = time.time() - self.session_start
            hours = int(duration // 3600)
            minutes = int((duration % 3600) // 60)
            seconds = int(duration % 60)
        else:
            hours = minutes = seconds = 0
        
        print("\n" + "="*80)
        print("SESSION SUMMARY")
        print("="*80)
        
        print(f"\n⏱️  Duration:")
        print(f"  {hours}h {minutes}m {seconds}s")
        
        print(f"\n📊 Training Stats:")
        print(f"  • Episodes:        {self.total_episodes}")
        print(f"  • Total Steps:     {self.trainer.agent.total_steps:,}")
        print(f"  • Best Episode:    {self.best_episode_frames} frames")
        print(f"  • Final Epsilon:   {self.trainer.agent.epsilon:.4f}")
        print(f"  • Buffer Fill:     {len(self.trainer.agent.replay_buffer):,}/{BUFFER_CAPACITY:,}")
        
        if self.total_episodes > 0:
            avg_steps = self.trainer.agent.total_steps / self.total_episodes
            print(f"  • Avg Episode:     {avg_steps:.1f} frames")
        
        print(f"\n💾 Saved Files:")
        print(f"  • Model:           {CHECKPOINT_PATH}")
        print(f"  • Training plot:   {PLOT_PATH}")
        
        print("\n" + "="*80)
        print("Training session complete! 🎉")
        print("="*80 + "\n")


def main():
    """Main entry point"""
    session = TrainingSession()
    session.setup()
    session.run()


if __name__ == "__main__":
    main()
