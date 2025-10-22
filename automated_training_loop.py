import os
import sys
import signal
import time
from offline_trainer import OfflineTrainer
from online_collector import OnlineCollector


class AutomatedTrainingLoop:
    
    def __init__(self, target_mse=0.4, max_train_steps=2000, 
                 episodes_per_cycle=40, epsilon=0.1, visualize=True):
        self.target_mse = target_mse
        self.max_train_steps = max_train_steps
        self.episodes_per_cycle = episodes_per_cycle
        self.epsilon = epsilon
        self.visualize = visualize
        
        self.cycle_number = 0
        self.should_exit = False
        
        signal.signal(signal.SIGINT, self.signal_handler)
    
    def signal_handler(self, sig, frame):
        print("\n\n" + "="*80)
        print("[CTRL+C] Interrupt detected! Finishing current operation...")
        print("="*80)
        self.should_exit = True
    
    def run_offline_training(self):
        print("\n" + "#"*80)
        print(f"# CYCLE {self.cycle_number} - OFFLINE TRAINING PHASE")
        print("#"*80 + "\n")
        
        try:
            import importlib
            import config as cfg
            importlib.reload(cfg)
            if hasattr(cfg, 'TARGET_MSE_OFFLINE'):
                old_mse = self.target_mse
                self.target_mse = cfg.TARGET_MSE_OFFLINE
                if old_mse != self.target_mse:
                    print(f"[CONFIG] Target MSE loaded from config: {self.target_mse:.4f} (was {old_mse:.4f})")
                else:
                    print(f"[CONFIG] Target MSE: {self.target_mse:.4f}")
        except Exception as e:
            print(f"[CONFIG] Using current target MSE: {self.target_mse:.4f}")
        
        trainer = OfflineTrainer(
            target_mse=self.target_mse,
            max_steps=self.max_train_steps,
            batch_size=100,
            save_interval=100
        )
        
        trainer.load_agent()
        
        if not trainer.load_buffer():
            print("\n[ERROR] Cannot train without buffer")
            print("[INFO] Skipping to collection phase to build buffer...")
            return False
        
        print(f"\n[TRAIN] Training until MSE < {self.target_mse} (max {self.max_train_steps:,} steps)...")
        trainer.train()
        
        print(f"\n[OK] Offline training complete for cycle {self.cycle_number}")
        return True
    
    def run_online_collection(self):
        print("\n" + "#"*80)
        print(f"# CYCLE {self.cycle_number} - ONLINE COLLECTION PHASE")
        print("#"*80 + "\n")
        
        epsilon_to_use = self.epsilon
        buffer_path = os.path.join("checkpoints", "replay_buffer.pkl")
        model_path = os.path.join("checkpoints", "dino_ddqn_model.pth")
        
        try:
            if os.path.exists(buffer_path):
                import pickle
                with open(buffer_path, 'rb') as f:
                    buffer_data = pickle.load(f)
                buffer_size = buffer_data.get('size', 0)
            else:
                buffer_size = 0
            
            model_exists = os.path.exists(model_path)
            
            if buffer_size == 0 and not model_exists:
                epsilon_to_use = 1.0
                print(f"[INFO] Fresh start (no model, no buffer) - using epsilon={epsilon_to_use:.1f} for exploration")
            else:
                if buffer_size > 0:
                    print(f"[INFO] Buffer has {buffer_size:,} experiences - using epsilon={epsilon_to_use:.3f}")
                if model_exists:
                    print(f"[INFO] Model checkpoint exists - using epsilon={epsilon_to_use:.3f}")
        except:
            if not os.path.exists(model_path):
                epsilon_to_use = 1.0
                print(f"[INFO] No model found - using epsilon={epsilon_to_use:.1f} for exploration")
        
        collector = OnlineCollector(
            num_episodes=self.episodes_per_cycle,
            max_frames_per_episode=10000,
            epsilon=epsilon_to_use,
            visualize=self.visualize
        )
        
        collector.load_agent()
        
        if not collector.open_game():
            print("\n[ERROR] Failed to open game")
            return False
        
        print(f"\n[COLLECT] Collecting {self.episodes_per_cycle} episodes...")
        collector.run()
        
        print(f"\n[OK] Online collection complete for cycle {self.cycle_number}")
        return True
    
    def run_cycle(self):
        self.cycle_number += 1
        
        print("\n\n" + "="*80)
        print("="*80)
        print(f"  STARTING CYCLE {self.cycle_number}")
        print("="*80)
        print("="*80)
        
        if not self.should_exit:
            success = self.run_offline_training()
            if not success and self.cycle_number == 1:
                print("\n[INFO] First cycle with empty buffer - will collect first")
        
        if not self.should_exit:
            success = self.run_online_collection()
            if not success:
                print("\n[ERROR] Collection failed - stopping")
                return False
        
        print("\n\n" + "="*80)
        print(f"CYCLE {self.cycle_number} COMPLETE")
        print("="*80)
        print(f"\nNext cycle will:")
        print(f"  1. Train until MSE < {self.target_mse}")
        print(f"  2. Collect {self.episodes_per_cycle} episodes")
        print(f"  3. Repeat...")
        print("="*80 + "\n")
        
        return True
    
    def run(self):
        print("\n" + "="*80)
        print("AUTOMATED TRAINING LOOP")
        print("="*80)
        print(f"\nConfiguration:")
        print(f"  Target MSE: {self.target_mse}")
        print(f"  Max training steps per cycle: {self.max_train_steps:,}")
        print(f"  Episodes per collection: {self.episodes_per_cycle}")
        print(f"  Exploration epsilon: {self.epsilon}")
        print(f"\nStrategy:")
        print(f"  1. Train offline until MSE < {self.target_mse}")
        print(f"  2. Collect {self.episodes_per_cycle} episodes")
        print(f"  3. Repeat indefinitely")
        print(f"\nPress Ctrl+C anytime to stop gracefully")
        print("="*80 + "\n")
                
        try:
            while not self.should_exit:
                success = self.run_cycle()
                
                if not success:
                    print("\n[ERROR] Cycle failed - stopping loop")
                    break
                
                if self.should_exit:
                    break
                
                print("\n[PAUSE] Waiting 5 seconds before next cycle...")
                time.sleep(5)
        
        except KeyboardInterrupt:
            print("\n\n[INTERRUPTED] Training loop interrupted")
        
        finally:
            print("\n" + "="*80)
            print("AUTOMATED TRAINING LOOP ENDED")
            print("="*80)
            print(f"\nCompleted cycles: {self.cycle_number}")
            print(f"\nAll progress saved!")
            print(f"  Model: checkpoints/dino_ddqn_model.pth")
            print(f"  Buffer: checkpoints/replay_buffer.pkl")
            print("\nYou can resume by running this script again")
            print("="*80 + "\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Automated Training Loop - Alternate between training and collection',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--target-mse', type=float, default=0.4,
                       help='Target MSE for offline training')
    parser.add_argument('--max-train-steps', type=int, default=2000,
                       help='Maximum training steps per cycle')
    parser.add_argument('--episodes', type=int, default=40,
                       help='Episodes to collect per cycle')
    parser.add_argument('--epsilon', type=float, default=0.1,
                       help='Exploration rate for collection')
    parser.add_argument('--no-visualize', action='store_true',
                       help='Disable episode visualization for faster collection')
    
    args = parser.parse_args()
    
    loop = AutomatedTrainingLoop(
        target_mse=args.target_mse,
        max_train_steps=args.max_train_steps,
        episodes_per_cycle=args.episodes,
        epsilon=args.epsilon,
        visualize=not args.no_visualize
    )
    
    loop.run()


if __name__ == "__main__":
    main()