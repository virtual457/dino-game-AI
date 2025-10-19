"""
Apply Action Persistence Fix
=============================

This script safely applies the action persistence fix to your DQN training pipeline.
"""

import os
import shutil
from datetime import datetime

def backup_file(filepath):
    """Create timestamped backup"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{filepath}.backup_{timestamp}"
    shutil.copy2(filepath, backup_path)
    print(f"✅ Backed up: {filepath} → {backup_path}")
    return backup_path

def apply_fix():
    """Apply the action persistence fix"""
    
    print("\n" + "="*70)
    print("APPLYING ACTION PERSISTENCE FIX")
    print("="*70 + "\n")
    
    # Step 1: Backup original game_env.py
    print("Step 1: Creating backup...")
    if os.path.exists("game_env.py"):
        backup_file("game_env.py")
    else:
        print("⚠️  game_env.py not found!")
        return False
    
    # Step 2: Replace with new version
    print("\nStep 2: Applying new game_env.py...")
    if os.path.exists("game_env_v2.py"):
        shutil.copy2("game_env_v2.py", "game_env.py")
        print("✅ game_env.py updated with action persistence!")
    else:
        print("❌ game_env_v2.py not found!")
        return False
    
    # Step 3: Verify trainer.py uses the correct import
    print("\nStep 3: Verifying trainer.py...")
    with open("trainer.py", "r") as f:
        content = f.read()
        if "from game_env import DinoGameEnv" in content:
            print("✅ trainer.py imports correctly from game_env")
        else:
            print("⚠️  trainer.py import might need checking")
    
    # Step 4: Check FPS settings
    print("\nStep 4: Checking FPS configuration...")
    with open("config.py", "r") as f:
        for line in f:
            if "TARGET_FPS" in line and "=" in line:
                print(f"  config.py: {line.strip()}")
    
    with open("trainer.py", "r") as f:
        content = f.read()
        if "config.TARGET_FPS" in content or "TARGET_FPS" in content:
            print("  ✅ trainer.py uses config.TARGET_FPS")
        else:
            print("  ⚠️  trainer.py FPS usage unclear")
    
    print("\n" + "="*70)
    print("FIX APPLIED SUCCESSFULLY!")
    print("="*70)
    
    print("\n📋 What changed:")
    print("  • game_env.py now has action persistence")
    print("  • Previous action tracking added")
    print("  • Keys held until action changes")
    print("  • Jump: momentary press")
    print("  • Duck: hold until released")
    
    print("\n🎯 Expected improvements:")
    print("  • No more action flickering")
    print("  • Ducks can be held across multiple frames")
    print("  • Cleaner action execution")
    print("  • Better learning stability")
    
    print("\n▶️  Next steps:")
    print("  1. Run: python main.py")
    print("  2. Watch first few episodes")
    print("  3. Check episode visualizer for sustained duck actions")
    print("  4. Monitor if agent learns better timing")
    
    print("\n🔄 Rollback:")
    print("  If issues occur, restore from backup:")
    print("  Find the .backup_* file and rename to game_env.py")
    
    return True

if __name__ == "__main__":
    success = apply_fix()
    
    if success:
        print("\n✅ Ready to train with action persistence!\n")
    else:
        print("\n❌ Fix application failed. Check errors above.\n")
