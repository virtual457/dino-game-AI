"""
System Diagnostics - Check everything before debugging

Verifies:
1. GPU availability
2. File structure
3. Checkpoint integrity
4. Configuration settings
5. Dependencies
"""

import os
import sys
import torch
import numpy as np
import cv2
import matplotlib

print("="*70)
print("SYSTEM DIAGNOSTICS - DQN Dino Game")
print("="*70)

# 1. Python version
print(f"\n1. Python Version")
print(f"   {sys.version}")

# 2. PyTorch and CUDA
print(f"\n2. PyTorch & CUDA")
print(f"   PyTorch version: {torch.__version__}")
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   CUDA version: {torch.version.cuda}")
    print(f"   GPU device: {torch.cuda.get_device_name(0)}")
    print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("   ⚠️  CUDA not available - will use CPU (slower training)")

# 3. Other dependencies
print(f"\n3. Dependencies")
print(f"   NumPy: {np.__version__}")
print(f"   OpenCV: {cv2.__version__}")
print(f"   Matplotlib: {matplotlib.__version__}")

# 4. Project structure
print(f"\n4. Project Structure")
project_path = "D:\\Git\\virtual457-projects\\dino-game"
required_files = [
    "main.py",
    "trainer.py",
    "agent.py",
    "game_env.py",
    "model.py",
    "config.py",
    "rewards.py",
    "replay_buffer.py",
    "episode_tracker.py",
    "episode_visualizer.py"
]

all_exist = True
for file in required_files:
    path = os.path.join(project_path, file)
    exists = os.path.exists(path)
    status = "✅" if exists else "❌"
    print(f"   {status} {file}")
    if not exists:
        all_exist = False

if not all_exist:
    print("\n   ⚠️  Some required files are missing!")

# 5. Checkpoint directory
print(f"\n5. Checkpoint Directory")
checkpoint_dir = os.path.join(project_path, "checkpoints")
if os.path.exists(checkpoint_dir):
    print(f"   ✅ Checkpoint directory exists")
    
    # List checkpoint files
    files = os.listdir(checkpoint_dir)
    if files:
        print(f"   Files found:")
        for f in files:
            path = os.path.join(checkpoint_dir, f)
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"     - {f} ({size_mb:.2f} MB)")
    else:
        print(f"   ⚠️  No checkpoint files found")
else:
    print(f"   ❌ Checkpoint directory does not exist")

# 6. Model checkpoint
print(f"\n6. Model Checkpoint")
checkpoint_path = os.path.join(project_path, "checkpoints", "dino_ddqn_model.pth")
if os.path.exists(checkpoint_path):
    print(f"   ✅ Model checkpoint exists")
    
    # Try loading
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"   Checkpoint contents:")
        for key in checkpoint.keys():
            print(f"     - {key}")
        
        if 'epsilon' in checkpoint:
            print(f"\n   Training state:")
            print(f"     Epsilon: {checkpoint['epsilon']:.4f}")
            print(f"     Total steps: {checkpoint.get('total_steps', 'N/A')}")
            print(f"     Learn steps: {checkpoint.get('learn_step_counter', 'N/A')}")
        
        print(f"   ✅ Checkpoint loaded successfully")
    except Exception as e:
        print(f"   ❌ Error loading checkpoint: {e}")
else:
    print(f"   ❌ No checkpoint found - model not trained yet")

# 7. Configuration
print(f"\n7. Configuration Settings")
try:
    from config import *
    print(f"   Frame stacking: {N_FRAME_STACK} frames")
    print(f"   State shape: {STATE_SHAPE}")
    print(f"   Actions: {N_ACTIONS}")
    print(f"   Target FPS: {TARGET_FPS}")
    print(f"   Buffer capacity: {BUFFER_CAPACITY:,}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Learning rate: {LEARNING_RATE}")
    print(f"   Gamma: {GAMMA}")
    print(f"   Epsilon decay: {EPSILON_DECAY_RATE}")
    print(f"   Reward alive: {REWARD_ALIVE}")
    print(f"   Reward death: {REWARD_DEATH}")
    print(f"   ✅ Configuration loaded")
except Exception as e:
    print(f"   ❌ Error loading config: {e}")

# 8. GPU memory test
print(f"\n8. GPU Memory Test")
if torch.cuda.is_available():
    try:
        # Create a small tensor
        test_tensor = torch.randn(100, 100).cuda()
        print(f"   ✅ GPU tensor creation successful")
        
        # Check memory usage
        allocated = torch.cuda.memory_allocated(0) / 1e6
        cached = torch.cuda.memory_reserved(0) / 1e6
        print(f"   Allocated: {allocated:.2f} MB")
        print(f"   Cached: {cached:.2f} MB")
        
        # Clean up
        del test_tensor
        torch.cuda.empty_cache()
        print(f"   ✅ GPU cleanup successful")
    except Exception as e:
        print(f"   ❌ GPU test failed: {e}")
else:
    print(f"   ⚠️  Skipped (CUDA not available)")

# 9. Screen capture test
print(f"\n9. Screen Capture Test")
try:
    import mss
    import pygetwindow as gw
    
    with mss.mss() as sct:
        monitor = sct.monitors[0]
        print(f"   ✅ mss working")
        print(f"   Primary monitor: {monitor['width']}x{monitor['height']}")
    
    # Check if Chrome is open
    windows = gw.getWindowsWithTitle('Chrome')
    if windows:
        print(f"   ✅ Found {len(windows)} Chrome window(s)")
    else:
        print(f"   ⚠️  No Chrome windows found")
    
except Exception as e:
    print(f"   ❌ Screen capture test failed: {e}")

# 10. Module imports test
print(f"\n10. Module Import Test")
modules_to_test = [
    'agent',
    'model',
    'game_env',
    'trainer',
    'replay_buffer',
    'episode_tracker'
]

sys.path.insert(0, project_path)
all_imported = True
for module_name in modules_to_test:
    try:
        __import__(module_name)
        print(f"   ✅ {module_name}")
    except Exception as e:
        print(f"   ❌ {module_name}: {e}")
        all_imported = False

if not all_imported:
    print(f"\n   ⚠️  Some modules failed to import")

# Summary
print(f"\n{'='*70}")
print("DIAGNOSTIC SUMMARY")
print(f"{'='*70}")

issues = []
if not torch.cuda.is_available():
    issues.append("CUDA not available (training will be slow)")
if not all_exist:
    issues.append("Missing required project files")
if not os.path.exists(checkpoint_path):
    issues.append("No trained model checkpoint found")
if not all_imported:
    issues.append("Some modules failed to import")

if issues:
    print("\n⚠️  Issues found:")
    for issue in issues:
        print(f"   - {issue}")
    print("\nRecommendations:")
    if not torch.cuda.is_available():
        print("   - Install CUDA-enabled PyTorch for GPU acceleration")
    if not os.path.exists(checkpoint_path):
        print("   - Run main.py to start training")
    if not all_imported:
        print("   - Check error messages above and fix module issues")
else:
    print("\n✅ All systems operational!")
    print("\nReady to:")
    print("   - Run training: python main.py")
    print("   - Analyze model: python analyze_training.py")
    print("   - Debug visually: python debug_dashboard.py")
    print("   - Quick test: python quick_test.py")

print(f"\n{'='*70}\n")
