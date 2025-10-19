"""
Quick test script for the new main.py and episode_visualizer.py
"""

print("Testing new implementations...")
print("="*70)

# Test 1: Episode Visualizer
print("\n1. Testing Episode Visualizer (matplotlib version)...")
try:
    from episode_visualizer import test_visualizer
    test_visualizer()
    print("✅ Episode Visualizer test passed!")
except Exception as e:
    print(f"❌ Episode Visualizer test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Main.py imports
print("\n2. Testing Main.py imports...")
try:
    from main import TrainingSession
    print("✅ Main.py imports successfully!")
    print("   - TrainingSession class available")
except Exception as e:
    print(f"❌ Main.py import failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Verify directory structure
print("\n3. Checking directory structure...")
import os
required_dirs = [
    'checkpoints',
    'checkpoints/episode_images'
]

for dir_path in required_dirs:
    if os.path.exists(dir_path):
        print(f"   ✅ {dir_path}")
    else:
        print(f"   ⚠️  {dir_path} (will be created on first run)")

print("\n" + "="*70)
print("All tests complete!")
print("\nYou can now run:")
print("  python main.py           # Start training")
print("  python episode_visualizer.py  # Test visualizer")
print("="*70)
