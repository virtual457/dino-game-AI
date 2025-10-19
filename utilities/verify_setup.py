"""
Quick verification that everything works
"""

print("\n" + "="*70)
print("VERIFYING NEW SYSTEM")
print("="*70)

# Test 1: Episode Visualizer
print("\n[1/3] Testing Episode Visualizer...")
try:
    from episode_visualizer import test_visualizer
    test_visualizer()
    print("✅ Visualizer works!")
except Exception as e:
    print(f"❌ Visualizer failed: {e}")

# Test 2: Check main.py
print("\n[2/3] Testing main.py...")
try:
    from main import TrainingSession
    print("✅ Main.py imports successfully!")
except Exception as e:
    print(f"❌ Main.py failed: {e}")

# Test 3: Check file
print("\n[3/3] Checking output file...")
import os
image_path = "checkpoints/episode_images/latest_episode.png"
if os.path.exists(image_path):
    size_kb = os.path.getsize(image_path) / 1024
    print(f"✅ Image saved: {image_path} ({size_kb:.1f} KB)")
else:
    print(f"❌ Image not found: {image_path}")

print("\n" + "="*70)
print("VERIFICATION COMPLETE")
print("="*70)
print("\nNext steps:")
print("  1. View the test image:")
print("     start checkpoints\\episode_images\\latest_episode.png")
print()
print("  2. If it looks good, start training:")
print("     python main.py")
print()
print("  3. After each episode, refresh the image to see results!")
print("="*70 + "\n")
