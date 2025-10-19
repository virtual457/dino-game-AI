"""
Test the new simplified episode visualizer
"""

print("="*70)
print("Testing New Simplified Episode Visualizer")
print("="*70)

try:
    from episode_visualizer import test_visualizer
    
    print("\nRunning test...")
    test_visualizer()
    
    print("\n" + "="*70)
    print("SUCCESS! Check the output:")
    print("="*70)
    print("\nOpen this file:")
    print("  checkpoints/episode_images/latest_episode.png")
    print()
    print("You should see:")
    print("  - Top row: First 5 frames with stats below each")
    print("  - Bottom row: Last 5 frames with stats below each")
    print("  - No performance stats panel")
    print("  - Full image space used for frames")
    print()
    print("To view:")
    print("  start checkpoints\\episode_images\\latest_episode.png")
    print("="*70)
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
