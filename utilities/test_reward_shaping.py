"""
Test reward shaping logic with diminishing penalties
Simulates an episode and verifies frozen frames are removed correctly
"""
import numpy as np
from config import REWARD_DEATH, REWARD_ALIVE


def test_reward_shaping():
    """Test the reward shaping logic with diminishing penalties"""
    print("="*70)
    print("TESTING REWARD SHAPING LOGIC - DIMINISHING PENALTIES")
    print("="*70)
    
    # Simulate an episode with 20 frames
    episode_experiences = []
    
    print("\n1. Creating simulated episode (20 frames)...")
    for i in range(20):
        exp = {
            'state': np.random.rand(4, 84, 252),
            'action': np.random.randint(0, 3),
            'reward': 0.0 if i >= 17 else REWARD_ALIVE,  # Last 3 are frozen
            'next_state': np.random.rand(4, 84, 252),
            'done': True if i == 17 else False,  # Game over detected at frame 17
            'frame_num': i,
            'timestamp': i * 0.1
        }
        episode_experiences.append(exp)
    
    print(f"   Total frames: {len(episode_experiences)}")
    print(f"   Frames 0-16: +{REWARD_ALIVE} (alive)")
    print(f"   Frames 17-19: 0.0 (frozen, placeholder)")
    print()
    print("   Timeline:")
    print("   Frames: ... 13   14   15   16   17   18   19")
    print("   Status: ... ok   ok   ok   ok  crash frozen frozen")
    
    # Apply reward shaping
    print("\n2. Applying reward shaping...")
    
    original_count = len(episode_experiences)
    frozen_count = 3
    
    # Remove last 3 frames (n+1=17, n+2=18, n+3=19 are frozen)
    valid_experiences = episode_experiences[:-frozen_count]
    
    # Apply diminishing penalties to last 4 frames
    # Frame 16 (n) = -10, Frame 15 (n-1) = -7, Frame 14 (n-2) = -5, Frame 13 (n-3) = -4
    penalties = [-10.0, -7.0, -5.0, -4.0]
    
    for i, penalty in enumerate(penalties):
        idx = -(i + 1)  # -1, -2, -3, -4
        if abs(idx) <= len(valid_experiences):
            valid_experiences[idx]['reward'] = penalty
            if i == 0:  # Mark crash frame as done
                valid_experiences[idx]['done'] = True
    
    print(f"   Original frames: {original_count}")
    print(f"   Removed frozen: {frozen_count} (frames 17, 18, 19)")
    print(f"   Valid frames: {len(valid_experiences)} (frames 0-16)")
    print()
    print("   Applied penalties:")
    print("   Frame 13 (n-3): -4.0")
    print("   Frame 14 (n-2): -5.0")
    print("   Frame 15 (n-1): -7.0")
    print("   Frame 16 (n):   -10.0 [CRASH FRAME]")
    
    # Verify results
    print("\n3. Verification:")
    assert len(valid_experiences) == 17, "Should have 17 valid frames (0-16)"
    assert valid_experiences[-1]['frame_num'] == 16, "Last valid frame should be 16"
    assert valid_experiences[-1]['reward'] == -10.0, "Frame 16 should have -10.0"
    assert valid_experiences[-2]['reward'] == -7.0, "Frame 15 should have -7.0"
    assert valid_experiences[-3]['reward'] == -5.0, "Frame 14 should have -5.0"
    assert valid_experiences[-4]['reward'] == -4.0, "Frame 13 should have -4.0"
    assert valid_experiences[-1]['done'] == True, "Frame 16 should be marked as done"
    print("   ✓ Correct number of frames (17)")
    print("   ✓ Last valid frame is #16")
    print("   ✓ Frame 16 (n):   -10.0")
    print("   ✓ Frame 15 (n-1): -7.0")
    print("   ✓ Frame 14 (n-2): -5.0")
    print("   ✓ Frame 13 (n-3): -4.0")
    print("   ✓ Done flag set on crash frame")
    
    # Check rewards distribution
    print("\n4. Reward distribution:")
    rewards = [exp['reward'] for exp in valid_experiences]
    print(f"   Frames 0-12:  {[f'+{REWARD_ALIVE}' for _ in range(13)]}")
    print(f"   Frame 13:     {rewards[-4]}")
    print(f"   Frame 14:     {rewards[-3]}")
    print(f"   Frame 15:     {rewards[-2]}")
    print(f"   Frame 16:     {rewards[-1]} [CRASH]")
    
    total_reward = sum(rewards)
    expected_reward = 13 * REWARD_ALIVE + (-4.0) + (-5.0) + (-7.0) + (-10.0)
    print(f"\n5. Total episode reward: {total_reward:.2f}")
    print(f"   Expected: {expected_reward:.2f}")
    print(f"   Calculation: 13×{REWARD_ALIVE} + (-4) + (-5) + (-7) + (-10)")
    assert abs(total_reward - expected_reward) < 0.01, "Total reward mismatch"
    print("   ✓ Total reward matches expected value")
    
    print("\n" + "="*70)
    print("ALL TESTS PASSED!")
    print("="*70)
    
    print("\nSummary:")
    print(f"  • Episode had {original_count} frames (0-19)")
    print(f"  • Removed {frozen_count} frozen frames (17, 18, 19)")
    print(f"  • Kept {len(valid_experiences)} valid frames (0-16)")
    print(f"  • Applied diminishing penalties:")
    print(f"    - Frame 16 (crash): -10.0")
    print(f"    - Frame 15: -7.0")
    print(f"    - Frame 14: -5.0")
    print(f"    - Frame 13: -4.0")
    print(f"  • Agent learns: 'Actions in frames 13-16 led to crash'")
    print(f"  • Network will propagate these penalties backward further")


if __name__ == "__main__":
    test_reward_shaping()
