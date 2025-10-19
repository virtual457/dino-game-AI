"""
Reward function for Dino Game RL Agent

Actions:
0 = Do nothing
1 = Jump (spacebar)
2 = Duck (down arrow)
"""

def calculate_reward(action, game_over, action_history, jump_count, duck_count, total_actions):
    """
    Calculate reward with dynamic penalty for jumping and ducking
    
    Args:
        action: Current action (0=nothing, 1=jump, 2=duck)
        game_over: Whether game crashed
        action_history: List of last 500 actions
        jump_count: Number of jumps in current episode
        duck_count: Number of ducks in current episode
        total_actions: Total actions in current episode
    
    Returns:
        reward: Float reward value
        current_penalty: The penalty applied (for logging)
    """
    if game_over:
        return -10.0, 0.0  # Large penalty for dying, no action penalty
    
    # Base reward for staying alive
    reward = 0.5
    
    # Penalize BOTH jump and duck actions (encourage doing nothing)
    if action in [1, 2]:  # Jump or Duck
        # Calculate movement % from last 500 actions
        if len(action_history) >= 100:
            # Count movements (jumps + ducks)
            recent_movements = sum(1 for a in action_history if a in [1, 2])
            movement_pct_500 = (recent_movements / len(action_history)) * 100
        else:
            # If not enough history, use current episode stats
            total_movements = jump_count + duck_count
            movement_pct_500 = (total_movements / total_actions * 100) if total_actions > 0 else 0.0
        
        # Penalty scale - lenient until 50%, then ramps up:
        # 0-10%:   0.0  (optimal!)
        # 10-30%:  0.05 (tiny nudge)
        # 30-50%:  0.1  (small nudge)
        # 50-60%:  1.0  (now it gets serious!)
        # 60-70%:  3.0  (heavy)
        # 70%+:    5.0  (extreme)
        
        if movement_pct_500 < 10:
            penalty = 0.0
        elif movement_pct_500 < 30:
            penalty = 0.05
        elif movement_pct_500 < 50:
            penalty = 0.1
        elif movement_pct_500 < 60:
            penalty = 1.0
        elif movement_pct_500 < 70:
            penalty = 3.0
        else:
            penalty = 5.0
        
        reward -= penalty
        return reward, penalty
    
    # No action taken - full reward
    return reward, 0.0


if __name__ == "__main__":
    # Test the reward function
    action_history = [0] * 400 + [1] * 50 + [2] * 50  # 10% jumps, 10% ducks = 20% total
    
    # Test different actions
    print("Testing reward function:")
    print()
    
    # No action
    r, p = calculate_reward(0, False, action_history, 5, 5, 100)
    print(f"No action:  reward={r:.2f}, penalty={p:.2f}")
    
    # Jump
    r, p = calculate_reward(1, False, action_history, 5, 5, 100)
    print(f"Jump:       reward={r:.2f}, penalty={p:.2f}")
    
    # Duck
    r, p = calculate_reward(2, False, action_history, 5, 5, 100)
    print(f"Duck:       reward={r:.2f}, penalty={p:.2f}")
    
    # Game over
    r, p = calculate_reward(0, True, action_history, 5, 5, 100)
    print(f"Game over:  reward={r:.2f}, penalty={p:.2f}")
