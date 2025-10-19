"""
Test script to verify DDQN V2 implementation
Tests all critical components before training
"""
import numpy as np
import torch
from collections import deque

print("="*70)
print("DDQN V2 - Component Testing")
print("="*70)

# Test 1: Config
print("\\n[TEST 1] Config imports...")
try:
    from config import *
    print(f"  ✓ N_FRAME_STACK = {N_FRAME_STACK}")
    print(f"  ✓ STATE_SHAPE = {STATE_SHAPE}")
    print(f"  ✓ BUFFER_CAPACITY = {BUFFER_CAPACITY:,}")
    print(f"  ✓ LEARNING_RATE = {LEARNING_RATE}")
    print(f"  ✓ EPSILON_DECAY_STEPS = {EPSILON_DECAY_STEPS:,}")
    assert STATE_SHAPE == (4, 84, 252), "State shape must be (4, 84, 252)"
    assert BUFFER_CAPACITY == 100000, "Buffer must be 100k"
    print("  ✓ Config OK")
except Exception as e:
    print(f"  ✗ Config ERROR: {e}")
    exit(1)

# Test 2: Model
print("\\n[TEST 2] Model architecture...")
try:
    from model import DQN
    model = DQN(n_frames=4, n_actions=3)
    test_input = torch.randn(2, 4, 84, 252)
    output = model(test_input)
    print(f"  ✓ Input shape: {test_input.shape}")
    print(f"  ✓ Output shape: {output.shape}")
    assert output.shape == (2, 3), "Output must be (batch, 3)"
    print("  ✓ Model OK")
except Exception as e:
    print(f"  ✗ Model ERROR: {e}")
    exit(1)

# Test 3: Agent
print("\\n[TEST 3] DDQN Agent...")
try:
    from agent import DDQNAgent
    agent = DDQNAgent(
        state_shape=(4, 84, 252),
        n_actions=3,
        epsilon_decay_steps=500000
    )
    print(f"  ✓ Agent created")
    print(f"  ✓ Epsilon: {agent.epsilon}")
    print(f"  ✓ Device: {agent.device}")
    
    # Test action selection
    test_state = np.random.rand(4, 84, 252).astype(np.float32)
    action = agent.select_action(test_state)
    print(f"  ✓ Action selected: {action}")
    assert action in [0, 1, 2], "Action must be 0, 1, or 2"
    
    # Test epsilon decay
    initial_eps = agent.epsilon
    
    # Add experiences to buffer first
    for _ in range(100):
        state = np.random.rand(4, 84, 252).astype(np.float32)
        action = np.random.randint(0, 3)
        reward = 0.1
        next_state = np.random.rand(4, 84, 252).astype(np.float32)
        done = False
        agent.store_transition(state, action, reward, next_state, done)
    
    # Now learn (which will decay epsilon)
    for _ in range(100):
        loss = agent.learn()
    
    print(f"  ✓ Epsilon after 100 learn steps: {agent.epsilon:.4f}")
    assert agent.epsilon < initial_eps, "Epsilon should decay"
    print("  ✓ Agent OK")
except Exception as e:
    print(f"  ✗ Agent ERROR: {e}")
    exit(1)

# Test 4: Rewards
print("\\n[TEST 4] Reward function...")
try:
    from rewards import calculate_reward, get_frames_for_action
    
    # Test alive rewards
    r1 = calculate_reward(0, False, 1)  # Nothing
    r2 = calculate_reward(1, False, 1)  # Jump
    r3 = calculate_reward(2, False, 3)  # Duck (3 frames)
    print(f"  ✓ Nothing (1 frame): {r1}")
    print(f"  ✓ Jump (1 frame): {r2}")
    print(f"  ✓ Duck (3 frames): {r3}")
    assert r1 == 0.1, "Nothing should give +0.1"
    assert r2 == 0.1, "Jump should give +0.1"
    assert abs(r3 - 0.3) < 0.001, "Duck should give +0.3 (3 frames)"
    
    # Test death
    r_death = calculate_reward(0, True, 1)
    print(f"  ✓ Death: {r_death}")
    assert r_death == -10.0, "Death should give -10.0"
    
    # Test frame counting
    assert get_frames_for_action(0) == 1, "Nothing = 1 frame"
    assert get_frames_for_action(1) == 1, "Jump = 1 frame"
    assert get_frames_for_action(2) == 3, "Duck = 3 frames"
    print("  ✓ Rewards OK")
except Exception as e:
    print(f"  ✗ Rewards ERROR: {e}")
    exit(1)

# Test 5: Frame stacking
print("\\n[TEST 5] Frame stacking...")
try:
    frame_stack = deque(maxlen=4)
    
    # Add frames
    for i in range(5):
        frame = np.random.rand(1, 84, 252).astype(np.float32)
        frame_stack.append(frame)
        
        stacked = np.vstack(list(frame_stack))
        print(f"  ✓ After frame {i+1}: Shape = {stacked.shape}")
    
    assert stacked.shape == (4, 84, 252), "Stacked shape must be (4, 84, 252)"
    print("  ✓ Frame stacking OK")
except Exception as e:
    print(f"  ✗ Frame stacking ERROR: {e}")
    exit(1)

# Test 6: Double DQN logic
print("\\n[TEST 6] Double DQN algorithm...")
try:
    from agent import DDQNAgent
    import torch.nn as nn
    
    agent = DDQNAgent(buffer_capacity=1000)
    
    # Add some fake experiences
    for _ in range(100):
        state = np.random.rand(4, 84, 252).astype(np.float32)
        action = np.random.randint(0, 3)
        reward = 0.1
        next_state = np.random.rand(4, 84, 252).astype(np.float32)
        done = False
        agent.store_transition(state, action, reward, next_state, done)
    
    # Try learning
    loss = agent.learn()
    print(f"  ✓ Buffer size: {len(agent.replay_buffer)}")
    print(f"  ✓ Training loss: {loss:.4f}")
    print(f"  ✓ Learn step counter: {agent.learn_step_counter}")
    assert loss is not None, "Should return loss"
    print("  ✓ DDQN OK")
except Exception as e:
    print(f"  ✗ DDQN ERROR: {e}")
    exit(1)

print("\\n" + "="*70)
print("✓ ALL TESTS PASSED!")
print("="*70)
print()
print("Ready to train! Run: python main.py")
