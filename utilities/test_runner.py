"""
Test Runner - Single Episode Analysis

Tests one complete episode and shows:
- Frame-by-frame timing
- Which frames were added to buffer
- Detailed performance metrics
"""

import time
import os
import numpy as np
import torch
import cv2
from config import *
from game_env import DinoGameEnv
from agent import DQNAgent
from rewards import calculate_reward
from action_tracker import ActionTracker


def test_single_episode():
    """Run one episode and analyze performance"""
    
    print("\n" + "="*70)
    print("SINGLE EPISODE TEST - Performance Analysis")
    print("="*70)
    
    # Create checkpoint directory
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Initialize components
    env = DinoGameEnv()
    agent = DQNAgent(
        n_actions=N_ACTIONS,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        epsilon_start=0.1,  # Low epsilon for more deterministic behavior
        epsilon_end=0.01,
        epsilon_decay=0.995,
        batch_size=BATCH_SIZE,
        buffer_capacity=BUFFER_CAPACITY,
        target_update_freq=TARGET_UPDATE_FREQ
    )
    action_tracker = ActionTracker()
    
    # Try to load existing model
    if os.path.exists(CHECKPOINT_PATH):
        print(f"\n💾 Loading model: {CHECKPOINT_PATH}")
        agent.load(CHECKPOINT_PATH)
    else:
        print("\n🆕 Using untrained model")
    
    # Open game
    if not env.open_game():
        print("❌ Failed to open game!")
        return
    
    print("\nStarting single episode test in 2 seconds...")
    time.sleep(2)
    
    # Episode state
    step_count = 0
    total_reward = 0
    prev_state = None
    frame_timings = []
    buffer_additions = []
    stored_frames = []  # Store frames that were added to buffer
    
    episode_start = time.time()
    
    print("\n" + "="*70)
    print("EPISODE RUNNING - Frame-by-Frame Analysis")
    print("="*70)
    print()
    
    try:
        while True:
            frame_start = time.time()
            
            # === TIMING: Capture ===
            capture_start = time.time()
            raw_frame = env.capture_frame()
            capture_time = (time.time() - capture_start) * 1000
            
            # === TIMING: Preprocess ===
            preprocess_start = time.time()
            state, gray_frame, resized_frame = env.preprocess_frame(raw_frame)
            preprocess_time = (time.time() - preprocess_start) * 1000
            
            # === TIMING: Inference ===
            inference_start = time.time()
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                q_values = agent.policy_net(state_tensor)
                q_values_np = q_values.cpu().numpy()[0]
                del state_tensor, q_values
                if agent.device.type == 'cuda':
                    torch.cuda.empty_cache()
            
            action = agent.select_action(state, training=True)
            inference_time = (time.time() - inference_start) * 1000
            
            # === TIMING: Game Over Check ===
            check_start = time.time()
            game_over = env.is_game_over(gray_frame)
            env.update_frame_history(gray_frame)
            check_time = (time.time() - check_start) * 1000
            
            # === TIMING: Reward Calculation ===
            reward_start = time.time()
            reward, penalty = calculate_reward(
                action, game_over,
                action_tracker.action_history,
                action_tracker.jump_count,
                action_tracker.duck_count,
                action_tracker.total_actions
            )
            reward_time = (time.time() - reward_start) * 1000
            total_reward += reward
            
            # Track action
            action_tracker.record_action(action, resized_frame)
            
            # === TIMING: Action Execution ===
            action_start = time.time()
            env.execute_action(action)
            action_time = (time.time() - action_start) * 1000
            
            # === TIMING: Buffer Storage ===
            storage_start = time.time()
            buffer_added = False
            if prev_state is not None:
                initial_buffer_size = len(agent.replay_buffer)
                agent.store_transition(prev_state, action, reward, state, game_over)
                buffer_added = len(agent.replay_buffer) > initial_buffer_size
                buffer_additions.append({
                    'frame': step_count,
                    'added': buffer_added,
                    'buffer_size': len(agent.replay_buffer)
                })
                
                # Store frame for visualization if added
                if buffer_added:
                    stored_frames.append({
                        'frame_num': step_count,
                        'image': resized_frame.copy(),
                        'action': action,
                        'reward': reward
                    })
            storage_time = (time.time() - storage_start) * 1000
            
            # === TIMING: Training ===
            train_start = time.time()
            loss = None
            if prev_state is not None:
                loss = agent.learn()
            train_time = (time.time() - train_start) * 1000
            
            # Update state
            if not game_over:
                prev_state = state.copy()
            else:
                prev_state = None
            
            # Total frame time
            total_frame_time = (time.time() - frame_start) * 1000
            
            # Store timing info
            action_names = ["NOTHING", "JUMP", "DUCK"]
            frame_timings.append({
                'frame': step_count,
                'capture': capture_time,
                'preprocess': preprocess_time,
                'inference': inference_time,
                'check_game_over': check_time,
                'reward_calc': reward_time,
                'action_exec': action_time,
                'storage': storage_time,
                'training': train_time,
                'total': total_frame_time,
                'action': action_names[action],
                'reward': reward,
                'q_values': q_values_np.tolist(),
                'buffer_added': buffer_added,
                'game_over': game_over
            })
            
            # Print frame info
            print(f"Frame {step_count:3d}: "
                  f"Cap={capture_time:5.2f}ms "
                  f"Pre={preprocess_time:4.2f}ms "
                  f"Inf={inference_time:4.2f}ms "
                  f"Act={action_time:5.2f}ms "
                  f"Train={train_time:5.2f}ms "
                  f"Tot={total_frame_time:6.2f}ms | "
                  f"Action={action_names[action]:7s} "
                  f"Reward={reward:+6.2f} "
                  f"Buf={'✅' if buffer_added else '  '}")
            
            step_count += 1
            
            # End episode on crash
            if game_over:
                print(f"\n🔴 CRASH at frame {step_count}!")
                break
            
            # FPS control
            elapsed = time.time() - frame_start
            if elapsed < FRAME_TIME:
                time.sleep(FRAME_TIME - elapsed)
    
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    
    # Episode complete - print analysis
    episode_duration = time.time() - episode_start
    
    print("\n" + "="*70)
    print("EPISODE COMPLETE - Analysis")
    print("="*70)
    
    # Summary stats
    print(f"\n📊 Episode Summary:")
    print(f"  Total frames:      {step_count}")
    print(f"  Total reward:      {total_reward:.2f}")
    print(f"  Episode duration:  {episode_duration:.2f}s")
    print(f"  Average FPS:       {step_count / episode_duration:.2f}")
    
    # Timing analysis
    if frame_timings:
        avg_timings = {
            'capture': np.mean([f['capture'] for f in frame_timings]),
            'preprocess': np.mean([f['preprocess'] for f in frame_timings]),
            'inference': np.mean([f['inference'] for f in frame_timings]),
            'check_game_over': np.mean([f['check_game_over'] for f in frame_timings]),
            'reward_calc': np.mean([f['reward_calc'] for f in frame_timings]),
            'action_exec': np.mean([f['action_exec'] for f in frame_timings]),
            'storage': np.mean([f['storage'] for f in frame_timings]),
            'training': np.mean([f['training'] for f in frame_timings]),
            'total': np.mean([f['total'] for f in frame_timings])
        }
        
        print(f"\n⏱️  Average Timings:")
        print(f"  Capture:          {avg_timings['capture']:.2f}ms")
        print(f"  Preprocess:       {avg_timings['preprocess']:.2f}ms")
        print(f"  Inference:        {avg_timings['inference']:.2f}ms")
        print(f"  Game Over Check:  {avg_timings['check_game_over']:.2f}ms")
        print(f"  Reward Calc:      {avg_timings['reward_calc']:.2f}ms")
        print(f"  Action Execution: {avg_timings['action_exec']:.2f}ms")
        print(f"  Buffer Storage:   {avg_timings['storage']:.2f}ms")
        print(f"  Training:         {avg_timings['training']:.2f}ms")
        print(f"  TOTAL per frame:  {avg_timings['total']:.2f}ms")
        print(f"  Target (10 FPS):  {FRAME_TIME*1000:.2f}ms")
        
        # Check if we're meeting FPS target
        if avg_timings['total'] < FRAME_TIME * 1000:
            headroom = FRAME_TIME * 1000 - avg_timings['total']
            print(f"  ✅ {headroom:.2f}ms headroom")
        else:
            print(f"  ⚠️  Running slower than target!")
    
    # Buffer analysis
    print(f"\n📦 Buffer Analysis:")
    print(f"  Final buffer size: {len(agent.replay_buffer)}")
    
    frames_added = [b for b in buffer_additions if b['added']]
    frames_skipped = [b for b in buffer_additions if not b['added']]
    
    print(f"  Frames added:      {len(frames_added)}")
    print(f"  Frames skipped:    {len(frames_skipped)}")
    
    if frames_added:
        print(f"\n  Added to buffer:")
        for entry in frames_added:
            print(f"    Frame {entry['frame']:3d} → Buffer size: {entry['buffer_size']}")
    
    if frames_skipped:
        print(f"\n  Skipped (frozen after crash):")
        for entry in frames_skipped[-5:]:  # Show last 5
            print(f"    Frame {entry['frame']:3d}")
    
    # Action breakdown
    episode_stats = action_tracker.get_episode_stats()
    print(f"\n🎮 Actions Taken:")
    print(f"  Total:    {episode_stats['total']}")
    print(f"  Jumps:    {episode_stats['jumps']} ({episode_stats['jump_pct']:.1f}%)")
    print(f"  Ducks:    {episode_stats['ducks']} ({episode_stats['duck_pct']:.1f}%)")
    print(f"  Nothing:  {episode_stats['nothing']} ({episode_stats['nothing_pct']:.1f}%)")
    
    # GPU stats
    if agent.device.type == 'cuda':
        print(f"\n🖥️  GPU:")
        gpu_memory = torch.cuda.memory_allocated() / 1024**2
        print(f"  Memory used: {gpu_memory:.1f}MB")
    
    # Visualize stored frames
    if stored_frames:
        print(f"\n🖼️  Visualizing {len(stored_frames)} frames stored in buffer...")
        print("  Press any key to advance, 'q' to skip visualization")
        
        action_names = ["NOTHING", "JUMP", "DUCK"]
        action_colors = [(150, 150, 150), (0, 255, 0), (255, 255, 0)]
        
        for i, frame_data in enumerate(stored_frames):
            # Create display - preserve aspect ratio
            display_width = 600
            display_height = int(display_width * (INPUT_HEIGHT / INPUT_WIDTH))  # Preserve ratio
            frame_display = cv2.resize(frame_data['image'], (display_width, display_height))
            frame_3ch = cv2.cvtColor(frame_display, cv2.COLOR_GRAY2BGR)
            
            # Create canvas with padding
            canvas_height = 400
            canvas = np.zeros((canvas_height, display_width, 3), dtype=np.uint8)
            
            # Center the frame vertically
            y_offset = (canvas_height - display_height) // 2
            canvas[y_offset:y_offset+display_height, 0:display_width] = frame_3ch
            
            # Add info overlay
            action_name = action_names[frame_data['action']]
            action_color = action_colors[frame_data['action']]
            
            cv2.putText(canvas, f"Frame {frame_data['frame_num']}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(canvas, f"Buffer #{i+1}/{len(stored_frames)}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            cv2.putText(canvas, f"Action: {action_name}", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, action_color, 2)
            cv2.putText(canvas, f"Reward: {frame_data['reward']:.2f}", (10, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(canvas, f"Input: {INPUT_WIDTH}x{INPUT_HEIGHT}", (10, canvas_height-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
            
            # Border color based on action
            cv2.rectangle(canvas, (0, y_offset), (display_width-1, y_offset+display_height-1), 
                         action_color, 5)
            
            cv2.imshow('Frames Stored in Buffer', canvas)
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('q'):
                print("  Skipping remaining frames...")
                break
        
        cv2.destroyAllWindows()
        print(f"  ✅ Visualization complete!")
    
    print("\n" + "="*70)
    print("✅ Test complete!")
    print("="*70)


if __name__ == "__main__":
    test_single_episode()
