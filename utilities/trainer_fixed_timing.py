    def run_training_step(self):
        """Execute one training step - 10 FPS LOCKED with proper timing"""
        step_start = time.time()
        
        # 1. Get current state (4 stacked frames)
        state = self.env.get_stacked_state()
        
        # 2. Select action (network inference)
        action = self.agent.select_action(state, training=True)
        
        # 3. Execute action (press/release key)
        self.env.execute_action(action)
        
        # 4. FPS limiter: Wait to let game react BEFORE capturing
        # This ensures we capture the frame AFTER the action has taken effect
        elapsed = time.time() - step_start
        target_frame_time = 0.1  # 10 FPS = 0.1s per frame
        if elapsed < target_frame_time:
            time.sleep(target_frame_time - elapsed)
        
        # 5. Capture NEXT frame (game has now reacted to our action)
        raw_frame = self.env.capture_frame()
        preprocessed_frame, gray_frame, resized_frame = self.env.preprocess_frame(raw_frame)
        
        # Add to frame stack
        self.env.add_frame_to_stack(preprocessed_frame)
        next_state = self.env.get_stacked_state()
        
        # 6. Check crash in NEXT state
        game_over = self.env.is_game_over(gray_frame)
        if game_over and len(self.env.crash_detection_buffer) >= 2:
            print(f"\nCRASH detected at step {self.step_count}!")
        
        self.env.update_crash_detection_buffer(gray_frame)
        
        # 7. Calculate base reward (NO death penalty yet)
        # Death penalty will be applied at episode end to the frame BEFORE frozen sequence
        if game_over:
            reward = 0.0  # Placeholder - frozen frames will be discarded anyway
        else:
            reward = calculate_reward(action, game_over=False)
        
        # Store transition in EPISODE buffer (NOT replay buffer yet)
        # We'll shape rewards and add to replay buffer at episode end
        timestamp = time.time() - self.episode_start_time
        self.episode_experiences.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': game_over,
            'frame_num': self.step_count,
            'timestamp': timestamp
        })
        
        # Track performance (total time for this step)
        step_time = (time.time() - step_start) * 1000
        self.step_times.append(step_time)
        
        self.step_count += 1
        
        # Progress with ACTUAL FPS (every 10 frames)
        if self.step_count % 10 == 0:
            total_elapsed = time.time() - self.episode_real_start_time
            fps = self.step_count / total_elapsed if total_elapsed > 0 else 0
            avg_time = np.mean(self.step_times[-10:])
            print(f"Step {self.step_count} | FPS: {fps:.1f} | "
                  f"{avg_time:.1f}ms/frame | Buf: {len(self.episode_experiences)} | "
                  f"e: {self.agent.epsilon:.3f}",
                  end='\r')
        
        return game_over
