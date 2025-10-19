import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
from model import DQN
from replay_buffer import ReplayBuffer


class DDQNAgent:
    def __init__(
        self,
        state_shape=(4, 84, 252),
        n_actions=3,
        learning_rate=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay_rate=0.9995,
        buffer_capacity=100000,
        batch_size=32,
        target_update_freq=10000,
        device=None
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_rate = epsilon_decay_rate
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.learn_step_counter = 0
        self.total_steps = 0
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        print(f"[AGENT] Using device: {self.device}")
        
        self.policy_net = DQN(n_frames=state_shape[0], n_actions=n_actions).to(self.device)
        self.target_net = DQN(n_frames=state_shape[0], n_actions=n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)
        
        print(f"[AGENT] DDQN initialized with {buffer_capacity:,} buffer capacity")
    
    def select_action(self, state, training=True, force_exploit=False):
        if len(self.replay_buffer) >= 1000:
            force_exploit = True
        
        if training and not force_exploit and random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax(dim=1).item()
    
    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
    
    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1)
            next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()
        
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            print(f"[TARGET] Network updated at step {self.learn_step_counter}")
        
        self.total_steps += 1
        if self.epsilon > self.epsilon_end:
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay_rate)
        
        return loss.item()
    
    def save(self, filepath):
        import tempfile
        import shutil
        import signal
        
        temp_path = filepath + '.tmp'
        original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        
        try:
            torch.save({
                'policy_net_state_dict': self.policy_net.state_dict(),
                'target_net_state_dict': self.target_net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'total_steps': self.total_steps,
                'learn_step_counter': self.learn_step_counter
            }, temp_path)
            
            if os.path.exists(filepath):
                os.remove(filepath)
            shutil.move(temp_path, filepath)
            
            print(f"[SAVE] Model saved to {filepath}")
        except Exception as e:
            print(f"[ERROR] Saving model: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise
        finally:
            signal.signal(signal.SIGINT, original_sigint_handler)
    
    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.total_steps = checkpoint.get('total_steps', 0)
        self.learn_step_counter = checkpoint['learn_step_counter']
        print(f"[LOAD] Model loaded from {filepath}")
        print(f"[LOAD] Epsilon: {self.epsilon:.4f}, Total steps: {self.total_steps:,}")
