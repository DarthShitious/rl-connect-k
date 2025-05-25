import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from agents.base_agent import Agent

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

class DQNAgent(Agent):
    def __init__(self, config, env):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_shape = env.observation_space.shape
        input_dim = int(np.prod(self.state_shape))
        self.action_dim = env.action_space.n
        self.policy_net = QNetwork(input_dim, self.action_dim).to(self.device)
        self.target_net = QNetwork(input_dim, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.dqn.lr)
        self.gamma = config.dqn.gamma
        self.epsilon = config.dqn.epsilon_start
        self.epsilon_end = config.dqn.epsilon_end
        self.epsilon_decay = config.dqn.epsilon_decay
        self.target_update_freq = config.dqn.target_update_freq
        self.memory = ReplayBuffer(int(1e5))
        self.steps_done = 0

    def select_action(self, state):
        self.steps_done += 1
        eps = self.epsilon_end + (self.epsilon - self.epsilon_end) *                     np.exp(-1. * self.steps_done / self.epsilon_decay)
        if random.random() < eps:
            return random.randrange(self.action_dim)
        else:
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                q_values = self.policy_net(state)
            return int(q_values.argmax().item())

    def update(self, batch_size):
        if len(self.memory) < batch_size:
            return None
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        q_values = self.policy_net(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target = rewards + self.gamma * next_q * (1 - dones)
        loss = nn.MSELoss()(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
