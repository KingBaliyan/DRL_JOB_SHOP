import torch
import torch.nn as nn
import torch.optim as optim
import random

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.memory = []
        self.gamma = 0.95
        self.epsilon = 1.0
        self.action_size = action_size

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state)
        return torch.argmax(self.model(state)).item()

    def remember(self, s, a, r, ns, d):
        self.memory.append((s, a, r, ns, d))
        if len(self.memory) > 2000:
            self.memory.pop(0)

    def train(self, batch=32):
        if len(self.memory) < batch:
            return

        samples = random.sample(self.memory, batch)

        for s, a, r, ns, d in samples:
            s = torch.FloatTensor(s)
            ns = torch.FloatTensor(ns)

            target = r
            if not d:
                target += self.gamma * torch.max(self.model(ns)).item()

            q_val = self.model(s)[a]
            loss = (q_val - target) ** 2

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.epsilon = max(0.01, self.epsilon * 0.995)