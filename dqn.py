"""DQN: MLP Q-network, replay buffer, target network, epsilon-greedy over discrete sell sizes."""
import copy
import random
from collections import deque

import numpy as np
import torch
from torch import nn

ACTIONS = np.array([0.0, 0.05, 0.10, 0.20, 0.40])  # fraction of remaining shares to sell
REWARD_SCALE = 1e-4  # dollars -> roughly O(1) per episode


def shares_for(env, action):
    return ACTIONS[action] * env.shares_left


class Agent:
    def __init__(self, state_dim=3, hidden=64, lr=1e-3, gamma=1.0, buffer_size=50_000, batch_size=64,
                 target_sync=500, seed=0):
        torch.manual_seed(seed)
        self.rng = random.Random(seed)
        self.q = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                               nn.Linear(hidden, len(ACTIONS)))
        self.target = copy.deepcopy(self.q)
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.buffer = deque(maxlen=buffer_size)
        self.gamma, self.batch_size, self.target_sync, self.updates = gamma, batch_size, target_sync, 0

    def act(self, state, epsilon=0.0):
        if self.rng.random() < epsilon:
            return self.rng.randrange(len(ACTIONS))
        with torch.no_grad():
            return int(self.q(torch.as_tensor(state)).argmax())

    def policy(self, env):
        """Greedy policy in the `policy(env) -> shares` form used by run_episode."""
        return shares_for(env, self.act(env.state()))

    def remember(self, s, a, r, s2, done):
        self.buffer.append((s, a, r * REWARD_SCALE, s2, float(done)))

    def update(self):
        if len(self.buffer) < self.batch_size:
            return None
        s, a, r, s2, done = (torch.as_tensor(np.array(x)) for x in zip(*self.rng.sample(self.buffer, self.batch_size)))
        q = self.q(s).gather(1, a.long().unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            target = r.float() + self.gamma * (1 - done.float()) * self.target(s2).max(1).values
        loss = nn.functional.smooth_l1_loss(q, target)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.updates += 1
        if self.updates % self.target_sync == 0:
            self.target.load_state_dict(self.q.state_dict())
        return loss.item()


def random_policy(seed=0):
    rng = np.random.default_rng(seed)
    return lambda env: shares_for(env, rng.integers(len(ACTIONS)))
