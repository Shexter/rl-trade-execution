"""Sell `total_shares` over `n_steps` on a simulated market (discrete Almgren-Chriss model).

Price: arithmetic random walk with permanent impact
    S[t+1] = S[t] + sigma * z[t] - gamma * n[t]
Each sale of n shares fills at S[t] - eta * n (temporary impact).
Cost = implementation shortfall = total_shares * S0 - cash received.
Reward per step is mark-to-market: n * (fill - S[t]) + shares_held_after * (S[t+1] - S[t]).
It telescopes, so episode return = -shortfall exactly, but each step only carries one step of price noise.
"""
import numpy as np


class ExecutionEnv:
    def __init__(self, total_shares=100_000, n_steps=20, s0=100.0, sigma=0.5, eta=1e-4, gamma=1e-5):
        self.total_shares, self.n_steps, self.s0 = total_shares, n_steps, s0
        self.sigma, self.eta, self.gamma = sigma, eta, gamma
        self.reset()

    def reset(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.price = self.prev_price = self.s0
        self.shares_left = float(self.total_shares)
        self.cash = 0.0
        return self.state()

    @property
    def steps_left(self):
        return self.n_steps - self.t

    def state(self):
        """[fraction of time left, fraction of shares left, last price move in sigmas]."""
        last_move = (self.price - self.prev_price) / self.sigma if self.sigma else 0.0
        return np.array([self.steps_left / self.n_steps, self.shares_left / self.total_shares, last_move],
                        dtype=np.float32)

    def step(self, shares):
        """Sell `shares` (clipped to [0, shares_left]); the final step force-sells everything left."""
        assert self.steps_left > 0, "episode is over, call reset()"
        n = self.shares_left if self.steps_left == 1 else float(np.clip(shares, 0, self.shares_left))
        fill = self.price - self.eta * n
        self.cash += n * fill
        self.shares_left -= n
        self.prev_price = self.price
        self.price += self.sigma * self.rng.standard_normal() - self.gamma * n
        reward = n * (fill - self.prev_price) + self.shares_left * (self.price - self.prev_price)
        self.t += 1
        return self.state(), reward, self.steps_left == 0

    def shortfall(self):
        return self.total_shares * self.s0 - self.cash


def run_episode(env, policy, seed=None):
    """Run `policy(env) -> shares` for one episode; return implementation shortfall in dollars."""
    env.reset(seed)
    done = False
    while not done:
        _, _, done = env.step(policy(env))
    return env.shortfall()
