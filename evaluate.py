"""Evaluate saved DQNs vs TWAP, Almgren-Chriss and random on 1,000 held-out episodes. Writes results/."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from baselines import almgren_chriss, twap
from dqn import Agent, random_policy
from env import ExecutionEnv, run_episode

SEEDS = range(2 * 10**6, 2 * 10**6 + 1000)  # disjoint from training and train-time eval seeds


def load_dqn(seed):
    agent = Agent(seed=seed)
    agent.q.load_state_dict(torch.load(f"dqn_seed{seed}.pt"))
    return agent.policy


def inventory_path(env, policy, seed):
    env.reset(seed)
    path, done = [env.shares_left], False
    while not done:
        _, _, done = env.step(policy(env))
        path.append(env.shares_left)
    return path


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    env = ExecutionEnv()
    policies = {"TWAP": twap, "Almgren-Chriss": almgren_chriss(), "Random": random_policy(0),
                **{f"DQN (seed {s})": load_dqn(s) for s in range(3)}}
    df = pd.DataFrame([{"policy": name, "episode": s, "shortfall": run_episode(env, p, seed=s)}
                       for name, p in policies.items() for s in SEEDS])
    df.to_csv("results/episodes.csv", index=False)
    table = df.groupby("policy", sort=False)["shortfall"].agg(["mean", "std"]).round(0)
    twap_mean = table.loc["TWAP", "mean"]
    table["vs TWAP"] = (table["mean"] / twap_mean - 1).map("{:+.1%}".format)
    print(table.to_markdown())

    sns.set_theme(style="whitegrid")
    ax = sns.boxplot(df, x="shortfall", y="policy", showfliers=False)
    ax.set(xlabel="Implementation shortfall ($), 1,000 held-out episodes", ylabel="")
    plt.tight_layout(); plt.savefig("results/cost_distribution.png", dpi=150); plt.close()

    for name in ("TWAP", "Almgren-Chriss", "DQN (seed 2)"):
        plt.plot(inventory_path(env, policies[name], SEEDS[0]), marker="o", ms=3, label=name)
    plt.gca().set(xlabel="Step", ylabel="Shares left", title="Inventory over one held-out episode")
    plt.legend(); plt.tight_layout(); plt.savefig("results/inventory.png", dpi=150)
