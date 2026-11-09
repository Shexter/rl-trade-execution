"""Train DQN on ExecutionEnv with MLflow logging. Usage: python train.py --seeds 0 1 2"""
import argparse

import mlflow
import numpy as np
import torch

from baselines import twap
from dqn import Agent, random_policy, shares_for
from env import ExecutionEnv, run_episode

EVAL_SEEDS = range(10**6, 10**6 + 200)  # held out from training episodes


def mean_cost(env, policy):
    return float(np.mean([run_episode(env, policy, seed=s) for s in EVAL_SEEDS]))


def train(seed, episodes, lr, eps_start=1.0, eps_end=0.05, eval_every=250):
    env = ExecutionEnv()
    agent = Agent(lr=lr, seed=seed)
    with mlflow.start_run(run_name=f"dqn-seed{seed}"):
        mlflow.log_params({"seed": seed, "episodes": episodes, "lr": lr, "eps_start": eps_start,
                           "eps_end": eps_end, **{k: getattr(env, k) for k in
                                                  ("total_shares", "n_steps", "sigma", "eta", "gamma")}})
        mlflow.log_metrics({"twap_cost": mean_cost(env, twap), "random_cost": mean_cost(env, random_policy(seed))})
        for ep in range(episodes):
            eps = max(eps_end, eps_start - (eps_start - eps_end) * ep / (0.6 * episodes))
            s, done, losses = env.reset(seed * 10**5 + ep), False, []
            while not done:
                a = agent.act(s, eps)
                s2, r, done = env.step(shares_for(env, a))
                agent.remember(s, a, r, s2, done)
                if (loss := agent.update()) is not None:
                    losses.append(loss)
                s = s2
            mlflow.log_metrics({"train_cost": env.shortfall(), "epsilon": eps,
                                "loss": float(np.mean(losses)) if losses else 0.0}, step=ep)
            if (ep + 1) % eval_every == 0:
                cost = mean_cost(env, agent.policy)
                mlflow.log_metric("eval_cost", cost, step=ep)
                print(f"seed {seed} ep {ep + 1}: eval cost {cost:,.0f}")
        torch.save(agent.q.state_dict(), path := f"dqn_seed{seed}.pt")
        mlflow.log_artifact(path)
        return cost


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--episodes", type=int, default=5000)
    p.add_argument("--lr", type=float, default=3e-4)
    args = p.parse_args()
    mlflow.set_experiment("dqn-execution")
    env = ExecutionEnv()
    print(f"TWAP {mean_cost(env, twap):,.0f}  random {mean_cost(env, random_policy()):,.0f}")
    for seed in args.seeds:
        train(seed, args.episodes, args.lr)
