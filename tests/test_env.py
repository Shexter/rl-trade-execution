import numpy as np

from baselines import almgren_chriss, twap
from env import ExecutionEnv, run_episode


def test_all_shares_sold_by_T():
    env = ExecutionEnv()
    for policy in (twap, almgren_chriss(), lambda e: 0.0):  # do-nothing policy gets force-liquidated
        run_episode(env, policy, seed=0)
        assert env.shares_left == 0 and env.steps_left == 0
        assert np.isclose(env.cash, env.total_shares * env.s0 - env.shortfall())


def test_twap_shortfall_zero_without_impact():
    env = ExecutionEnv(eta=0.0, gamma=0.0, sigma=0.0)
    assert np.isclose(run_episode(env, twap, seed=0), 0.0)
    # with volatility, cost is pure noise: mean ~0 relative to its spread
    env = ExecutionEnv(eta=0.0, gamma=0.0)
    costs = np.array([run_episode(env, twap, seed=s) for s in range(2000)])
    assert abs(costs.mean()) < 4 * costs.std() / np.sqrt(len(costs))


def test_same_seed_same_episode():
    env = ExecutionEnv()
    assert run_episode(env, twap, seed=42) == run_episode(env, twap, seed=42)
    assert run_episode(env, twap, seed=42) != run_episode(env, twap, seed=43)


def test_twap_cost_matches_closed_form_without_noise():
    # sigma=0: shortfall = eta * sum(n^2) + gamma * sum_t n_t * (shares sold before t)
    env = ExecutionEnv(sigma=0.0)
    n = env.total_shares / env.n_steps
    sold_before = n * np.arange(env.n_steps)
    expected = env.eta * env.n_steps * n**2 + env.gamma * np.sum(n * sold_before)
    assert np.isclose(run_episode(env, twap, seed=0), expected)


def test_almgren_chriss_front_loads_and_tends_to_twap():
    env = ExecutionEnv()
    env.reset(0)
    assert almgren_chriss(1e-6)(env) > twap(env)
    assert np.isclose(almgren_chriss(0.0)(env), twap(env))
