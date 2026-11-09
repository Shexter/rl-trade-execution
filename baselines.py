"""Baseline policies. A policy is `policy(env) -> shares to sell this step`."""
import numpy as np


def twap(env):
    """Sell evenly over the remaining steps."""
    return env.shares_left / env.steps_left


def almgren_chriss(risk_aversion=1e-6):
    """Almgren-Chriss (2000) optimal static schedule for linear impact and mean-variance utility.

    Holdings x_j = X sinh(kappa (N - j)) / sinh(kappa N), where
    cosh(kappa) = 1 + risk_aversion * sigma^2 / (2 * (eta - gamma / 2)). risk_aversion -> 0 gives TWAP.
    """
    cache = {}

    def policy(env):
        key = (env.total_shares, env.n_steps, env.sigma, env.eta, env.gamma)
        if key not in cache:
            eta_tilde = env.eta - env.gamma / 2
            assert eta_tilde > 0, "Almgren-Chriss needs eta > gamma / 2"
            kappa = np.arccosh(1 + risk_aversion * env.sigma**2 / (2 * eta_tilde))
            j = np.arange(env.n_steps + 1)
            if kappa < 1e-12:
                cache[key] = env.total_shares * (1 - j / env.n_steps)
            else:
                cache[key] = env.total_shares * np.sinh(kappa * (env.n_steps - j)) / np.sinh(kappa * env.n_steps)
        return env.shares_left - cache[key][env.t + 1]

    return policy
