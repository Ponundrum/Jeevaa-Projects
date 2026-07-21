"""Monte Carlo Greeks under GBM by three independent methods, each validated
against the closed-form :func:`qmc.analytic.bs_greeks`:

- **pathwise derivative** — differentiate the payoff along the path (low variance,
  needs a differentiable payoff);
- **likelihood ratio** — differentiate the density, not the payoff (works for
  discontinuous payoffs like digitals; higher variance);
- **finite difference** with common random numbers — bump-and-revalue reusing the
  same normals, so the differencing noise cancels.

All use the exact one-step terminal ``S_T = S0 exp((r-q-sigma^2/2)T + sigma sqrt(T) Z)``.
"""
from __future__ import annotations

import numpy as np

from .config import get_rng


def _indicator_itm(ST, K, kind):
    return (ST > K).astype(float) if kind == "call" else -(ST < K).astype(float)


def mc_greeks(S0, K, T, r, sigma, q=0.0, kind="call", n_paths=400_000, rng=None, h_rel=1e-3):
    """Return delta/vega/gamma by pathwise, likelihood-ratio, and finite-difference
    (common random numbers) methods, as a nested dict ``{method: {greek: value}}``."""
    rng = rng or get_rng(7)
    disc = np.exp(-r * T)
    Z = rng.standard_normal(n_paths)
    sqrtT = np.sqrt(T)

    def terminal(s0, sig):
        return s0 * np.exp((r - q - 0.5 * sig ** 2) * T + sig * sqrtT * Z)

    ST = terminal(S0, sigma)
    payoff = np.maximum(ST - K, 0.0) if kind == "call" else np.maximum(K - ST, 0.0)

    # --- pathwise ---
    ind = _indicator_itm(ST, K, kind)
    delta_pw = disc * np.mean(ind * ST / S0)
    vega_pw = disc * np.mean(ind * ST * (sqrtT * Z - sigma * T))

    # --- likelihood ratio ---
    delta_lr = disc * np.mean(payoff * Z / (S0 * sigma * sqrtT))
    vega_lr = disc * np.mean(payoff * ((Z ** 2 - 1) / sigma - Z * sqrtT))
    gamma_lr = disc * np.mean(payoff * (Z ** 2 - Z * sigma * sqrtT - 1) / (S0 ** 2 * sigma ** 2 * T))

    # --- finite difference, common random numbers ---
    hS, hsig = S0 * h_rel, sigma * h_rel
    price = lambda s0, sig: disc * np.mean(np.maximum((terminal(s0, sig) - K) * (1 if kind == "call" else -1), 0.0))
    p0 = price(S0, sigma)
    delta_fd = (price(S0 + hS, sigma) - price(S0 - hS, sigma)) / (2 * hS)
    vega_fd = (price(S0, sigma + hsig) - price(S0, sigma - hsig)) / (2 * hsig)
    gamma_fd = (price(S0 + hS, sigma) - 2 * p0 + price(S0 - hS, sigma)) / hS ** 2

    return {
        "pathwise": {"delta": delta_pw, "vega": vega_pw},
        "likelihood_ratio": {"delta": delta_lr, "vega": vega_lr, "gamma": gamma_lr},
        "finite_diff": {"delta": delta_fd, "vega": vega_fd, "gamma": gamma_fd},
    }
