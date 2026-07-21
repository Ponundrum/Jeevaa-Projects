"""qmc — a from-scratch Monte Carlo derivatives-pricing and stochastic-volatility
engine.

The package is the plumbing (models, payoffs, estimator, Greeks, calibration); the
notebooks are the research. Correctness is provable against closed-form
mathematics — run :func:`qmc.self_test` before trusting any number.

Typical use::

    from qmc import get_rng, self_test
    from qmc.analytic import bs_price
    from qmc.processes import simulate_gbm
    from qmc.payoffs import european
    from qmc.engine import mc_price

    self_test()
    rng = get_rng()
    paths = simulate_gbm(100, 0.05, 0.0, 0.2, 1.0, 1, 100_000, rng)
    mc_price(paths, european("call", 100), r=0.05, T=1.0)   # ~ bs_price(...)
"""
from __future__ import annotations

from . import config, analytic, processes, payoffs, engine, greeks
from .config import get_rng, rng, SEED, DEFAULT_R, DEFAULT_Q, CACHE
from .plotting import apply_style, CLR
from .selftest import self_test

__all__ = [
    "config", "analytic", "processes", "payoffs", "engine", "greeks",
    "get_rng", "rng", "SEED", "DEFAULT_R", "DEFAULT_Q", "CACHE",
    "apply_style", "CLR", "self_test",
]
