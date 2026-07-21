"""Payoff factories. Each returns a pure function ``paths -> undiscounted payoff``
(one value per path), so the engine can price ANY payoff on ANY simulated process.

Path-dependent payoffs average / monitor over columns ``1:`` (the dates after
``t_0``), matching the closed-form geometric-Asian benchmark's monitoring grid.
``kind`` is ``"call"`` or ``"put"``; barriers take a level ``B`` and a ``style``.
"""
from __future__ import annotations

import numpy as np


def _vanilla(ST, K, kind):
    return np.maximum(ST - K, 0.0) if kind == "call" else np.maximum(K - ST, 0.0)


def european(kind, K):
    return lambda paths: _vanilla(paths[:, -1], K, kind)


def asian_arithmetic(kind, K):
    return lambda paths: _vanilla(paths[:, 1:].mean(axis=1), K, kind)


def asian_geometric(kind, K):
    return lambda paths: _vanilla(np.exp(np.log(paths[:, 1:]).mean(axis=1)), K, kind)


def barrier(kind, K, B, style):
    """Knock-in / knock-out barrier. ``style`` in {up-and-out, down-and-out,
    up-and-in, down-and-in}; monitored on the simulated grid (discrete)."""
    def f(paths):
        mx, mn = paths.max(axis=1), paths.min(axis=1)
        payoff = _vanilla(paths[:, -1], K, kind)
        if style == "up-and-out":
            alive = mx < B
        elif style == "down-and-out":
            alive = mn > B
        elif style == "up-and-in":
            alive = mx >= B
        elif style == "down-and-in":
            alive = mn <= B
        else:
            raise ValueError(f"unknown barrier style {style!r}")
        return payoff * alive
    return f


def lookback(kind):
    """Floating-strike lookback: call pays ``S_T - min``, put pays ``max - S_T``."""
    def f(paths):
        if kind == "call":
            return paths[:, -1] - paths.min(axis=1)
        return paths.max(axis=1) - paths[:, -1]
    return f


def digital(kind, K, cash=1.0):
    """Cash-or-nothing digital paying ``cash`` if ITM at expiry."""
    def f(paths):
        ST = paths[:, -1]
        return cash * ((ST > K) if kind == "call" else (ST < K)).astype(float)
    return f
