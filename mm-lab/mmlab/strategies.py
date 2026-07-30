"""The quoting strategies. Each is a callable ``strategy(S, q, t) -> (delta_b,
delta_a)`` giving the half-spread to post below the mid (bid) and above it (ask),
vectorised over paths, so it drops straight into :func:`mmlab.simulate.run`.

Two strategies carry the project:

- :class:`Naive` — a constant half-spread on both sides, inventory ignored. The
  strawman. It *thinks* it earns the spread on every round trip; Layer 3 shows why
  it doesn't.
- :class:`AvellanedaStoikov` — quotes skewed around a **reservation price** that
  leans away from inventory, with an optimal spread built from two interpretable
  terms (exposed separately below because an interviewer will ask about each).

The closed-form pieces are module-level functions so the self-test can check them
analytically (the ``gamma -> 0`` limit, the zero-inventory reservation price)
without running a simulation.
"""
from __future__ import annotations

import numpy as np


def reservation_price(S, q, gamma, sigma, tau):
    """Avellaneda-Stoikov reservation price ``r = S - q*gamma*sigma^2*tau``: the
    inventory-adjusted value the maker quotes around. Above the mid when short,
    below it when long — it leans the maker's quotes toward flattening. Equals the
    mid exactly when ``q = 0``."""
    return S - q * gamma * sigma ** 2 * tau


def inventory_risk_term(gamma, sigma, tau):
    """First half-spread term ``gamma*sigma^2*(T-t)`` — the **inventory risk
    premium**. Grows with volatility, risk aversion, and time left to be caught
    holding; vanishes at ``t = T``. This is the part that makes the spread
    time-dependent."""
    return gamma * sigma ** 2 * tau


def edge_term(gamma, kappa):
    """Second half-spread term ``(2/gamma)*ln(1 + gamma/kappa)`` — the **fill/edge
    tradeoff**. Constant in time: the spread you need even with zero inventory
    risk, because filling at the mid earns nothing. As ``gamma -> 0`` it tends to
    ``2/kappa`` (see the self-test); the ``gamma == 0`` case is handled as that
    limit."""
    gamma = np.asarray(gamma, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        val = (2.0 / gamma) * np.log1p(gamma / kappa)
    return np.where(np.abs(gamma) < 1e-12, 2.0 / kappa, val)


def optimal_spread(gamma, sigma, kappa, tau):
    """Total optimal spread ``delta*`` = inventory-risk term + edge term."""
    return inventory_risk_term(gamma, sigma, tau) + edge_term(gamma, kappa)


class Naive:
    """Constant half-spread ``delta`` on both sides; inventory-blind."""

    def __init__(self, half_spread):
        self.half_spread = float(half_spread)

    def __call__(self, S, q, t):
        d = np.full(np.shape(S), self.half_spread, dtype=float)
        return d, d


class AvellanedaStoikov:
    """Avellaneda-Stoikov quoting.

    Quotes are ``bid = r - delta*/2``, ``ask = r + delta*/2`` around the
    reservation price ``r``, which turns into the two half-spreads
    ``delta_b = q*gamma*sigma^2*tau + delta*/2`` and ``delta_a =
    -q*gamma*sigma^2*tau + delta*/2`` — wider on the side that would grow the
    position, tighter on the side that would flatten it.

    In a 24/7 market the ``tau = T - t`` clock has no natural end, so
    ``frozen_horizon`` (default: on, from ``config.RISK_HORIZON``) replaces it with
    a constant "time until I can flatten". Pass ``frozen_horizon=None`` to use the
    genuine countdown ``T - t`` (the textbook finite-horizon form used for the
    ``t = T`` checks).
    """

    def __init__(self, gamma, sigma, kappa, T, frozen_horizon="config"):
        self.gamma = float(gamma)
        self.sigma = float(sigma)
        self.kappa = float(kappa)
        self.T = float(T)
        if frozen_horizon == "config":
            from .config import RISK_HORIZON
            frozen_horizon = RISK_HORIZON
        self.frozen_horizon = frozen_horizon

    def tau(self, t):
        if self.frozen_horizon is not None:
            return self.frozen_horizon
        return np.maximum(self.T - t, 0.0)

    def __call__(self, S, q, t):
        tau = self.tau(t)
        skew = q * self.gamma * self.sigma ** 2 * tau                 # reservation-price shift
        half = 0.5 * optimal_spread(self.gamma, self.sigma, self.kappa, tau)
        db = skew + half                                              # S - bid
        da = -skew + half                                             # ask - S
        return np.broadcast_to(db, np.shape(S)).astype(float), np.broadcast_to(da, np.shape(S)).astype(float)
