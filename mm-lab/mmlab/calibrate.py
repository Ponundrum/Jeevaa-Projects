"""Estimating the three simulator parameters — ``sigma``, ``A``, ``kappa`` — from
data. Layer 3a.

The intensity fit is the interesting one. A limit order resting at distance
``delta`` from the mid is filled when a same-side aggressor trade *penetrates* to
at least that depth, so the fill rate at distance ``delta`` is simply the arrival
rate of trades reaching depth ``>= delta``::

    lambda(delta) = (# aggressor trades with penetration >= delta) / duration

Fitting ``lambda(delta) = A e^{-kappa delta}`` is then a straight-line regression
of ``log lambda`` on ``delta``: slope ``-kappa``, intercept ``log A``. If the points
are not close to a line in log-space, the exponential model is wrong for the asset
and the notebook says so (rather than reporting a fit anyway).

Before this is ever pointed at real data, :func:`mmlab.selftest` validates it by
recovering known ``A``/``kappa`` from a :func:`synthetic_depth_tape` — the same
ground-truth discipline the options project uses.
"""
from __future__ import annotations

import numpy as np


def realized_sigma(mid, dt):
    """Realized volatility of the mid in **price units per sqrt(second)** — the
    same ``sigma`` the simulator consumes. ``mid`` is sampled on a uniform grid of
    spacing ``dt`` seconds; ``sigma = std(diff(mid)) / sqrt(dt)``. Keeping the units
    identical to the simulator is the single most important unit discipline in the
    project."""
    d = np.diff(np.asarray(mid, dtype=float))
    return float(np.std(d, ddof=1) / np.sqrt(dt))


def fit_intensity(depths, duration, deltas, min_count=20):
    """Fit ``lambda(delta) = A e^{-kappa delta}`` from one side's trade penetration
    ``depths`` (price units) observed over ``duration`` seconds.

    Returns ``(A, kappa, deltas_used, lambdas)`` — the last two for plotting the fit
    with its points. Buckets with fewer than ``min_count`` trades are dropped so the
    log-regression is not dominated by noisy deep-book estimates."""
    depths = np.asarray(depths, dtype=float)
    deltas = np.asarray(deltas, dtype=float)
    counts = np.array([(depths >= d).sum() for d in deltas])
    lam = counts / duration
    keep = counts >= min_count
    if keep.sum() < 2:
        raise ValueError("too few populated buckets to fit an intensity curve")
    x = deltas[keep]
    y = np.log(lam[keep])
    slope, intercept = np.polyfit(x, y, 1)
    return float(np.exp(intercept)), float(-slope), x, lam[keep]


def trade_depths(price, mid, is_buyer_maker):
    """Split real aggTrades into the two penetration series the fit needs.

    ``is_buyer_maker`` is Binance's flag: ``True`` means the resting order was a bid
    and the aggressor was a **seller** (a market sell), so that trade could have
    filled a resting *bid* at depth ``mid - price``. ``False`` is a market buy that
    could have filled a resting *ask* at depth ``price - mid``. Returns
    ``(bid_side_depths, ask_side_depths)``, each already restricted to non-negative
    penetrations."""
    price = np.asarray(price, dtype=float)
    mid = np.asarray(mid, dtype=float)
    ibm = np.asarray(is_buyer_maker, dtype=bool)
    bid_depths = (mid - price)[ibm]                 # market sells hit resting bids
    ask_depths = (price - mid)[~ibm]                # market buys lift resting asks
    return bid_depths[bid_depths >= 0], ask_depths[ask_depths >= 0]


def synthetic_depth_tape(A, kappa, duration, rng):
    """Ground-truth tape for the recovery self-test: a Poisson number of trades
    (rate ``A``) whose penetration depths are ``Exponential(kappa)``, so that by
    construction ``P(depth >= delta) = e^{-kappa delta}`` and the fill rate is
    exactly ``A e^{-kappa delta}``. Recovering ``A``/``kappa`` from this validates
    :func:`fit_intensity`."""
    n = rng.poisson(A * duration)
    depths = rng.exponential(1.0 / kappa, size=n)
    return depths, duration


def default_delta_grid(depths, n=15, hi_quantile=0.85):
    """A sensible ``delta`` grid for :func:`fit_intensity`: linear from 0 to the
    ``hi_quantile`` of observed depths, so buckets stay populated."""
    hi = float(np.quantile(np.asarray(depths, dtype=float), hi_quantile))
    return np.linspace(0.0, hi, n)
