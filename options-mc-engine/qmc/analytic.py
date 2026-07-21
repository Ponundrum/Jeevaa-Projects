"""Closed-form Black–Scholes prices and Greeks — the ground truth.

The whole project earns trust by matching Monte Carlo output to the exact
formulas in this module (the way the sibling project asserts ``self_test`` before
reporting a number). Two exact benchmarks live here: vanilla European options,
and the **discretely-monitored geometric-average Asian** option, whose average is
still lognormal and therefore has a closed form — giving an exact check on the
path-dependent machinery, not just the terminal-value one.

Conventions: continuous compounding; ``r`` risk-free, ``q`` dividend yield; time
``T`` in years; ``sigma`` annualised. ``kind`` is ``"call"`` or ``"put"``.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


def _d1_d2(S, K, T, r, sigma, q):
    S, K, T, sigma = map(np.asarray, (S, K, T, sigma))
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return d1, d2


def bs_price(S, K, T, r, sigma, q=0.0, kind="call"):
    """Black–Scholes price of a European option."""
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    df_r, df_q = np.exp(-r * T), np.exp(-q * T)
    if kind == "call":
        return S * df_q * norm.cdf(d1) - K * df_r * norm.cdf(d2)
    elif kind == "put":
        return K * df_r * norm.cdf(-d2) - S * df_q * norm.cdf(-d1)
    raise ValueError("kind must be 'call' or 'put'")


def bs_greeks(S, K, T, r, sigma, q=0.0, kind="call"):
    """Closed-form delta, gamma, vega, theta, rho. Vega/gamma are per unit vol/
    spot; theta is per year; rho per unit rate. Returns a dict."""
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    df_r, df_q = np.exp(-r * T), np.exp(-q * T)
    pdf = norm.pdf(d1)
    sqrtT = np.sqrt(T)
    gamma = df_q * pdf / (S * sigma * sqrtT)
    vega = S * df_q * pdf * sqrtT
    if kind == "call":
        delta = df_q * norm.cdf(d1)
        theta = (-S * df_q * pdf * sigma / (2 * sqrtT)
                 - r * K * df_r * norm.cdf(d2) + q * S * df_q * norm.cdf(d1))
        rho = K * T * df_r * norm.cdf(d2)
    else:
        delta = -df_q * norm.cdf(-d1)
        theta = (-S * df_q * pdf * sigma / (2 * sqrtT)
                 + r * K * df_r * norm.cdf(-d2) - q * S * df_q * norm.cdf(-d1))
        rho = -K * T * df_r * norm.cdf(-d2)
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


def bs_implied_vol(price, S, K, T, r, q=0.0, kind="call", tol=1e-8):
    """Invert Black–Scholes for the implied volatility via Brent's method with
    robust bracketing. Returns NaN if the price is outside the no-arbitrage bounds."""
    df_r, df_q = np.exp(-r * T), np.exp(-q * T)
    intrinsic = (max(S * df_q - K * df_r, 0.0) if kind == "call"
                 else max(K * df_r - S * df_q, 0.0))
    upper = S * df_q if kind == "call" else K * df_r          # price bound as sigma -> inf
    if price < intrinsic - tol or price > upper + tol:
        return np.nan
    f = lambda vol: bs_price(S, K, T, r, vol, q, kind) - price
    lo, hi = 1e-9, 5.0
    if f(lo) > 0:                                             # price below intrinsic numerically
        return np.nan
    while f(hi) < 0 and hi < 50:                              # widen bracket for deep ITM / long T
        hi *= 2
    return brentq(f, lo, hi, xtol=tol, rtol=tol)


def put_call_parity_gap(call, put, S, K, T, r, q=0.0):
    """Residual of put–call parity ``C - P - (S e^{-qT} - K e^{-rT})``; ~0 for
    consistent prices. Used as a test on both closed-form and MC prices."""
    return call - put - (S * np.exp(-q * T) - K * np.exp(-r * T))


def geometric_asian_price(S, K, T, r, sigma, q=0.0, n_steps=50, kind="call"):
    """Closed form for a **discretely-monitored geometric-average** Asian option.

    The average is taken over ``n_steps`` equally-spaced dates ``t_i = i·T/n_steps``
    (i = 1..n_steps), matching the Monte Carlo monitoring exactly. The geometric
    average of lognormals is lognormal, so ``ln G`` is Gaussian with a mean and
    variance computed directly from the GBM covariance ``Cov(W_{t_i}, W_{t_j}) =
    min(t_i, t_j)``; the option is then priced by the lognormal expectation. This
    is an EXACT benchmark for the path-dependent MC estimator.
    """
    dt = T / n_steps
    t = np.arange(1, n_steps + 1) * dt
    mu = np.log(S) + (r - q - 0.5 * sigma ** 2) * t.mean()             # E[ln G]
    cov_sum = np.minimum.outer(t, t).sum()                            # ΣΣ min(t_i, t_j)
    var = sigma ** 2 * cov_sum / n_steps ** 2                          # Var[ln G]
    s = np.sqrt(var)
    m = mu                                                            # ln G ~ Normal(m, s^2)
    d1 = (m - np.log(K) + var) / s
    d2 = d1 - s
    fwd = np.exp(m + 0.5 * var)                                        # E[G]
    if kind == "call":
        return np.exp(-r * T) * (fwd * norm.cdf(d1) - K * norm.cdf(d2))
    elif kind == "put":
        return np.exp(-r * T) * (K * norm.cdf(-d2) - fwd * norm.cdf(-d1))
    raise ValueError("kind must be 'call' or 'put'")
