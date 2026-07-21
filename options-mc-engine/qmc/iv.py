"""Implied-volatility surface tools: invert a chain to IVs, fit the SVI / SSVI
parameterisation per maturity / across the surface, and check the fitted surface
for static arbitrage (calendar-spread and butterfly). A fitted surface that is
arbitrage-free is a real result to state, not a given.

SVI (Gatheral's raw parameterisation) writes the **total implied variance**
``w(k) = sigma_BS(k)^2 * T`` as a function of log-moneyness ``k = ln(K/F)``:

    w(k) = a + b ( rho (k - m) + sqrt((k - m)^2 + s^2) )

with ``b >= 0``, ``|rho| < 1``, ``s > 0``, and ``a + b s sqrt(1 - rho^2) >= 0``
(so ``w >= 0``).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from .analytic import bs_implied_vol


# ---------------------------------------------------------------------------
# Implied-vol inversion of a chain
# ---------------------------------------------------------------------------
def implied_vol_surface(chain, S, r, q=0.0):
    """Add an ``iv`` column to a cleaned option chain. ``chain`` needs columns
    ``T`` (years), ``K`` (strike), ``price`` (mid), ``kind`` ('call'/'put'). Also
    returns log-moneyness ``k = ln(K/F)`` and total variance ``w = iv^2 T``."""
    out = chain.copy()
    F = S * np.exp((r - q) * out["T"])
    out["k"] = np.log(out["K"] / F)
    out["iv"] = [bs_implied_vol(p, S, K, T, r, q, kind)
                 for p, K, T, kind in zip(out["price"], out["K"], out["T"], out["kind"])]
    out["w"] = out["iv"] ** 2 * out["T"]
    return out.dropna(subset=["iv"])


# ---------------------------------------------------------------------------
# SVI (per-maturity slice)
# ---------------------------------------------------------------------------
def svi_total_variance(k, params):
    """Raw-SVI total variance ``w(k)``. ``params = (a, b, rho, m, s)``."""
    a, b, rho, m, s = params
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + s ** 2))


def fit_svi(k, w, weights=None):
    """Least-squares fit of raw-SVI to one maturity slice ``(k, w)``. Returns the
    5-tuple ``(a, b, rho, m, s)``. Bounds enforce ``b>=0, |rho|<1, s>0``."""
    k = np.asarray(k, float)
    w = np.asarray(w, float)
    weights = np.ones_like(w) if weights is None else np.asarray(weights, float)
    p0 = [max(w.min(), 1e-4), 0.1, -0.3, 0.0, 0.1]
    lb = [-np.inf, 0.0, -0.999, -np.inf, 1e-6]
    ub = [np.inf, np.inf, 0.999, np.inf, np.inf]
    resid = lambda p: np.sqrt(weights) * (svi_total_variance(k, p) - w)
    sol = least_squares(resid, p0, bounds=(lb, ub), method="trf", max_nfev=5000)
    return tuple(sol.x)


def svi_butterfly_g(k, params):
    """Gatheral's ``g(k)``; ``g(k) >= 0`` for all ``k`` is the no-butterfly-
    arbitrage (non-negative risk-neutral density) condition for a slice."""
    a, b, rho, m, s = params
    w = svi_total_variance(k, params)
    wp = b * (rho + (k - m) / np.sqrt((k - m) ** 2 + s ** 2))                       # w'(k)
    wpp = b * s ** 2 / ((k - m) ** 2 + s ** 2) ** 1.5                               # w''(k)
    return (1 - k * wp / (2 * w)) ** 2 - (wp ** 2 / 4) * (1 / w + 0.25) + wpp / 2


# ---------------------------------------------------------------------------
# SSVI (whole surface, Gatheral-Jacquier) — power-law phi
# ---------------------------------------------------------------------------
def ssvi_total_variance(k, theta, rho, eta, gamma):
    """SSVI total variance for ATM total variance ``theta`` and a power-law
    ``phi(theta) = eta / theta^gamma``."""
    phi = eta / theta ** gamma
    return 0.5 * theta * (1 + rho * phi * k + np.sqrt((phi * k + rho) ** 2 + (1 - rho ** 2)))


def fit_ssvi(surface):
    """Fit SSVI ``(rho, eta, gamma)`` jointly across maturities. ``surface`` has
    columns ``T, k, w``; the ATM total variance ``theta(T)`` is read off per
    maturity by interpolating ``w`` to ``k=0``. Returns ``(rho, eta, gamma, thetas)``."""
    thetas = {}
    for T, g in surface.groupby("T"):
        g = g.sort_values("k")
        thetas[T] = float(np.interp(0.0, g["k"], g["w"]))
    k = surface["k"].values
    w = surface["w"].values
    th = surface["T"].map(thetas).values
    resid = lambda p: ssvi_total_variance(k, th, np.clip(p[0], -0.999, 0.999), p[1], p[2]) - w
    sol = least_squares(resid, [-0.5, 1.0, 0.4], bounds=([-0.999, 1e-3, 0.0], [0.999, 20, 0.5]), max_nfev=5000)
    return sol.x[0], sol.x[1], sol.x[2], thetas


# ---------------------------------------------------------------------------
# Static-arbitrage checks
# ---------------------------------------------------------------------------
def check_no_arbitrage(slice_params, ks=None):
    """Check a set of fitted SVI slices for static arbitrage.

    ``slice_params`` maps maturity ``T -> (a,b,rho,m,s)``. Returns a dict with
    ``butterfly`` (min ``g(k)`` per slice; negative = arbitrage) and ``calendar``
    (whether total variance is non-decreasing in ``T`` at every tested ``k``)."""
    ks = np.linspace(-1.0, 1.0, 201) if ks is None else np.asarray(ks)
    Ts = sorted(slice_params)
    butterfly = {T: float(svi_butterfly_g(ks, slice_params[T]).min()) for T in Ts}
    calendar_ok = True
    calendar_violations = 0
    for T1, T2 in zip(Ts[:-1], Ts[1:]):
        w1 = svi_total_variance(ks, slice_params[T1])
        w2 = svi_total_variance(ks, slice_params[T2])
        bad = int((w2 < w1 - 1e-8).sum())
        calendar_violations += bad
        calendar_ok &= bad == 0
    return {"butterfly_min_g": butterfly,
            "butterfly_ok": all(v >= -1e-6 for v in butterfly.values()),
            "calendar_ok": calendar_ok,
            "calendar_violations": calendar_violations}
