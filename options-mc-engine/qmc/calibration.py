"""Calibrate the stochastic-vol models to a market implied-vol surface by least
squares, pricing with the Monte Carlo engine inside the objective.

Two things keep MC-in-the-loop tractable: (1) **common random numbers** — every
objective evaluation reuses the same seed, so the objective is a smooth,
deterministic function of the parameters and the optimiser is not fighting Monte
Carlo noise; (2) simulate once per maturity and price all strikes off those paths.
The fit is reported as an IV-space RMSE (in vol points) with the parameter
estimates. It is a snapshot calibration, not a backtest.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from .analytic import bs_implied_vol
from .config import get_rng
from . import payoffs, processes
from .engine import mc_price


def _model_ivs(sim_by_T, market, S, r):
    """Given a dict ``{T: paths}`` of simulated price paths, price each market quote
    by MC and invert to implied vol. Returns an array aligned with ``market.index``."""
    ivs = np.full(len(market), np.nan)
    for i, (T, K, kind, q) in enumerate(zip(market["T"], market["K"], market["kind"], market["q"])):
        paths = sim_by_T[T]
        price = mc_price(paths, payoffs.european(kind, K), r, T).price
        ivs[i] = bs_implied_vol(price, S, K, T, r, q, kind)
    return ivs


def calibrate_heston(market, S, r, n_paths=40_000, steps_per_year=100, seed=123,
                     max_nfev=60, verbose=True):
    """Fit Heston ``(v0, kappa, theta, xi, rho)`` to a market IV surface (columns
    ``T, K, kind, q, iv``). Returns ``(params_dict, rmse_volpts, model_iv)``."""
    Ts = sorted(market["T"].unique())
    steps = {T: max(20, int(T * steps_per_year)) for T in Ts}
    q_by_T = market.groupby("T")["q"].first().to_dict()      # each expiry priced on its OWN forward (P1.3)
    mkt_iv = market["iv"].values

    def simulate(params):
        v0, kappa, theta, xi, rho = params
        g = get_rng(seed)                                    # common random numbers -> smooth objective
        return {T: processes.simulate_heston(S, v0, kappa, theta, xi, rho, r, q_by_T[T],
                                             T, steps[T], n_paths, g) for T in Ts}

    def resid(params):
        model_iv = _model_ivs(simulate(params), market, S, r)
        d = np.nan_to_num(model_iv - mkt_iv, nan=1.0)        # penalise NaNs (arb-violating params)
        return d

    x0 = [0.03, 2.0, 0.03, 0.4, -0.6]
    lb = [1e-4, 0.1, 1e-4, 0.05, -0.98]
    ub = [0.5, 15.0, 0.5, 3.0, 0.0]
    sol = least_squares(resid, x0, bounds=(lb, ub), max_nfev=max_nfev, verbose=0)
    params = dict(zip(["v0", "kappa", "theta", "xi", "rho"], sol.x))
    # Feller condition 2*kappa*theta >= xi^2: when the ratio < 1 the variance hits zero
    # often (the QE scheme handles it, but it's a regime worth flagging).
    params["feller_ratio"] = 2 * params["kappa"] * params["theta"] / params["xi"] ** 2
    model_iv = _model_ivs(simulate(sol.x), market, S, r)
    rmse = float(np.sqrt(np.nanmean((model_iv - mkt_iv) ** 2)))
    if verbose:
        feller = params["feller_ratio"]
        print(f"Heston calibrated: {', '.join(f'{k}={v:.3f}' for k, v in params.items() if k != 'feller_ratio')} | "
              f"IV RMSE {rmse * 100:.2f} vol points | Feller 2kt/xi^2 = {feller:.2f} "
              f"({'satisfied' if feller >= 1 else 'VIOLATED — variance touches zero, QE handles it'})")
    return params, rmse, model_iv


def calibrate_rough_bergomi(market, S, r, xi0=None, n_paths=20_000, steps_per_year=200,
                            seed=321, max_nfev=40, verbose=True):
    """Fit rough Bergomi ``(eta, rho, H)`` to (typically short-dated) market slices;
    ``xi0`` (the forward variance level) defaults to the shortest-maturity ATM total
    variance. Returns ``(params_dict, rmse_volpts, model_iv)``. Heavier than Heston
    (the fractional driver needs a Cholesky per maturity), so use a small grid."""
    Ts = sorted(market["T"].unique())
    steps = {T: max(30, int(T * steps_per_year)) for T in Ts}
    q_by_T = market.groupby("T")["q"].first().to_dict()      # each expiry priced on its OWN forward (P1.3)
    mkt_iv = market["iv"].values
    if xi0 is None:
        near = market[market["T"] == Ts[0]]
        xi0 = float(near.loc[near["k"].abs().idxmin(), "iv"] ** 2)

    def simulate(params):
        eta, rho, H = params
        g = get_rng(seed)
        return {T: processes.simulate_rough_bergomi(S, xi0, eta, rho, H, r, q_by_T[T],
                                                    T, steps[T], n_paths, g) for T in Ts}

    def resid(params):
        model_iv = _model_ivs(simulate(params), market, S, r)
        return np.nan_to_num(model_iv - mkt_iv, nan=1.0)

    x0 = [1.5, -0.6, 0.15]
    lb = [0.5, -0.98, 0.05]
    ub = [4.0, 0.0, 0.49]
    sol = least_squares(resid, x0, bounds=(lb, ub), max_nfev=max_nfev, verbose=0)
    params = dict(zip(["eta", "rho", "H"], sol.x), xi0=xi0)
    model_iv = _model_ivs(simulate(sol.x), market, S, r)
    rmse = float(np.sqrt(np.nanmean((model_iv - mkt_iv) ** 2)))
    if verbose:
        print(f"Rough Bergomi calibrated: xi0={xi0:.4f}, "
              f"{', '.join(f'{k}={v:.3f}' for k, v in params.items() if k != 'xi0')} | "
              f"IV RMSE {rmse * 100:.2f} vol points")
    return params, rmse, model_iv
