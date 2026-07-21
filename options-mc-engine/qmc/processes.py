"""Path simulators. Each returns an array of shape ``(n_paths, n_steps+1)`` whose
first column is ``S0``, and takes an explicit ``numpy.random.Generator`` so every
path is reproducible from the seed. Everything is vectorised — no Python loop over
paths.

- ``simulate_gbm``            : exact log-Euler geometric Brownian motion.
- ``simulate_heston``         : Heston stochastic vol via the Andersen QE scheme.
- ``simulate_rough_bergomi``  : rough Bergomi, with the fractional (Volterra)
  driver built by Cholesky decomposition of its exact covariance matrix.

Memory note: a path array is ``n_paths x (n_steps+1)`` floats, so fine-grid
path-dependent pricing (e.g. 2000 steps x 400k paths ~ 6 GB) can exhaust RAM —
chunk ``n_paths`` and average across batches if memory-bound.
"""
from __future__ import annotations

import numpy as np
from scipy.special import hyp2f1


def _draw_normals(n_paths, n_steps, rng, antithetic):
    """Standard normals of shape (n_paths, n_steps); if antithetic, the second
    half of the paths are the negatives of the first (variance reduction)."""
    if antithetic:
        half = (n_paths + 1) // 2
        Z = rng.standard_normal((half, n_steps))
        return np.concatenate([Z, -Z], axis=0)[:n_paths]
    return rng.standard_normal((n_paths, n_steps))


# ---------------------------------------------------------------------------
# Geometric Brownian motion (exact)
# ---------------------------------------------------------------------------
def simulate_gbm(S0, r, q, sigma, T, n_steps, n_paths, rng, antithetic=False):
    """Exact geometric Brownian motion under the risk-neutral measure. The
    log-increments are exactly Gaussian, so this has no discretisation bias for any
    ``n_steps``; ``n_steps`` only sets path resolution for path-dependent payoffs."""
    dt = T / n_steps
    Z = _draw_normals(n_paths, n_steps, rng, antithetic)
    logincr = (r - q - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
    logpaths = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(logincr, axis=1)], axis=1)
    return S0 * np.exp(logpaths)


def gbm_terminal(S0, r, q, sigma, T, n_paths, rng, antithetic=False):
    """Terminal ``S_T`` with the driving normals ``Z``, for pathwise / likelihood-
    ratio Greeks. Returns ``(S_T, Z)``."""
    Z = _draw_normals(n_paths, 1, rng, antithetic)[:, 0]
    ST = S0 * np.exp((r - q - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)
    return ST, Z


# ---------------------------------------------------------------------------
# Heston (Andersen Quadratic-Exponential scheme)
# ---------------------------------------------------------------------------
def simulate_heston(S0, v0, kappa, theta, xi, rho, r, q, T, n_steps, n_paths, rng, scheme="QE"):
    """Heston model, variance simulated with the Andersen (2008) Quadratic-
    Exponential (QE) scheme — stable and low-bias even for large vol-of-vol ``xi``
    where plain Euler needs many steps and still goes negative.

    ``dS = (r-q) S dt + sqrt(v) S dW_S``,
    ``dv = kappa (theta - v) dt + xi sqrt(v) dW_v``, ``corr(dW_S, dW_v) = rho``.
    ``scheme='euler'`` selects a documented full-truncation Euler fallback.
    """
    dt = T / n_steps
    logS = np.full(n_paths, np.log(S0))
    v = np.full(n_paths, float(v0))
    out = np.empty((n_paths, n_steps + 1))
    out[:, 0] = S0
    psi_c = 1.5
    emkt = np.exp(-kappa * dt)
    # Andersen's martingale-correct log-spot coefficients (central, gamma1=gamma2=1/2).
    g1 = g2 = 0.5
    K0 = -rho * kappa * theta * dt / xi
    K1 = g1 * dt * (kappa * rho / xi - 0.5) - rho / xi
    K2 = g2 * dt * (kappa * rho / xi - 0.5) + rho / xi
    K3 = g1 * dt * (1 - rho ** 2)
    K4 = g2 * dt * (1 - rho ** 2)
    for t in range(1, n_steps + 1):
        v_old = v
        if scheme == "QE":
            m = theta + (v_old - theta) * emkt                                   # E[v' | v]
            s2 = (v_old * xi ** 2 * emkt * (1 - emkt) / kappa
                  + theta * xi ** 2 * (1 - emkt) ** 2 / (2 * kappa))             # Var[v' | v]
            psi = s2 / np.maximum(m ** 2, 1e-300)
            v_new = np.empty(n_paths)
            lo = psi <= psi_c                                                    # quadratic branch
            inv = 2.0 / psi[lo]
            b2 = inv - 1 + np.sqrt(inv) * np.sqrt(np.maximum(inv - 1, 0.0))
            a = m[lo] / (1 + b2)
            v_new[lo] = a * (np.sqrt(b2) + rng.standard_normal(lo.sum())) ** 2
            hi = ~lo                                                             # exponential branch
            p = (psi[hi] - 1) / (psi[hi] + 1)
            beta = (1 - p) / np.maximum(m[hi], 1e-300)
            u = rng.random(hi.sum())
            v_new[hi] = np.where(u <= p, 0.0,
                                 np.log(np.maximum((1 - p) / np.maximum(1 - u, 1e-300), 1e-300)) / beta)
            v = np.maximum(v_new, 0.0)
            # Correlated, martingale-consistent log-spot update with an INDEPENDENT normal.
            Z = rng.standard_normal(n_paths)
            logS += (r - q) * dt + K0 + K1 * v_old + K2 * v + np.sqrt(np.maximum(K3 * v_old + K4 * v, 0.0)) * Z
        else:
            vp = np.maximum(v_old, 0.0)
            Zv = rng.standard_normal(n_paths)
            Zs = rho * Zv + np.sqrt(1 - rho ** 2) * rng.standard_normal(n_paths)
            logS += (r - q - 0.5 * vp) * dt + np.sqrt(vp * dt) * Zs
            v = v_old + kappa * (theta - vp) * dt + xi * np.sqrt(vp * dt) * Zv
        out[:, t] = np.exp(logS)
    return out


# ---------------------------------------------------------------------------
# Rough Bergomi (fractional driver via Cholesky of the exact covariance)
# ---------------------------------------------------------------------------
def _rbergomi_covariance(t, H):
    """Joint covariance of ``(Y_{t_1..t_n}, W_{t_1..t_n})`` on the grid ``t``, where
    ``Y`` is the Riemann-Liouville (Volterra) fractional Gaussian process
    ``Y_t = sqrt(2H) \\int_0^t (t-s)^{H-1/2} dW_s`` and ``W`` is the driving BM.

    * ``Cov(W_{t_i}, W_{t_j}) = min(t_i, t_j)``
    * ``Cov(Y_{t_i}, W_{t_j}) = sqrt(2H)/(H+1/2) * (t_i^{H+1/2} - max(t_i-t_j,0)^{H+1/2})``
    * ``Cov(Y_{s}, Y_{t}) = 2H/(H+1/2) * s^{H+1/2} t^{H-1/2} * 2F1(1/2-H, 1; H+3/2; s/t)``
      for ``s = min <= t = max`` (this closed form gives ``Var(Y_t)=t^{2H}`` exactly).
    """
    n = len(t)
    ti, tj = t[:, None], t[None, :]
    s = np.minimum(ti, tj)
    u = np.maximum(ti, tj)
    G_WW = np.minimum(ti, tj)
    c = np.sqrt(2 * H) / (H + 0.5)
    G_YW = c * (ti ** (H + 0.5) - np.maximum(ti - tj, 0.0) ** (H + 0.5))
    G_YY = (2 * H / (H + 0.5)) * s ** (H + 0.5) * u ** (H - 0.5) * hyp2f1(0.5 - H, 1.0, H + 1.5, s / u)
    Sigma = np.empty((2 * n, 2 * n))
    Sigma[:n, :n] = G_YY
    Sigma[:n, n:] = G_YW
    Sigma[n:, :n] = G_YW.T
    Sigma[n:, n:] = G_WW
    return Sigma


def simulate_rough_bergomi(S0, xi0, eta, rho, H, r, q, T, n_steps, n_paths, rng, return_v=False):
    """Rough Bergomi (Bayer-Friz-Gatheral 2016) — the standout model.

    Instantaneous variance ``v_t = xi0 * exp(eta Y_t - 0.5 eta^2 t^{2H})`` where
    ``Y`` is a rough (Hurst ``H<1/2``) Riemann-Liouville process. ``Y`` and the
    price Brownian motion ``W`` are jointly Gaussian on the grid, so we form their
    exact joint covariance ``Sigma`` and simulate by **Cholesky decomposition**
    ``Sigma = L Lᵀ``, drawing ``[Y; W] = L Z`` from i.i.d. normals — the discrete
    Karhunen-Loève / covariance-decomposition recipe applied to a rough process
    (exact on the grid; O(n^3) setup, O(n^2) per draw). The Bennedsen-Lunde-
    Pakkanen hybrid scheme is a faster O(n log n) alternative; the exact Cholesky
    version is used here for transparency and correctness.
    """
    dt = T / n_steps
    t = np.arange(1, n_steps + 1) * dt
    n = n_steps
    L = np.linalg.cholesky(_rbergomi_covariance(t, H) + 1e-14 * np.eye(2 * n))
    draws = (L @ rng.standard_normal((2 * n, n_paths))).T           # (n_paths, 2n)
    Y = np.concatenate([np.zeros((n_paths, 1)), draws[:, :n]], axis=1)   # prepend Y_0 = 0
    W = np.concatenate([np.zeros((n_paths, 1)), draws[:, n:]], axis=1)

    t_full = np.concatenate([[0.0], t])
    v = xi0 * np.exp(eta * Y - 0.5 * eta ** 2 * t_full[None, :] ** (2 * H))   # E[v_t] = xi0
    dW = np.diff(W, axis=1)                                          # price BM increments (var dt)
    dW_perp = np.sqrt(dt) * rng.standard_normal((n_paths, n))
    dZ = rho * dW + np.sqrt(1 - rho ** 2) * dW_perp                 # correlated price driver
    v_left = v[:, :-1]                                              # variance at each interval start
    logret = (r - q - 0.5 * v_left) * dt + np.sqrt(v_left) * dZ
    logS = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(logret, axis=1)], axis=1)
    S = S0 * np.exp(logS)
    return (S, v) if return_v else S
