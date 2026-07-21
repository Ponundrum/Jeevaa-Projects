"""A Crank-Nicolson finite-difference PDE pricer — a second, independent numerical
method. It (a) validates European prices against Black-Scholes a different way than
Monte Carlo, and (b) unlocks the one payoff class Monte Carlo cannot price cleanly:
the **American option**, via projected successive over-relaxation (PSOR) on the
early-exercise linear-complementarity problem.

The Black-Scholes PDE ``V_t + 0.5 sigma^2 S^2 V_SS + (r-q) S V_S - r V = 0`` is
solved backward in time on a log-spot grid ``x = ln S`` (constant coefficients,
which makes the scheme clean and second-order accurate in space and time).
"""
from __future__ import annotations

import numpy as np


def _thomas(a, b, c, d):
    """Solve a tridiagonal system (sub-diagonal a, diagonal b, super-diagonal c)."""
    n = len(b)
    cp = np.empty(n)
    dp = np.empty(n)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):
        m = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / m
        dp[i] = (d[i] - a[i] * dp[i - 1]) / m
    x = np.empty(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def crank_nicolson(S0, K, T, r, sigma, q=0.0, kind="call", american=False,
                   n_space=400, n_time=400, x_width=6.0, omega=1.4, psor_tol=1e-9):
    """Price a European or American option by Crank-Nicolson on a log-spot grid.

    ``x_width`` sets the grid half-width in standard deviations of log-return.
    American options solve the early-exercise complementarity by projected SOR at
    each time step. Returns the price at ``S0`` (linearly interpolated on the grid)."""
    xc = np.log(S0)
    half = x_width * sigma * np.sqrt(T) + abs((r - q - 0.5 * sigma ** 2) * T)
    x = np.linspace(xc - half, xc + half, n_space + 1)
    dx = x[1] - x[0]
    dt = T / n_time
    S = np.exp(x)
    payoff = np.maximum(S - K, 0.0) if kind == "call" else np.maximum(K - S, 0.0)
    V = payoff.copy()

    # Constant-coefficient operator on x: 0.5 sigma^2 V_xx + (r-q-0.5 sigma^2) V_x - r V
    nu = r - q - 0.5 * sigma ** 2
    A = 0.5 * sigma ** 2 / dx ** 2 - nu / (2 * dx)     # sub-diagonal coefficient
    B = -sigma ** 2 / dx ** 2 - r                       # diagonal coefficient
    C = 0.5 * sigma ** 2 / dx ** 2 + nu / (2 * dx)     # super-diagonal coefficient

    n = n_space - 1                                    # interior nodes
    lower = -0.5 * dt * A * np.ones(n)
    diag = (1 - 0.5 * dt * B) * np.ones(n)
    upper = -0.5 * dt * C * np.ones(n)
    rl = 0.5 * dt * A
    rd = 1 + 0.5 * dt * B
    ru = 0.5 * dt * C

    for step in range(n_time):
        tau = (step + 1) * dt                          # time-to-maturity of the new layer
        # boundary values (Dirichlet) from the exact continuation/limit behaviour
        if kind == "call":
            lo = 0.0
            hi = np.exp(x[-1]) * np.exp(-q * tau) - K * np.exp(-r * tau)
        else:
            lo = K * np.exp(-r * tau) - np.exp(x[0]) * np.exp(-q * tau)
            hi = 0.0
        rhs = rl * V[:-2] + rd * V[1:-1] + ru * V[2:]   # explicit half-step (uses OLD boundaries)
        if not american:
            rt = rhs.copy()
            rt[0] -= lower[0] * lo                       # move NEW-boundary implicit terms to the RHS
            rt[-1] -= upper[-1] * hi
            V[1:-1] = _thomas(lower, diag, upper, rt)
        else:
            # projected SOR against the intrinsic value (early exercise)
            g = payoff[1:-1]
            w = V[1:-1].copy()
            for _ in range(10000):
                w_old = w.copy()
                for i in range(n):
                    lo_i = lower[i] * (w[i - 1] if i > 0 else lo)
                    up_i = upper[i] * (w[i + 1] if i < n - 1 else hi)
                    y = (rhs[i] - lo_i - up_i) / diag[i]
                    w[i] = max(g[i], w[i] + omega * (y - w[i]))
                if np.max(np.abs(w - w_old)) < psor_tol:
                    break
            V[1:-1] = w
        V[0], V[-1] = lo, hi
    return float(np.interp(xc, x, V))
