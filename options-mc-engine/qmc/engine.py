"""The Monte Carlo pricing engine: turn simulated paths + a payoff into a price,
with an honest standard error and confidence interval, optional variance
reduction, and a convergence helper.

Trust model (mirrors the sibling project's ``self_test`` ethos): the engine
reports a standard error and a CI with every price, so "MC agrees with Black-
Scholes" is always a statement about *how many standard errors apart* they are —
never an eyeballed match.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MCResult:
    price: float
    std_error: float
    ci95: tuple
    n_paths: int

    def __repr__(self):
        return (f"MCResult(price={self.price:.6f}, se={self.std_error:.2e}, "
                f"ci95=[{self.ci95[0]:.6f}, {self.ci95[1]:.6f}], n={self.n_paths})")


def mc_price(paths, payoff, r, T, control=None):
    """Discounted Monte Carlo price of ``payoff`` on ``paths``.

    ``control`` optionally applies a control variate as ``(control_fn, mean)``,
    where ``control_fn(paths)`` is an undiscounted control payoff whose *discounted*
    expectation ``mean`` is known exactly (e.g. the geometric-Asian closed form as a
    control for the arithmetic Asian, or the discounted terminal price for a
    European). The optimal coefficient is estimated from the sample."""
    disc = np.exp(-r * T)
    y = disc * payoff(paths)
    if control is not None:
        control_fn, control_mean = control
        x = disc * control_fn(paths)
        beta = np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1)
        y = y - beta * (x - control_mean)
    n = len(y)
    price = float(y.mean())
    se = float(y.std(ddof=1) / np.sqrt(n))
    return MCResult(price, se, (price - 1.96 * se, price + 1.96 * se), n)


def variance_reduction_factor(paths, payoff, r, T, control):
    """Ratio of plain-MC variance to control-variate variance at the same paths —
    how many times more paths the plain estimator would need for the same error."""
    disc = np.exp(-r * T)
    y = disc * payoff(paths)
    x = disc * control[0](paths)
    beta = np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1)
    y_cv = y - beta * (x - control[1])
    return float(np.var(y, ddof=1) / np.var(y_cv, ddof=1))


def convergence(sim_fn, payoff, r, T, Ns, true_price, rng, n_reps=40):
    """RMSE of the MC price vs a known ``true_price`` at a ladder of path counts.

    ``sim_fn(n_paths, rng)`` simulates the paths. Returns ``(Ns, rmse)``; the
    log-log slope of ``rmse`` vs ``Ns`` should be about -1/2 (Monte Carlo's
    ``O(N^{-1/2})`` rate)."""
    Ns = np.asarray(Ns)
    rmse = np.empty(len(Ns), dtype=float)
    for i, N in enumerate(Ns):
        errs = [mc_price(sim_fn(int(N), rng), payoff, r, T).price - true_price for _ in range(n_reps)]
        rmse[i] = np.sqrt(np.mean(np.square(errs)))
    return Ns, rmse


def convergence_slope(Ns, rmse):
    """Least-squares slope of log(rmse) on log(Ns) — expected near -0.5."""
    return float(np.polyfit(np.log(Ns), np.log(rmse), 1)[0])


BGK_BETA = 0.5826    # = -zeta(1/2)/sqrt(2*pi), the Broadie-Glasserman-Kou constant


def bgk_barrier_shift(B, sigma, dt, up):
    """Broadie-Glasserman-Kou continuity correction: to price a CONTINUOUSLY-
    monitored barrier with a DISCRETELY-monitored (m-step) simulation, monitor
    against a barrier shifted toward the spot by ``exp(±BGK_BETA sigma sqrt(dt))``
    (down barriers up, up barriers down). This removes most of the ``O(1/sqrt(m))``
    discrete-monitoring bias."""
    return B * np.exp((-1 if up else 1) * BGK_BETA * sigma * np.sqrt(dt))
