"""The market simulator: an arithmetic-Brownian mid and a Poisson fill engine,
matching the Avellaneda-Stoikov assumptions *exactly* so the closed-form quoting
solution is ground truth against it. This is the whole of Layer 1 — deliberately
tiny, so it can be drawn on a whiteboard.

Conventions (see ``config`` for the full statement): time in seconds, price in
USDT. A strategy is any callable ``quote(S, q, t) -> (delta_b, delta_a)`` returning
the half-spread to post below the mid (bid) and above it (ask), in price units,
vectorised over the path dimension.

Per step ``dt`` each side fills independently as a Poisson process with intensity
``lambda(delta) = A * exp(-kappa * delta)``; we use the **exact** per-step fill
probability ``1 - exp(-lambda*dt)`` (not the ``lambda*dt`` small-step approximation).
One unit per fill.

The optional ``adverse`` argument implements Layer 3c: after a fill the mid is
nudged *against* the market maker by that many price units (a passive buy is
followed by a fall, a passive sell by a rise) — the empirically measured adverse
selection, fed back into the sim. ``adverse_steps`` controls *how* that move is
injected: as a single jump at the fill step (``1``, the default) or spread evenly
over the next ``adverse_steps`` steps. This is a genuine modelling choice — a maker
that can flatten before the adverse move fully arrives keeps more of its PnL — so
the notebook reports a **sensitivity** over it rather than one cell of the table.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SimResult:
    """One batch of ``n_paths`` independent simulations. Arrays are ``(n_paths,
    n_steps+1)`` for time series and ``(n_paths,)`` for per-path scalars.

    ``inventory[:, t]`` is the inventory *carried over the interval* ``[t, t+1]``
    (i.e. after any fills at step ``t``), which is exactly the quantity the price
    move ``mid[:, t+1] - mid[:, t]`` acts on — see :func:`mmlab.metrics.decompose_pnl`.
    """
    mid: np.ndarray            # (n_paths, n_steps+1) the mid-price path
    inventory: np.ndarray      # (n_paths, n_steps+1) post-fill inventory (carry)
    bid_fills: np.ndarray      # (n_paths, n_steps) 1 where the bid was hit
    ask_fills: np.ndarray      # (n_paths, n_steps) 1 where the ask was lifted
    cash: np.ndarray           # (n_paths,) terminal cash X_T (starts at 0, net of fees)
    spread_pnl: np.ndarray     # (n_paths,) sum of half-spreads captured on fills (gross)
    fee_pnl: np.ndarray        # (n_paths,) total maker fees paid (>=0; 0 at fee_per_fill=0)
    dt: float
    T: float

    @property
    def terminal_inventory(self) -> np.ndarray:
        return self.inventory[:, -1]

    @property
    def total_pnl(self) -> np.ndarray:
        """Mark-to-market at the end: cash plus inventory valued at the final mid."""
        return self.cash + self.inventory[:, -1] * self.mid[:, -1]


def run(strategy, *, S0, sigma, A, kappa, T, dt, n_paths, rng, adverse=0.0, adverse_steps=1,
        fee_per_fill=0.0):
    """Simulate ``n_paths`` market-making runs of horizon ``T`` and return a
    :class:`SimResult`.

    ``strategy(S, q, t)`` is called each step with the current mid, pre-fill
    inventory, and elapsed time (all vectors of length ``n_paths``) and returns
    the two half-spreads. ``A``/``kappa`` set the fill intensity; ``sigma`` the
    mid volatility (price per sqrt-second); ``adverse`` the post-fill drift, spread
    evenly over ``adverse_steps`` steps starting at the fill (``1`` = a single jump
    at the fill step, the original behaviour). ``fee_per_fill`` is the maker fee in
    **price units**, charged on every fill on both sides (default ``0.0`` — the fee is
    the dominant term at retail tiers but off by default so the self-test identity is
    unchanged); a *negative* value models an exchange market-maker rebate.
    """
    n = int(round(T / dt))
    K = max(int(adverse_steps), 1)
    dW = sigma * np.sqrt(dt) * rng.standard_normal((n_paths, n))   # exogenous mid increments
    U_b = rng.random((n_paths, n))                                 # fill draws, bid / ask
    U_a = rng.random((n_paths, n))

    S = np.full(n_paths, float(S0))
    q = np.zeros(n_paths)
    X = np.zeros(n_paths)
    sched = np.zeros((n_paths, K))                                # adverse drift scheduled ahead

    mid = np.empty((n_paths, n + 1))
    mid[:, 0] = S
    inv = np.empty((n_paths, n + 1))
    bid_fills = np.zeros((n_paths, n))
    ask_fills = np.zeros((n_paths, n))
    spread_pnl = np.zeros(n_paths)
    fee_pnl = np.zeros(n_paths)

    for t in range(n):
        db, da = strategy(S, q, t * dt)
        db = np.asarray(db, dtype=float)
        da = np.asarray(da, dtype=float)
        p_b = 1.0 - np.exp(-A * np.exp(-kappa * db) * dt)          # exact per-step fill prob
        p_a = 1.0 - np.exp(-A * np.exp(-kappa * da) * dt)
        fb = (U_b[:, t] < p_b).astype(float)                      # bid hit -> we BUY 1 @ S-db
        fa = (U_a[:, t] < p_a).astype(float)                      # ask lifted -> we SELL 1 @ S+da

        X += fa * (S + da) - fb * (S - db)                        # sell adds cash, buy spends it
        fills = fb + fa
        X -= fee_per_fill * fills                                 # maker fee on every fill, both sides
        fee_pnl += fee_per_fill * fills
        q += fb - fa
        spread_pnl += fb * db + fa * da                           # half-spread earned per fill (gross)
        bid_fills[:, t] = fb
        ask_fills[:, t] = fa
        inv[:, t] = q                                             # inventory carried over [t, t+1]

        if adverse:
            sched += (adverse * (fa - fb) / K)[:, None]           # sell -> rises against us; buy -> falls
        adv_now = sched[:, 0].copy()
        S = S + dW[:, t] + adv_now
        mid[:, t + 1] = S
        if K > 1:
            sched[:, :-1] = sched[:, 1:]                          # roll the schedule forward one step
        sched[:, -1] = 0.0

    inv[:, n] = q
    return SimResult(mid=mid, inventory=inv, bid_fills=bid_fills, ask_fills=ask_fills,
                     cash=X, spread_pnl=spread_pnl, fee_pnl=fee_pnl, dt=dt, T=T)


def simulate_mid(S0, sigma, T, dt, n_paths, rng):
    """Bare arithmetic Brownian mid ``dS = sigma dW`` (no drift, no fills), for
    plotting the price process and for volatility-estimator tests. Arithmetic (not
    geometric) because that is what Avellaneda-Stoikov assumes; it permits negative
    prices, which is harmless over the short horizons used here."""
    n = int(round(T / dt))
    incr = sigma * np.sqrt(dt) * rng.standard_normal((n_paths, n))
    return S0 + np.concatenate([np.zeros((n_paths, 1)), np.cumsum(incr, axis=1)], axis=1)
