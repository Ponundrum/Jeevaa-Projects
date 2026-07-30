"""``self_test()`` — the seven correctness checks the whole project rests on,
run first in every notebook (the same "trust it before you use it" discipline as
the sibling projects). Two are pure analytic identities on the strategy formulas;
the rest validate the simulator and the calibrator by Monte Carlo or against known
ground truth. Nothing here touches the network.

The checks, in order (matching the build spec):

1. ``gamma -> 0`` limit: the edge term ``-> 2/kappa`` and the reservation price
   ``-> mid``.
2. Zero inventory: ``q = 0`` implies reservation price ``== mid`` exactly.
3. Accounting identity: ``total PnL == spread PnL + inventory PnL`` per path.
4. No fills: ``A = 0`` gives zero fills, inventory and PnL.
5. Simulator fill rate: measured fill rate matches ``A e^{-kappa delta}``.
6. Estimator recovery: the Layer-3a fit recovers known ``A``/``kappa`` from a
   synthetic tape (the important one — it validates calibration before real data).
7. AS beats naive on **inventory** (not PnL): matched-parameter terminal inventory
   std is strictly lower for Avellaneda-Stoikov.
"""
from __future__ import annotations

import numpy as np

from . import calibrate, metrics, simulate
from .config import get_rng
from .strategies import (AvellanedaStoikov, Naive, edge_term, optimal_spread,
                         reservation_price)


def _check(name, ok, detail=""):
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {name}{'  — ' + detail if detail else ''}")
    if not ok:
        raise AssertionError(f"self-test failed: {name} ({detail})")


def self_test(verbose=True):
    """Run checks 1-7. Raises ``AssertionError`` on the first failure; returns
    ``True`` if all pass."""
    if verbose:
        print("mm-lab self-test")

    # 1. gamma -> 0 limit --------------------------------------------------
    kappa = 1.5
    g_small = 1e-8
    edge_lim = edge_term(g_small, kappa)
    res_lim = reservation_price(100.0, 5.0, g_small, 0.2, 600.0)
    _check("1. gamma->0: edge term -> 2/kappa and reservation -> mid",
           abs(float(edge_lim) - 2.0 / kappa) < 1e-4 and abs(res_lim - 100.0) < 1e-3,
           f"edge={float(edge_lim):.5f} vs 2/kappa={2/kappa:.5f}")

    # 2. zero inventory ----------------------------------------------------
    _check("2. q=0 => reservation price == mid",
           reservation_price(100.0, 0.0, 0.1, 0.2, 600.0) == 100.0)

    # 3. accounting identity ----------------------------------------------
    rng = get_rng(1)
    sigma, A, kap, T, dt = 0.1, 1.0, 1.5, 600.0, 1.0
    tau = 600.0
    half = 0.5 * float(optimal_spread(0.1, sigma, kap, tau))
    res_as = simulate.run(AvellanedaStoikov(0.1, sigma, kap, T), S0=100.0, sigma=sigma,
                          A=A, kappa=kap, T=T, dt=dt, n_paths=300, rng=rng)
    resid = metrics.accounting_residual(res_as)
    _check("3. total PnL == spread + inventory PnL", resid < 1e-8, f"max residual={resid:.2e}")

    # 4. no fills ----------------------------------------------------------
    rng = get_rng(2)
    res0 = simulate.run(Naive(half), S0=100.0, sigma=sigma, A=0.0, kappa=kap,
                        T=T, dt=dt, n_paths=50, rng=rng)
    _check("4. A=0 => zero fills, inventory and PnL",
           res0.bid_fills.sum() == 0 and res0.ask_fills.sum() == 0
           and np.all(res0.terminal_inventory == 0) and np.allclose(res0.total_pnl, 0.0))

    # 5. simulator fill rate ----------------------------------------------
    # The engine uses the exact per-step prob 1-exp(-lambda*dt), so the measured
    # fill RATE -> lambda only as dt -> 0; a small dt makes the two agree.
    rng = get_rng(3)
    delta = 1.0
    T5, dt5, n5 = 100.0, 0.02, 2000
    rate = simulate.run(Naive(delta), S0=100.0, sigma=0.0, A=A, kappa=kap,
                        T=T5, dt=dt5, n_paths=n5, rng=rng)
    measured = rate.bid_fills.sum() / (n5 * T5)
    expected = A * np.exp(-kap * delta)
    _check("5. measured fill rate == A*exp(-kappa*delta)",
           abs(measured - expected) / expected < 0.03,
           f"measured={measured:.4f} vs expected={expected:.4f}")

    # 6. estimator recovery -----------------------------------------------
    rng = get_rng(4)
    A_true, k_true = 2.0, 1.5
    depths, dur = calibrate.synthetic_depth_tape(A_true, k_true, duration=8000.0, rng=rng)
    grid = calibrate.default_delta_grid(depths)
    A_hat, k_hat, _, _ = calibrate.fit_intensity(depths, dur, grid)
    _check("6. calibration recovers known A and kappa",
           abs(A_hat - A_true) / A_true < 0.10 and abs(k_hat - k_true) / k_true < 0.10,
           f"A={A_hat:.3f} (true {A_true}), kappa={k_hat:.3f} (true {k_true})")

    # 7. AS beats naive on inventory --------------------------------------
    rng = get_rng(5)
    common = dict(S0=100.0, sigma=sigma, A=A, kappa=kap, T=T, dt=dt, n_paths=400)
    r_naive = simulate.run(Naive(half), rng=get_rng(50), **common)
    r_as = simulate.run(AvellanedaStoikov(0.1, sigma, kap, T), rng=get_rng(50), **common)
    inv_naive = r_naive.terminal_inventory.std(ddof=1)
    inv_as = r_as.terminal_inventory.std(ddof=1)
    _check("7. AS terminal-inventory std < naive",
           inv_as < inv_naive,
           f"AS std={inv_as:.2f} < naive std={inv_naive:.2f}")

    if verbose:
        print("all checks passed.")
    return True
