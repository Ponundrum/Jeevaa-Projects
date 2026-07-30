"""PnL decomposition and inventory metrics (check 7)."""
import numpy as np

from mmlab import metrics, simulate
from mmlab.config import get_rng
from mmlab.strategies import AvellanedaStoikov, Naive, optimal_spread

SIG, A, KAP, T, DT = 0.1, 1.0, 1.5, 600.0, 1.0
HALF = 0.5 * float(optimal_spread(0.1, SIG, KAP, 600.0))


def _run(strat, seed):
    return simulate.run(strat, S0=100.0, sigma=SIG, A=A, kappa=KAP, T=T, dt=DT,
                        n_paths=400, rng=get_rng(seed))


def test_decomposition_sums_to_total():
    res = _run(Naive(HALF), 10)
    total, spread, inv = metrics.decompose_pnl(res)
    assert np.allclose(total, spread + inv, atol=1e-8)


def test_naive_captures_more_gross_spread():
    # Naive quotes tight and constantly, so its gross spread PnL exceeds AS's.
    n = metrics.summary("Naive", _run(Naive(HALF), 11))
    a = metrics.summary("AS", _run(AvellanedaStoikov(0.1, SIG, KAP, T), 11))
    assert n["spread_pnl"] > a["spread_pnl"]


def test_as_has_lower_terminal_inventory_std():
    # The one behavioural claim worth pinning (check 7) — inventory, NOT PnL.
    n = metrics.summary("Naive", _run(Naive(HALF), 12))
    a = metrics.summary("AS", _run(AvellanedaStoikov(0.1, SIG, KAP, T), 12))
    assert a["term_inv_std"] < n["term_inv_std"]


def test_inventory_std_curve_grows_for_naive():
    res = _run(Naive(HALF), 13)
    curve = metrics.inventory_std_over_time(res)
    assert curve[-1] > curve[len(curve) // 4]      # random walk keeps widening


def test_sharpe_of_constant_is_nan():
    assert np.isnan(metrics.sharpe(np.ones(10)))
