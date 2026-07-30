"""The simulator and the PnL accounting identity (checks 3, 4, 5)."""
import numpy as np

from mmlab import metrics, simulate
from mmlab.config import get_rng
from mmlab.strategies import AvellanedaStoikov, Naive, optimal_spread

SIG, A, KAP, T, DT = 0.1, 1.0, 1.5, 600.0, 1.0


def _half():
    return 0.5 * float(optimal_spread(0.1, SIG, KAP, 600.0))


def test_accounting_identity_naive():
    res = simulate.run(Naive(_half()), S0=100.0, sigma=SIG, A=A, kappa=KAP,
                       T=T, dt=DT, n_paths=200, rng=get_rng(1))
    assert metrics.accounting_residual(res) < 1e-8


def test_accounting_identity_as():
    res = simulate.run(AvellanedaStoikov(0.1, SIG, KAP, T), S0=100.0, sigma=SIG,
                       A=A, kappa=KAP, T=T, dt=DT, n_paths=200, rng=get_rng(2))
    assert metrics.accounting_residual(res) < 1e-8


def test_accounting_identity_holds_under_adverse_selection():
    # The identity is pure bookkeeping, so it must survive the Layer-3c drift too.
    res = simulate.run(Naive(_half()), S0=100.0, sigma=SIG, A=A, kappa=KAP,
                       T=T, dt=DT, n_paths=200, rng=get_rng(3), adverse=0.05)
    assert metrics.accounting_residual(res) < 1e-8


def test_no_fills_when_A_zero():
    res = simulate.run(Naive(_half()), S0=100.0, sigma=SIG, A=0.0, kappa=KAP,
                       T=T, dt=DT, n_paths=50, rng=get_rng(4))
    assert res.bid_fills.sum() == 0 and res.ask_fills.sum() == 0
    assert np.all(res.terminal_inventory == 0)
    assert np.allclose(res.total_pnl, 0.0)


def test_fill_rate_matches_intensity():
    # Small dt so measured rate -> lambda = A*exp(-kappa*delta).
    delta, T5, dt5, n5 = 1.0, 100.0, 0.02, 2000
    res = simulate.run(Naive(delta), S0=100.0, sigma=0.0, A=A, kappa=KAP,
                       T=T5, dt=dt5, n_paths=n5, rng=get_rng(5))
    measured = res.bid_fills.sum() / (n5 * T5)
    expected = A * np.exp(-KAP * delta)
    assert abs(measured - expected) / expected < 0.03


def test_adverse_steps_default_equals_single_jump():
    # adverse_steps=1 (default) must reproduce the single-jump-at-fill behaviour exactly.
    a = simulate.run(Naive(_half()), S0=100.0, sigma=SIG, A=A, kappa=KAP, T=T, dt=DT,
                     n_paths=200, rng=get_rng(9), adverse=0.05)
    b = simulate.run(Naive(_half()), S0=100.0, sigma=SIG, A=A, kappa=KAP, T=T, dt=DT,
                     n_paths=200, rng=get_rng(9), adverse=0.05, adverse_steps=1)
    assert np.allclose(a.mid, b.mid) and np.allclose(a.total_pnl, b.total_pnl)


def test_accounting_identity_holds_for_spread_adverse():
    # Spreading the adverse move over many steps must not break the PnL identity.
    for k in (5, 30, 60):
        res = simulate.run(Naive(_half()), S0=100.0, sigma=SIG, A=A, kappa=KAP, T=T, dt=DT,
                           n_paths=200, rng=get_rng(10), adverse=0.05, adverse_steps=k)
        assert metrics.accounting_residual(res) < 1e-8
