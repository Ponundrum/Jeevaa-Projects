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


def test_fee_three_term_identity_and_default_unchanged():
    # With fees on, total == spread - fee + inventory (to float); the two-term identity
    # (check 3) still holds exactly at the default zero fee.
    base = simulate.run(Naive(_half()), S0=100.0, sigma=SIG, A=A, kappa=KAP, T=T, dt=DT,
                        n_paths=200, rng=get_rng(11))
    assert base.fee_pnl.sum() == 0.0
    assert metrics.accounting_residual(base) < 1e-8              # two-term identity unchanged
    fee = 0.02                                                  # price units per fill
    withfee = simulate.run(Naive(_half()), S0=100.0, sigma=SIG, A=A, kappa=KAP, T=T, dt=DT,
                           n_paths=200, rng=get_rng(11), fee_per_fill=fee)
    assert metrics.fee_adjusted_residual(withfee) < 1e-8        # three-term identity holds
    # fee_pnl equals fee x total fills, and it lowers PnL by exactly that.
    fills = withfee.bid_fills.sum(axis=1) + withfee.ask_fills.sum(axis=1)
    assert np.allclose(withfee.fee_pnl, fee * fills)
    assert np.allclose(withfee.total_pnl, base.total_pnl - withfee.fee_pnl)


def test_negative_fee_is_a_rebate():
    # A negative fee (market-maker rebate) increases PnL relative to zero fee.
    base = simulate.run(Naive(_half()), S0=100.0, sigma=SIG, A=A, kappa=KAP, T=T, dt=DT,
                        n_paths=200, rng=get_rng(12))
    rebate = simulate.run(Naive(_half()), S0=100.0, sigma=SIG, A=A, kappa=KAP, T=T, dt=DT,
                          n_paths=200, rng=get_rng(12), fee_per_fill=-0.01)
    assert rebate.total_pnl.mean() > base.total_pnl.mean()
