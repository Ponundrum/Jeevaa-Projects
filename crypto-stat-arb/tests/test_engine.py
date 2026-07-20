"""Engine correctness — the properties every downstream number depends on."""
import numpy as np
import pandas as pd

from qsa.engine import (backtest, backtest_carry, dn_weights, maintenance_turnover,
                        sharpe, sortino, maxdd, self_test)


def test_engine_self_test():
    # No look-ahead, turnover charged, buy-hold identity — the built-in asserts.
    assert "passed" in self_test().lower()


def test_no_lookahead_and_cost():
    r = pd.DataFrame({"A": [0.0, 0.10, -0.05, 0.02]})
    # constant fully-invested weight reproduces buy-and-hold from day 1 (weights lag one bar)
    net, _ = backtest(pd.DataFrame({"A": [1., 1, 1, 1]}), r, tcost=0.0)
    assert abs(net.iloc[1:].sum() - r["A"].iloc[1:].sum()) < 1e-12
    # a higher cost strictly lowers PnL on a round trip
    n0, _ = backtest(pd.DataFrame({"A": [0, 1, 1, 0]}), r, tcost=0.0)
    n1, _ = backtest(pd.DataFrame({"A": [0, 1, 1, 0]}), r, tcost=0.01)
    assert n0.sum() - n1.sum() > 0


def test_dn_weights_dollar_neutral(ds):
    w = dn_weights(ds.close_all / ds.close_all.rolling(20).max(), ds)
    rows = w.abs().sum(1) > 0
    # each active row is dollar-neutral (long = short) and gross-1
    assert (w[rows].sum(1).abs() < 1e-9).all()
    assert (np.abs(w[rows].abs().sum(1) - 1.0) < 1e-9).all()


def test_dn_weights_no_infeasible_short(ds):
    w = dn_weights(-ds.resid_all.rolling(15).std(), ds)
    # the non-shortable coin must never carry a negative (short) weight
    assert w[ds.NONSHORT].min() >= -1e-12


def test_metrics_known_inputs():
    idx = pd.date_range("2021-01-01", periods=100, freq="D")
    up = pd.Series(np.linspace(0.001, 0.002, 100), index=idx)   # strictly positive returns
    assert maxdd(up) == 0.0                                     # monotone up -> no drawdown
    assert sharpe(up) > 0
    # Sortino >= Sharpe when there is no (or little) downside relative to total vol
    mixed = pd.Series(np.r_[np.full(90, 0.01), np.full(10, -0.005)], index=idx)
    assert sortino(mixed) >= sharpe(mixed) - 1e-9


def test_sharpe_ddof_consistency():
    # sharpe() (pandas ddof=1) and a manual ddof=1 Sharpe agree exactly
    x = pd.Series(np.random.default_rng(1).normal(0.001, 0.02, 300))
    manual = x.mean() / x.std(ddof=1) * np.sqrt(365)
    assert abs(sharpe(x) - manual) < 1e-12


def test_maintenance_turnover_zero_on_rebalance_days(ds):
    w = dn_weights(ds.close_all / ds.close_all.rolling(20).max(), ds)
    mt = maintenance_turnover(w, ds.ret_all, rebal=7)
    assert (mt.iloc[::7] == 0).all()                           # zero on rebalance days
    assert mt.min() >= 0


def test_backtest_carry_matches_manual(ds):
    # backtest_carry reproduces the explicit long-spot/short-perp formula
    spot = ds.close_all
    perp = ds.close_all * (1 + 0.001)                          # perp tracks spot with a tiny basis
    funding = pd.DataFrame(0.0001, index=spot.index, columns=spot.columns)
    syms = list(spot.columns[:4])
    net, _ = backtest_carry(spot, perp, funding, syms, capture=0.85, drag=0.02, rebal=7)
    sr, pr = spot[syms].pct_change(fill_method=None), perp[syms].pct_change(fill_method=None)
    av = (spot[syms].notna() & perp[syms].notna()).astype(float)
    w = av.div(av.sum(1).replace(0, np.nan), axis=0)
    w = w.where(pd.Series(np.arange(len(w)) % 7 == 0, index=w.index), np.nan).ffill()
    turn = (w.shift(1) - w.shift(2)).abs().sum(1).fillna(0.0)
    manual = (w.shift(1) * (sr - pr + 0.85 * funding[syms])).sum(1) - 2 * 0.0020 * turn - 0.02 / 365
    assert np.allclose(net.fillna(0), manual.fillna(0), atol=1e-15)
