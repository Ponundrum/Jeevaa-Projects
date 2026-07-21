"""Crank-Nicolson PDE pricer: a second numerical method that agrees with
Black-Scholes and prices American early exercise correctly."""
import pytest

from qmc.analytic import bs_price
from qmc.pde import crank_nicolson

S, K, T, r, SIG, Q = 100.0, 100.0, 1.0, 0.05, 0.2, 0.0


def test_european_pde_matches_black_scholes():
    for kind in ("call", "put"):
        pde = crank_nicolson(S, K, T, r, SIG, Q, kind, american=False, n_space=300, n_time=300)
        assert pde == pytest.approx(bs_price(S, K, T, r, SIG, Q, kind), abs=5e-3)


def test_american_call_no_dividend_equals_european():
    # a non-dividend American call is never exercised early -> equals the European
    ac = crank_nicolson(S, K, T, r, SIG, 0.0, "call", american=True, n_space=200, n_time=200)
    assert ac == pytest.approx(bs_price(S, K, T, r, SIG, 0.0, "call"), abs=1e-2)


def test_american_put_has_positive_early_exercise_premium():
    ap = crank_nicolson(S, K, T, r, SIG, Q, "put", american=True, n_space=200, n_time=200)
    ep = bs_price(S, K, T, r, SIG, Q, "put")
    assert ap > ep + 0.1                                              # meaningful early-exercise premium
    assert ap >= K - S - 1e-6                                         # at least intrinsic
