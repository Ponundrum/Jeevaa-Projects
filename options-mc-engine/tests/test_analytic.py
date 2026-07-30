"""Closed-form Black-Scholes ground truth — the anchor everything else is checked
against, so it is pinned to textbook values."""
import numpy as np
import pytest

from qmc.analytic import (bs_price, bs_greeks, bs_implied_vol, put_call_parity_gap,
                          geometric_asian_price, barrier_price, lookback_floating_price,
                          digital_price, digital_delta)

S, K, T, r, SIG, Q = 100.0, 100.0, 1.0, 0.05, 0.2, 0.0


def test_bs_textbook_values():
    assert bs_price(S, K, T, r, SIG, Q, "call") == pytest.approx(10.4506, abs=1e-3)
    assert bs_price(S, K, T, r, SIG, Q, "put") == pytest.approx(5.5735, abs=1e-3)


def test_bs_greeks_textbook():
    g = bs_greeks(S, K, T, r, SIG, Q, "call")
    assert g["delta"] == pytest.approx(0.6368, abs=1e-3)
    assert g["gamma"] == pytest.approx(0.01876, abs=1e-4)
    assert g["vega"] == pytest.approx(37.524, abs=1e-2)


def test_put_call_parity_closed_form():
    c = bs_price(S, K, T, r, SIG, Q, "call")
    p = bs_price(S, K, T, r, SIG, Q, "put")
    assert put_call_parity_gap(c, p, S, K, T, r, Q) == pytest.approx(0.0, abs=1e-10)


def test_vega_matches_finite_difference():
    g = bs_greeks(S, K, T, r, SIG, Q, "call")
    fd = (bs_price(S, K, T, r, SIG + 1e-5, Q) - bs_price(S, K, T, r, SIG - 1e-5, Q)) / 2e-5
    assert g["vega"] == pytest.approx(fd, rel=1e-4)


def test_implied_vol_roundtrips():
    for kind in ("call", "put"):
        price = bs_price(S, K, T, r, SIG, Q, kind)
        assert bs_implied_vol(price, S, K, T, r, Q, kind) == pytest.approx(SIG, abs=1e-6)


def test_implied_vol_out_of_bounds_is_nan():
    assert np.isnan(bs_implied_vol(1e-9, S, K, T, r, Q, "call"))       # below intrinsic-ish
    assert np.isnan(bs_implied_vol(200.0, S, K, T, r, Q, "call"))      # above the upper bound


def test_geometric_asian_below_european():
    ga = geometric_asian_price(S, K, T, r, SIG, Q, 50, "call")
    eu = bs_price(S, K, T, r, SIG, Q, "call")
    assert 0 < ga < eu                                                 # averaging lowers vol -> cheaper


def test_barrier_in_out_parity():
    # knock-in + knock-out = vanilla (no rebate), exactly
    for style_in, style_out, kind, B in [("down-and-in", "down-and-out", "call", 90),
                                         ("up-and-in", "up-and-out", "call", 120),
                                         ("down-and-in", "down-and-out", "put", 90)]:
        ki = barrier_price(S, K, B, T, r, SIG, Q, kind, style_in)
        ko = barrier_price(S, K, B, T, r, SIG, Q, kind, style_out)
        assert ki + ko == pytest.approx(bs_price(S, K, T, r, SIG, Q, kind), abs=1e-10)


def test_barrier_knockout_below_vanilla():
    ko = barrier_price(S, K, 90, T, r, SIG, Q, "call", "down-and-out")
    assert 0 < ko < bs_price(S, K, T, r, SIG, Q, "call")               # knocking out can only lower value


def test_lookback_call_exceeds_vanilla():
    # a floating-strike lookback call (S_T - min) dominates the European call
    lb = lookback_floating_price(S, T, r, SIG, Q, "call")
    assert lb > bs_price(S, K, T, r, SIG, Q, "call") > 0


def test_digital_delta_matches_finite_difference():
    fd = (digital_price(S + 1e-3, K, T, r, SIG, Q, "call") - digital_price(S - 1e-3, K, T, r, SIG, Q, "call")) / 2e-3
    assert digital_delta(S, K, T, r, SIG, Q, "call") == pytest.approx(fd, rel=1e-4)


def test_lookback_is_continuous_at_zero_carry():
    # r == q used to raise ZeroDivisionError; the b -> 0 limit is finite and the
    # function must be continuous through it.
    for kind in ("call", "put"):
        at_zero = lookback_floating_price(S, T, 0.05, SIG, 0.05, kind)
        near_zero = lookback_floating_price(S, T, 0.05, SIG, 0.05 - 1e-8, kind)
        assert at_zero == pytest.approx(near_zero, abs=1e-6)
        assert at_zero > 0
