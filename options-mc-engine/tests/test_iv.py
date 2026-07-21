"""Implied-vol surface tools: inversion round-trip, SVI fit, and the static-
arbitrage checks. No network."""
import numpy as np
import pandas as pd

from qmc.analytic import bs_price
from qmc import iv

S, r, q = 100.0, 0.05, 0.0


def test_iv_inversion_roundtrips_to_1e6():
    chain = pd.DataFrame([{"T": T, "K": K, "kind": "call" if K >= 100 else "put",
                           "price": bs_price(S, K, T, r, 0.2, q, "call" if K >= 100 else "put")}
                          for T in (0.25, 1.0) for K in (80, 90, 100, 110, 120)])
    surf = iv.implied_vol_surface(chain, S, r, q)
    assert np.allclose(surf["iv"], 0.20, atol=1e-6)


def test_svi_recovers_synthetic_slice():
    k = np.linspace(-0.5, 0.5, 21)
    true = (0.04, 0.1, -0.4, 0.0, 0.1)
    w = iv.svi_total_variance(k, true)
    fit = iv.fit_svi(k, w)
    assert np.sqrt(np.mean((iv.svi_total_variance(k, fit) - w) ** 2)) < 1e-6


def test_calm_svi_slice_is_butterfly_free():
    # a mild, symmetric smile has a non-negative density everywhere
    params = (0.04, 0.1, -0.3, 0.0, 0.2)
    g = iv.svi_butterfly_g(np.linspace(-1, 1, 201), params)
    assert g.min() >= 0


def test_calendar_check_flags_decreasing_total_variance():
    # two slices where the longer maturity has LOWER total variance -> calendar arbitrage
    near = (0.06, 0.1, -0.3, 0.0, 0.2)      # higher level
    far = (0.03, 0.1, -0.3, 0.0, 0.2)       # lower level at longer T -> violation
    res = iv.check_no_arbitrage({0.25: near, 1.0: far})
    assert not res["calendar_ok"] and res["calendar_violations"] > 0


def test_no_arbitrage_passes_on_consistent_slices():
    near = (0.03, 0.1, -0.3, 0.0, 0.2)
    far = (0.06, 0.12, -0.3, 0.0, 0.2)      # higher total variance at longer T
    res = iv.check_no_arbitrage({0.25: near, 1.0: far})
    assert res["calendar_ok"] and res["butterfly_ok"]
