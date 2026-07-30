"""The Avellaneda-Stoikov formulas, checked analytically (no simulation)."""
import numpy as np
import pytest

from mmlab.strategies import (AvellanedaStoikov, Naive, edge_term,
                              inventory_risk_term, optimal_spread, reservation_price)

SIG, KAP, TAU = 0.2, 1.5, 600.0


def test_edge_term_gamma_to_zero_limit():
    # (2/gamma)*ln(1+gamma/kappa) -> 2/kappa as gamma -> 0 (check 1).
    for g in (1e-2, 1e-4, 1e-6):
        assert float(edge_term(g, KAP)) == pytest.approx(2.0 / KAP, rel=g)
    assert float(edge_term(0.0, KAP)) == pytest.approx(2.0 / KAP)   # handled as the limit


def test_zero_inventory_reservation_is_mid():
    # q = 0 => reservation price == mid, exactly (check 2).
    assert reservation_price(100.0, 0.0, 0.1, SIG, TAU) == 100.0


def test_reservation_leans_against_inventory():
    # long -> reservation below mid (quote to sell); short -> above.
    assert reservation_price(100.0, 5.0, 0.1, SIG, TAU) < 100.0
    assert reservation_price(100.0, -5.0, 0.1, SIG, TAU) > 100.0


def test_spread_is_sum_of_two_terms():
    g = 0.1
    assert optimal_spread(g, SIG, KAP, TAU) == pytest.approx(
        inventory_risk_term(g, SIG, TAU) + float(edge_term(g, KAP)))


def test_inventory_term_vanishes_at_expiry():
    assert inventory_risk_term(0.1, SIG, 0.0) == 0.0


def test_as_skews_quotes_with_inventory():
    # When long, AS should quote a tighter ask (sell) and wider bid (buy) than flat.
    strat = AvellanedaStoikov(0.1, SIG, KAP, T=600.0, frozen_horizon=600.0)
    db, da = strat(np.array([100.0]), np.array([5.0]), 0.0)
    assert da < db          # cheaper to lift our ask -> we shed the long


def test_naive_is_symmetric_and_flat():
    strat = Naive(0.5)
    db, da = strat(np.array([100.0, 100.0]), np.array([3.0, -3.0]), 0.0)
    assert np.all(db == 0.5) and np.all(da == 0.5)


def test_min_half_spread_floors_quotes():
    # At high inventory the AS ask can go negative (quote through the mid); the floor clamps it.
    raw = AvellanedaStoikov(0.01, SIG, KAP, T=600.0, frozen_horizon=600.0)
    db, da = raw(np.array([100.0]), np.array([50.0]), 0.0)
    assert da < 0                                      # unclamped crosses the mid
    floored = AvellanedaStoikov(0.01, SIG, KAP, T=600.0, frozen_horizon=600.0, min_half_spread=0.0)
    db2, da2 = floored(np.array([100.0]), np.array([50.0]), 0.0)
    assert da2 >= 0.0 and db2 >= 0.0                   # floor holds both sides at >= 0
