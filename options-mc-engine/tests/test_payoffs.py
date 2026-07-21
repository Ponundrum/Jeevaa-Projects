"""Payoff functions on hand-built paths — deterministic, no randomness."""
import numpy as np
import pytest

from qmc import payoffs

# two paths: one rises to 120, one dips to 80 then ends at 100
PATHS = np.array([[100., 110., 120.],
                  [100., 80., 100.]])


def test_european():
    assert list(payoffs.european("call", 100)(PATHS)) == [20.0, 0.0]
    assert list(payoffs.european("put", 100)(PATHS)) == [0.0, 0.0]


def test_asian_arithmetic_and_geometric():
    # average over columns 1: -> [(110+120)/2, (80+100)/2] = [115, 90]
    assert payoffs.asian_arithmetic("call", 100)(PATHS)[0] == 15.0
    g = payoffs.asian_geometric("call", 100)(PATHS)
    assert g[0] == pytest.approx(np.sqrt(110 * 120) - 100)             # geometric < arithmetic
    assert g[0] < 15.0


def test_barrier_knockout_and_knockin():
    # up-and-out at 115: path 0 breaches (max 120) -> dead; path 1 survives
    assert list(payoffs.barrier("call", 100, 115, "up-and-out")(PATHS)) == [0.0, 0.0]
    # down-and-in at 85: path 1 touches 80 -> knocks in (payoff 0 at K=100 anyway)
    assert list(payoffs.barrier("call", 90, 85, "down-and-in")(PATHS)) == [0.0, 10.0]


def test_lookback_and_digital():
    assert list(payoffs.lookback("call")(PATHS)) == [20.0, 20.0]       # S_T - min
    assert list(payoffs.digital("call", 100)(PATHS)) == [1.0, 0.0]     # ST>100 only path 0
