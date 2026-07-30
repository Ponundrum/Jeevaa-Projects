"""mm-lab — a market-making laboratory.

A minimal Poisson-fill simulator matched to the Avellaneda-Stoikov assumptions,
the two quoting strategies built on it, and a Layer-3 pipeline that measures
adverse selection from Binance trade data and feeds it back into the sim.

Trust the lab first::

    from mmlab import self_test
    self_test()
"""
from __future__ import annotations

from . import calibrate, data, markout, metrics, plotting, simulate, strategies
from .config import MONTH, RISK_HORIZON, SEED, SYMBOL, get_rng, rng
from .selftest import self_test
from .strategies import AvellanedaStoikov, Naive

__all__ = [
    "AvellanedaStoikov",
    "MONTH",
    "Naive",
    "RISK_HORIZON",
    "SEED",
    "SYMBOL",
    "calibrate",
    "data",
    "get_rng",
    "markout",
    "metrics",
    "plotting",
    "rng",
    "self_test",
    "simulate",
    "strategies",
]
