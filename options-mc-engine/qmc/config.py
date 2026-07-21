"""Project-wide conventions: the seeded RNG, rates, and cache paths.

Every source of randomness in the project flows through :func:`get_rng` (or the
module-level :data:`rng`), so every notebook and test is deterministic given the
seed. Keeping the conventions here means no module can silently disagree about
the risk-free rate, the day count, or where the data cache lives — the same
single-source-of-truth principle as the sibling project's ``qsa/config.py``.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# ---- Randomness -----------------------------------------------------------
SEED = 20240101                      # stated in the README; every result reproduces from it


def get_rng(seed: int | None = None) -> np.random.Generator:
    """Return a fresh, deterministic NumPy generator. Pass an explicit ``seed``
    for an independent stream (tests use this); omit it for the project seed."""
    return np.random.default_rng(SEED if seed is None else seed)


rng = get_rng()                      # the default project stream

# ---- Market conventions ---------------------------------------------------
DEFAULT_R = 0.05                     # risk-free rate (annualised, continuous comp.)
DEFAULT_Q = 0.0                      # dividend yield (annualised, continuous comp.)
TRADING_DAYS = 252                   # day-count for annualising realised quantities

# ---- Paths ----------------------------------------------------------------
# Anchor the cache to the project root (dir above the qmc package), not the CWD,
# so notebooks and scripts share one cache regardless of where they are launched.
_ROOT = Path(os.environ.get("QMC_DATA_DIR", Path(__file__).resolve().parent.parent))
CACHE = _ROOT / "_cache"
