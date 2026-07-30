"""Project-wide conventions: the seeded RNG, the one symbol/month we study, the
risk horizon, and the cache path. One source of truth so no notebook or test can
quietly disagree — the same principle as the sibling projects' ``qsa/config.py``
and ``qmc/config.py``.

Units convention (stated once, obeyed everywhere): **time is in seconds** and
**prices are in quote-currency units (USDT)**. So ``sigma`` is price per
``sqrt(second)``, a fill intensity ``lambda`` is per second, and ``kappa`` is per
price unit. Unit slips between these are the single most likely bug in the whole
project, which is why they live here in one place.
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

# ---- What we study --------------------------------------------------------
# One liquid symbol, one month. Deliberately narrow: nobody asks why you only did
# one symbol, but plenty of people ask what kappa means (see the README).
SYMBOL = "BTCUSDT"
MONTH = "2024-01"                    # YYYY-MM; the aggTrades archive month pulled in notebook 02

# ---- Risk horizon ---------------------------------------------------------
# A 24/7 crypto market has no session close, so the Avellaneda-Stoikov (T - t) term
# has no natural end. RISK_HORIZON is the pragmatic answer: "how long until I expect
# to be able to flatten", held constant, in seconds. See strategies.avellaneda_stoikov.
RISK_HORIZON = 600.0                 # 10 minutes, in seconds

# ---- Paths ----------------------------------------------------------------
# Anchor the cache to the project root (dir above the mmlab package), not the CWD,
# so notebooks and scripts share one cache regardless of where they are launched.
_ROOT = Path(os.environ.get("MMLAB_DATA_DIR", Path(__file__).resolve().parent.parent))
CACHE = _ROOT / "_cache"             # raw Binance zips + parsed parquet cached here
