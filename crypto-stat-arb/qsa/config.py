"""Project-wide configuration: paths, cost/annualisation constants, and the
train / validation / full evaluation windows.

Every number in the project is anchored to these three windows:

- ``TRAIN``  2020-2023  in-sample design window (all model/parameter choices)
- ``VAL``    2024-2026  out-of-sample test window (never used for selection)
- ``FULL``   2020-2026  whole sample (drawdown, beta and regime statistics only)

Keeping them in one place means a notebook can never quietly disagree with the
engine about what "out-of-sample" means.
"""
from pathlib import Path
import numpy as np

# ---- Paths ----------------------------------------------------------------
DATA = Path("crypto_data/processed")          # cached parquet lives here
ARCHIVE = Path("crypto_data/_archive")        # raw Binance zips cached here

# ---- Cost / annualisation -------------------------------------------------
TCOST = 0.0020                # 20 bps all-in, charged on traded notional (turnover)
ANN = np.sqrt(365)           # crypto trades 365 days/yr -> daily Sharpe * sqrt(365)

# ---- Evaluation windows ---------------------------------------------------
TRAIN = ("2020-01-01", "2023-12-31")   # in-sample design window
VAL = ("2024-01-01", "2026-05-31")     # out-of-sample test window
FULL = ("2020-01-01", "2026-05-31")    # whole sample (risk / regime stats only)


def seg(s, window):
    """Slice a Series/DataFrame to one of the evaluation windows: ``seg(x, VAL)``."""
    return s.loc[window[0]:window[1]]
