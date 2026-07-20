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
# The 20 bps all-in turnover cost, itemised (P1.5) so it is a defensible stack, not a magic number:
#   FEE    — Binance spot taker fee (~10 bps; lower on VIP tiers / with BNB, so this is conservative)
#   SPREAD — half the bid/ask spread paid by a market order in a liquid major (~10 bps)
# Impact beyond top-of-book is modelled separately as a square-root cost (engine.liq_cost_frame) and
# reported alongside. FEE + SPREAD == TCOST, so the headline number is unchanged.
FEE_BPS = 0.0010
SPREAD_BPS = 0.0010
TCOST = FEE_BPS + SPREAD_BPS  # 20 bps all-in, charged on traded notional (turnover)
ANN = np.sqrt(365)           # crypto trades 365 days/yr -> daily Sharpe * sqrt(365)

# Risk-free / minimum-acceptable return (P1.6). Zero because every book here is dollar-neutral and
# self-funding — the long and short legs finance each other — so excess return ~= raw return and the
# ~4-5% USD rate over 2022-24 nets out. Sharpe/Sortino default to this; exposed so the choice is deliberate.
RF = 0.0

# ---- Evaluation windows ---------------------------------------------------
TRAIN = ("2020-01-01", "2023-12-31")   # in-sample design window
VAL = ("2024-01-01", "2026-05-31")     # out-of-sample test window
FULL = ("2020-01-01", "2026-05-31")    # whole sample (risk / regime stats only)


def seg(s, window):
    """Slice a Series/DataFrame to one of the evaluation windows: ``seg(x, VAL)``."""
    return s.loc[window[0]:window[1]]
