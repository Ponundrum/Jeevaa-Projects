"""Signal library.

A *signal* is a cross-sectional DataFrame (dates x coins) of conviction scores,
aligned to ``ds.close_all``. Higher = more attractive to be long. Turning a
signal into a book is the job of the construction rules in :mod:`qsa.engine`
(``eq_weights`` for directional, ``dn_weights`` for market-neutral).

The two survivors (``near_high``, ``idio_vol``) and the rejected candidates all
live here so the combined-book notebook and the signal-research notebook read
from exactly the same definitions — no divergence between "the version I traded"
and "the version I tested".
"""
import numpy as np


# ---- Survivors ------------------------------------------------------------
def near_high(ds, L=90):
    """Proximity-to-trailing-high momentum: ``close / rolling_max(close, L)``.
    1.0 = at its high, lower = further below. Long the coins nearest their high."""
    return ds.close_all / ds.close_all.rolling(L).max()


def idio_vol(ds, L=60):
    """Idiosyncratic low-volatility / betting-against-beta: negative trailing
    std of the BTC-residual return, so LOW idiosyncratic vol ranks highest."""
    return -ds.resid_all.rolling(L).std()


# ---- Rejected candidates (studied in 02_signal_research) ------------------
def ts_trend(ds, horizons=(30, 60, 90)):
    """Directional time-series trend conviction: fraction of look-back horizons
    over which price is up. For ``eq_weights`` (long-only) -> ends up ~market beta."""
    px = ds.close_all
    return sum((px.pct_change(h, fill_method=None) > 0).astype(float) for h in horizons) / len(horizons)


def cs_momentum(ds, L=30):
    """Raw cross-sectional momentum: trailing L-day return. Long winners / short
    losers. Contaminated by short-term reversal in the most recent winners."""
    return ds.close_all.pct_change(L, fill_method=None)


def reversal(ds, L=3):
    """Short-horizon price reversal: negative trailing L-day return (long recent
    losers). Real gross signal, but ~daily turnover buries it after cost."""
    return -ds.close_all.pct_change(L, fill_method=None)


def idio_reversal(ds, L=3):
    """Beta-neutral (idiosyncratic) reversal: negative trailing L-day sum of the
    BTC-residual return."""
    return -ds.resid_all.rolling(L).sum()


def max_return_lottery(ds, L=30):
    """Lottery / MAX effect: short coins with the largest single-day return over
    the window (negative of the rolling max daily return)."""
    return -ds.ret_all.rolling(L).max()


def return_skew(ds, L=30):
    """Short high positive-skew ('lottery-like') coins: negative rolling skew."""
    return -ds.ret_all.rolling(L).skew()


def long_horizon_momentum(ds, L=270):
    """Classic 9-12 month formation momentum (hold monthly via the backtest's
    ``rebal``)."""
    return ds.close_all.pct_change(L, fill_method=None)


def amihud_illiquidity(ds, L=30):
    """Amihud illiquidity: rolling mean of |return| / dollar-volume. Higher =
    more illiquid. Long illiquid / short liquid."""
    illiq = ds.ret_all.abs() / ds.qvol_all.replace(0, np.nan)
    return illiq.rolling(L).mean()


def orderflow_raw(ds, L=5):
    """'Order flow' the WRONG way: rolling taker-buy BASE volume (token units).
    A scale artifact — token counts aren't comparable across coins."""
    return ds.tbb_all.rolling(L).sum()


def orderflow_scalefree(ds, L=5):
    """Scale-free order flow: taker-buy DOLLAR fraction of total dollar volume,
    averaged over L days. The honest version of the signal above."""
    buy_dollars = ds.tbb_all * ds.close_all
    frac = buy_dollars / ds.qvol_all.replace(0, np.nan)
    return frac.rolling(L).mean()
