"""Markouts — the empirical heart of the project (Layer 3b).

For every trade on the tape, signed from the **passive** maker's point of view,
we measure how the mid moves afterwards::

    markout(h) = (mid_{t+h} - fill_price) * (+1 if the maker bought, -1 if sold)

Averaged across trades at several horizons ``h`` and reported in basis points, this
is the adverse selection facing a resting quote. The *measured* curve on BTCUSDT is
slightly positive at 1s (the causal mid proxy lags by ~1s, so the maker still looks
up by roughly the spread it just earned), then turns firmly negative by 5s and
**plateaus near -0.65 bps out to 300s** — adverse selection is essentially fully
realized within a few seconds and does not keep accumulating. (Going in, the
textbook expectation was a curve that *worsens* monotonically with horizon; the flat
plateau is what the data actually show, and it usefully bounds how long a maker has
to react.)

``is_buyer_maker`` (Binance's flag) identifies the passive side: ``True`` means the
maker was the buyer (a resting bid got hit), so the maker **bought** -> sign ``+1``.
"""
from __future__ import annotations

import numpy as np


def _mid_at(t_mid, p_mid, query):
    """Step-interpolate the mid: value at the most recent mid observation at or
    before each ``query`` time. ``t_mid`` must be sorted ascending. Queries beyond
    the last observation return NaN (dropped by the caller)."""
    idx = np.searchsorted(t_mid, query, side="right") - 1
    out = np.where(idx >= 0, p_mid[np.clip(idx, 0, len(p_mid) - 1)], np.nan)
    out = np.where(query > t_mid[-1], np.nan, out)                 # no future data past the tape
    return out


def markout_curve(t_trade, p_fill, is_buyer_maker, t_mid, p_mid, horizons):
    """Mean passive markout at each horizon, in **basis points** of the fill price.

    ``t_trade``/``p_fill``/``is_buyer_maker`` describe the trades; ``t_mid``/``p_mid``
    the (sorted) mid series used to look up ``mid_{t+h}``. Returns a dict
    ``{h: mean_markout_bps}`` and a parallel ``{h: n_trades_used}`` count dict."""
    t_trade = np.asarray(t_trade, dtype=float)
    p_fill = np.asarray(p_fill, dtype=float)
    sign = np.where(np.asarray(is_buyer_maker, dtype=bool), 1.0, -1.0)   # maker bought -> +1
    t_mid = np.asarray(t_mid, dtype=float)
    p_mid = np.asarray(p_mid, dtype=float)

    means, counts = {}, {}
    for h in horizons:
        future_mid = _mid_at(t_mid, p_mid, t_trade + h)
        mo = (future_mid - p_fill) * sign
        bps = 1e4 * mo / p_fill
        valid = np.isfinite(bps)
        means[h] = float(np.mean(bps[valid])) if valid.any() else float("nan")
        counts[h] = int(valid.sum())
    return means, counts
