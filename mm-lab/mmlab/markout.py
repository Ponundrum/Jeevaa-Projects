"""Markouts — the empirical heart of the project (Layer 3b).

For every trade on the tape, signed from the **passive** maker's point of view,
we measure how the mid moves afterwards::

    markout(h) = (mid_{t+h} - fill_price) * (+1 if the maker bought, -1 if sold)

Averaged across trades at several horizons ``h`` and reported in basis points, this
is the adverse selection facing a resting quote: the number is expected to be
*negative and worsening with horizon* — the market systematically moves against
whoever provided the liquidity. Comparing its magnitude to the half-spread a maker
thinks they capture is the punchline of the whole repo.

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


def half_spread_captured_bps(bid_depths, ask_depths, mid_level):
    """The other half of the punchline: the mean half-spread a maker captures per
    fill, in bps, estimated as the average penetration depth at which trades arrive
    (i.e. how far from the mid a resting quote typically fills). Compared against
    ``|markout|`` at the holding horizon, this says whether naive market making is a
    business or a subsidy to the takers."""
    depths = np.concatenate([np.asarray(bid_depths, dtype=float),
                             np.asarray(ask_depths, dtype=float)])
    return float(1e4 * np.mean(depths) / mid_level)
