"""Turning a :class:`~mmlab.simulate.SimResult` into the numbers that decide the
project. The centrepiece is the **PnL decomposition** — splitting total PnL into
the spread captured and the cost of holding inventory — because the contrast
between those two pieces for the naive vs the Avellaneda-Stoikov strategy is the
whole teaching point.

The decomposition rests on one accounting identity, asserted in
:func:`mmlab.selftest` and the test suite::

    total PnL  ==  spread PnL  +  inventory PnL

- **spread PnL** = sum over fills of ``(fill_price - mid_at_fill) * (+1 sell, -1 buy)``,
  i.e. the half-spread earned on each fill. Always the market maker's "gross edge".
- **inventory PnL** = sum over steps of ``q_t * (S_{t+1} - S_t)`` where ``q_t`` is the
  inventory held over that step — the mark-to-market of carrying a position while the
  mid moves. This is where adverse selection shows up.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def decompose_pnl(res):
    """Return per-path ``(total, spread, inventory)`` PnL arrays. ``total`` is an
    independent mark-to-market (cash + inventory at the final mid), so checking it
    equals ``spread + inventory`` genuinely tests the bookkeeping rather than
    restating it."""
    spread = res.spread_pnl
    inventory = np.sum(res.inventory[:, :-1] * np.diff(res.mid, axis=1), axis=1)
    total = res.total_pnl
    return total, spread, inventory


def accounting_residual(res):
    """Max absolute violation of ``total == spread + inventory`` across paths — the
    quantity check 3 drives to floating-point zero."""
    total, spread, inventory = decompose_pnl(res)
    return float(np.max(np.abs(total - (spread + inventory))))


def sharpe(pnl):
    """Cross-path PnL Sharpe: ``mean / std``. Dimensionless; not annualised, because
    these are terminal PnLs of independent fixed-horizon runs, not a return series."""
    pnl = np.asarray(pnl, dtype=float)
    s = pnl.std(ddof=1)
    return float(pnl.mean() / s) if s > 0 else float("nan")


def inventory_std_over_time(res):
    """Cross-path standard deviation of inventory at each step — the curve that
    separates the strategies: naive random-walks (std grows like ``sqrt(t)``), AS
    mean-reverts (std plateaus)."""
    return res.inventory.std(axis=0, ddof=1)


def summary(name, res):
    """One row of the Layer-2 comparison table for a single strategy."""
    total, spread, inventory = decompose_pnl(res)
    term_inv = res.terminal_inventory
    return {
        "strategy": name,
        "mean_pnl": float(total.mean()),
        "std_pnl": float(total.std(ddof=1)),
        "pnl_sharpe": sharpe(total),
        "spread_pnl": float(spread.mean()),
        "inventory_pnl": float(inventory.mean()),
        "term_inv_mean": float(term_inv.mean()),
        "term_inv_std": float(term_inv.std(ddof=1)),
        "term_inv_maxabs": float(np.abs(term_inv).max()),
        "bid_fills": float(res.bid_fills.sum(axis=1).mean()),
        "ask_fills": float(res.ask_fills.sum(axis=1).mean()),
    }


def comparison_table(results: dict):
    """Assemble the Layer-2 deliverable: one row per strategy. ``results`` maps a
    display name to its :class:`SimResult`."""
    return pd.DataFrame([summary(name, res) for name, res in results.items()]).set_index("strategy")
