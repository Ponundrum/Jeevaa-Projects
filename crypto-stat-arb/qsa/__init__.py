"""qsa — a small crypto statistical-arbitrage research toolkit.

The package is the *plumbing* (self-contained data download, backtest engine,
metrics, signal formulas); the notebooks are the *research* (they construct each
book from these primitives, in the open, so a reviewer can read the logic).

Typical use::

    from qsa import Dataset, config as C
    from qsa.engine import dn_weights, backtest, sharpe, sortino, maxdd
    from qsa.signals import near_high, idio_vol

    ds = Dataset.load()                       # downloads on first run, then cached
    w = dn_weights(near_high(ds, 90), ds)     # market-neutral, liquidity+short screened
    net, _ = backtest(w, ds.ret_all, rebal=7)
    sharpe(net.loc[C.VAL[0]:C.VAL[1]])        # out-of-sample Sharpe
"""
from . import config
from .config import TCOST, ANN, TRAIN, VAL, FULL, seg
from .data import Dataset, SPOT_SYMBOLS, CARRY_SYMBOLS, PERP_SYMBOLS
from .plotting import apply_style, style_table, CLR

__all__ = [
    "config", "TCOST", "ANN", "TRAIN", "VAL", "FULL", "seg",
    "Dataset", "SPOT_SYMBOLS", "CARRY_SYMBOLS", "PERP_SYMBOLS",
    "apply_style", "style_table", "CLR",
]
