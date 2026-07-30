"""Binance public-archive loader for the one thing Layer 3 needs: a tape of
aggregated trades (``aggTrades``) for one symbol, cached on disk. Same *approach*
as ``qsa/data.py`` in the sibling project — retry with backoff, cache a negative
sentinel only on a genuine 404 — re-implemented here rather than imported, so each
project stays standalone.

**Data choice, documented honestly.** ``aggTrades`` gives price, size, timestamp
and ``isBuyerMaker`` (the aggressor side) — enough for calibration and markouts.
The futures ``bookTicker`` feed *does* exist and would give a true bid/ask mid, but
at ~188 MB/day versus ~15 MB/day for aggTrades it is 10x heavier; :func:`probe_bookticker`
records that, and we fall back to a **trade-price mid proxy** (:func:`mid_grid`).
That proxy is contaminated by bid-ask bounce (~one spread wide, mean-reverting), so
it damages the shortest-horizon markouts most — which is why the README leans on the
longer horizons for its headline. Spot BTCUSDT is used: no funding/basis to muddy
the mid.
"""
from __future__ import annotations

import io
import random
import time
import zipfile
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from .config import CACHE

_AGG = "https://data.binance.vision/data/spot/daily/aggTrades"
_BOOK = "https://data.binance.vision/data/futures/um/daily/bookTicker"
_COLS = ["agg_id", "price", "qty", "first_id", "last_id", "ts", "is_buyer_maker", "is_best_match"]


def _download_cached(url, fp, retries=3):
    """Fetch ``url`` to ``fp`` (bytes), cached. Negative sentinel (empty file) only
    on a genuine 404; transient failures retry with backoff and are left uncached so
    a later run retries rather than permanently dropping a day."""
    fp.parent.mkdir(parents=True, exist_ok=True)
    if fp.exists():
        b = fp.read_bytes()
        return b if b else None
    delay = 0.5
    for attempt in range(retries):
        try:
            b = urlopen(Request(url, headers={"User-Agent": "mmlab"}), timeout=60).read()
            fp.write_bytes(b)
            return b
        except HTTPError as e:
            if e.code == 404:
                fp.write_bytes(b"")
                return None
        except Exception:
            pass
        if attempt < retries - 1:
            time.sleep(delay + random.uniform(0, 0.25))
            delay *= 2
    return None


def _parse_day(b):
    """Parse one aggTrades daily zip into a tidy frame: ``time`` (UTC seconds, float),
    ``price``, ``qty``, ``is_buyer_maker`` (bool). Handles both the header-less legacy
    CSVs and the newer header rows, and both millisecond and microsecond epochs."""
    with zipfile.ZipFile(io.BytesIO(b)) as z, z.open(z.namelist()[0]) as f:
        head = f.read(64)
    has_header = head[:6].lower() == b"agg_tr" or head[:11].lower() == b"aggtradeid,"
    with zipfile.ZipFile(io.BytesIO(b)) as z, z.open(z.namelist()[0]) as f:
        df = pd.read_csv(f, header=0 if has_header else None,
                         names=None if has_header else _COLS)
    df.columns = [str(c).lower() for c in df.columns]
    ts = pd.to_numeric(df.iloc[:, 5], errors="coerce").to_numpy(dtype="float64")
    scale = 1e6 if np.nanmedian(ts) > 1e15 else 1e3          # microseconds vs milliseconds
    ibm = df.iloc[:, 6]
    if ibm.dtype == object:
        ibm = ibm.astype(str).str.lower().isin(["true", "1"])
    return pd.DataFrame({
        "time": ts / scale,
        "price": pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy(dtype="float64"),
        "qty": pd.to_numeric(df.iloc[:, 2], errors="coerce").to_numpy(dtype="float64"),
        "is_buyer_maker": np.asarray(ibm, dtype=bool),
    }).dropna()


def load_aggtrades(symbol, dates, verbose=True):
    """Load and concatenate aggTrades for ``symbol`` over ``dates`` (``YYYY-MM-DD``
    strings), cached under ``config.CACHE``. Returns a time-sorted frame with columns
    ``time`` (UTC seconds), ``price``, ``qty``, ``is_buyer_maker``. Missing days are
    skipped with a note rather than raising."""
    frames = []
    for d in dates:
        fp = CACHE / "aggtrades" / f"{symbol}-aggTrades-{d}.zip"
        b = _download_cached(f"{_AGG}/{symbol}/{symbol}-aggTrades-{d}.zip", fp)
        if not b:
            if verbose:
                print(f"  [skip] {symbol} {d}: not available")
            continue
        day = _parse_day(b)
        frames.append(day)
        if verbose:
            print(f"  [ok]   {symbol} {d}: {len(day):,} trades")
    if not frames:
        raise RuntimeError("no aggTrades days loaded — check the dates / network")
    return pd.concat(frames).sort_values("time").reset_index(drop=True)


def mid_grid(trades, step=1.0, smooth=3):
    """Build a mid-price **proxy** on a uniform ``step``-second grid: the median trade
    price within each bin (median damps bid-ask bounce better than last-trade), then a
    short centred rolling median of width ``smooth`` bins, then forward-fill gaps.
    Returns ``(t_grid, mid)`` arrays for the markout lookups. This is a proxy, not a
    quoted mid — see the module docstring and the README limitations."""
    t = trades["time"].to_numpy()
    p = trades["price"].to_numpy()
    t0 = np.floor(t[0] / step) * step
    binned = pd.Series(p, index=((t - t0) / step).astype(int))
    mid = binned.groupby(level=0).median()
    full = pd.Series(index=np.arange(0, mid.index.max() + 1), dtype="float64")
    full.loc[mid.index] = mid.to_numpy()
    full = full.rolling(smooth, center=True, min_periods=1).median().ffill().bfill()
    t_grid = t0 + full.index.to_numpy(dtype="float64") * step
    return t_grid, full.to_numpy()


def probe_bookticker(symbol, date):
    """Check whether the futures ``bookTicker`` feed exists for ``symbol``/``date``,
    returning ``(available, size_bytes)`` without downloading it. Used to document the
    aggTrades fallback decision rather than to design around the (much heavier) feed."""
    url = f"{_BOOK}/{symbol}/{symbol}-bookTicker-{date}.zip"
    try:
        with urlopen(Request(url, headers={"User-Agent": "mmlab", "Range": "bytes=0-0"}),
                     timeout=30) as r:
            cr = r.headers.get("Content-Range", "")
            size = int(cr.split("/")[-1]) if "/" in cr else -1
            return True, size
    except HTTPError as e:
        return (False, -1) if e.code == 404 else (True, -1)
    except Exception:
        return False, -1
