"""Real futures **quotes** (the ``bookTicker`` feed) — the actual best bid/ask,
replacing the trade-price mid proxy for notebook 02's Layer-3 measurements. This is
the upgrade the proxy could not give: a real mid (no lag), a directly observable
**touch half-spread** ``(ask-bid)/2`` (the ceiling on what a maker at the touch
earns), and a ``kappa`` fitted against genuine book penetration rather than ~1s of
volatility. The aggTrades path in ``data.py`` stays as-is — both are needed for the
proxy-vs-quotes comparison.

Two things this module is careful about, both non-negotiable:

- **Sequence landmine.** Binance's own tracker (binance-public-data#305) reports that
  some Jan-2024 BTC/ETH ``bookTicker`` files are *out of time order*. Everything
  downstream (``np.searchsorted``) assumes sorted timestamps, so :func:`load_bookticker`
  sorts by ``update_id`` (the matching-engine sequence number), **asserts**
  ``transaction_time`` is non-decreasing afterwards, and reports how many rows were out
  of order (for BTCUSDT 2024-01-15 that count is 0 — the file is already ordered — but
  the guard stays).
- **Causality.** A trade at time ``t`` is joined to the quote strictly *before* ``t``
  (:func:`quote_at`), never one at or after it — a quote reflecting the trade's own
  effect on the book would be lookahead. Pinned by ``tests/test_quotes.py``.
"""
from __future__ import annotations

import hashlib
import io
import zipfile
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from . import data as _data
from .config import CACHE

_BOOK = "https://data.binance.vision/data/futures/um/daily/bookTicker"
_FUT_AGG = "https://data.binance.vision/data/futures/um/daily/aggTrades"
_USECOLS = ["update_id", "best_bid_price", "best_bid_qty",
            "best_ask_price", "best_ask_qty", "transaction_time"]
_DTYPE = {"update_id": "int64", "best_bid_price": "float64", "best_bid_qty": "float64",
          "best_ask_price": "float64", "best_ask_qty": "float64", "transaction_time": "int64"}


def _fetch(url, fp, timeout=180):
    """Download ``url`` to ``fp`` (bytes), reusing the cache if present."""
    fp.parent.mkdir(parents=True, exist_ok=True)
    if fp.exists():
        return fp.read_bytes()
    b = urlopen(Request(url, headers={"User-Agent": "mmlab"}), timeout=timeout).read()
    fp.write_bytes(b)
    return b


def load_futures_aggtrades(symbol, dates, verbose=True):
    """Futures ``aggTrades`` (the same market as ``bookTicker``) — needed because the
    spot tape in ``data.py`` trades at a basis to the futures book, so joining spot
    trades to futures quotes would measure the basis, not book penetration. Reuses the
    format-agnostic parser in ``data.py``; same tidy frame (``time``, ``price``, ``qty``,
    ``is_buyer_maker``)."""
    frames = []
    for d in dates:
        fp = CACHE / "fut_aggtrades" / f"{symbol}-aggTrades-{d}.zip"
        b = _data._download_cached(f"{_FUT_AGG}/{symbol}/{symbol}-aggTrades-{d}.zip", fp)
        if not b:
            if verbose:
                print(f"  [skip] futures {symbol} {d}: not available")
            continue
        day = _data._parse_day(b)
        frames.append(day)
        if verbose:
            print(f"  [ok]   futures {symbol} {d}: {len(day):,} trades")
    if not frames:
        raise RuntimeError("no futures aggTrades days loaded — check the dates / network")
    return pd.concat(frames).sort_values("time").reset_index(drop=True)


def load_bookticker(symbol, date, verbose=True):
    """Download, checksum, parse, **sort**, and cache one day of futures ``bookTicker``.

    Returns a frame with columns ``time`` (UTC seconds, float), ``bid``, ``ask``,
    ``bid_qty``, ``ask_qty``, ordered by time. The derived frame is cached as parquet
    (the raw ~188 MB zip is cached too, only to avoid re-downloading). Asserts
    ``time`` is non-decreasing after the sequence sort."""
    parquet = CACHE / "bookticker" / f"{symbol}-{date}.parquet"
    if parquet.exists():
        return pd.read_parquet(parquet)

    url = f"{_BOOK}/{symbol}/{symbol}-bookTicker-{date}.zip"
    zp = CACHE / "bookticker" / f"{symbol}-bookTicker-{date}.zip"
    b = _fetch(url, zp)
    try:                                                          # verify sha256 if the companion exists
        cs = urlopen(Request(url + ".CHECKSUM", headers={"User-Agent": "mmlab"}), timeout=30).read()
        want = cs.decode().split()[0]
        got = hashlib.sha256(b).hexdigest()
        if got != want:
            raise RuntimeError(f"bookTicker checksum mismatch for {symbol} {date}: {got} != {want}")
    except HTTPError:
        pass                                                      # no checksum companion — skip

    with zipfile.ZipFile(io.BytesIO(b)) as z, z.open(z.namelist()[0]) as f:
        df = pd.read_csv(f, usecols=_USECOLS, dtype=_DTYPE)

    # Sequence landmine: sort by the matching-engine sequence number and report the count.
    uid = df["update_id"].to_numpy()
    out_of_order = int((uid[1:] < uid[:-1]).sum())
    if out_of_order:
        df = df.sort_values("update_id", kind="stable").reset_index(drop=True)
    tt = df["transaction_time"].to_numpy()
    assert np.all(np.diff(tt) >= 0), "transaction_time not monotone after sorting by update_id"

    out = pd.DataFrame({
        "time": tt / 1000.0,                                      # ms -> s (same clock as aggTrades)
        "bid": df["best_bid_price"].to_numpy(),
        "ask": df["best_ask_price"].to_numpy(),
        "bid_qty": df["best_bid_qty"].to_numpy(),
        "ask_qty": df["best_ask_qty"].to_numpy(),
    })
    parquet.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(parquet)
    if verbose:
        print(f"  [ok] {symbol} {date}: {len(out):,} quotes; {out_of_order} rows out of sequence (sorted)")
    return out


def quote_mid(quotes):
    """The real mid, ``(bid + ask) / 2``."""
    return (quotes["bid"].to_numpy() + quotes["ask"].to_numpy()) / 2.0


def touch_half_spread(quotes):
    """``(ask - bid) / 2`` per quote update — the most a maker resting at the touch can
    capture per fill (in price units)."""
    return (quotes["ask"].to_numpy() - quotes["bid"].to_numpy()) / 2.0


def quote_at(quotes, t_query):
    """The quote in effect **strictly before** each ``t_query`` — the causal join. A
    trade must see the book as it was just *before* it, not the update its own fill
    produced. Returns aligned ``(bid, ask)`` arrays; ``NaN`` where no prior quote
    exists. ``quotes`` must be time-sorted (``load_bookticker`` guarantees it)."""
    t = quotes["time"].to_numpy()
    bid = quotes["bid"].to_numpy()
    ask = quotes["ask"].to_numpy()
    idx = np.searchsorted(t, t_query, side="left") - 1            # last quote with time < t_query
    valid = idx >= 0
    clipped = np.clip(idx, 0, len(t) - 1)
    b = np.where(valid, bid[clipped], np.nan)
    a = np.where(valid, ask[clipped], np.nan)
    return b, a


def microprice(quotes):
    """Queue-imbalance-weighted fair value ``(bid*ask_qty + ask*bid_qty) /
    (bid_qty + ask_qty)`` — a better short-horizon fair value than the mid because it
    leans toward the side with less resting size (which is where the price is headed)."""
    b = quotes["bid"].to_numpy()
    a = quotes["ask"].to_numpy()
    bq = quotes["bid_qty"].to_numpy()
    aq = quotes["ask_qty"].to_numpy()
    return (b * aq + a * bq) / (bq + aq)
