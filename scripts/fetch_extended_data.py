#!/usr/bin/env python3
"""
Extended data acquisition for QRJeevaa.

Adds two data types the spot-only pipeline can't provide:
  - Funding rates  (Binance USDT-M perpetuals) -> enables CARRY strategies
  - Perp daily klines                          -> enables BASIS (perp vs spot)
and optionally extends SPOT daily history back to 2017 for the oldest coins
(USDT-M futures only launched 2019-09, so funding/perp history starts there).

Heavy downloads live here, NOT in the notebook: this script populates
crypto_data/ with processed parquet that the notebook simply loads.

Usage:
    python scripts/fetch_extended_data.py                 # default core, futures era
    python scripts/fetch_extended_data.py --spot-from 2017-01   # also extend spot
All downloads are cache-guarded and exception-safe; re-running is cheap.
"""
import argparse
import io
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "crypto_data"
RAW_FUT = DATA_DIR / "raw_futures"
RAW_SPOT = DATA_DIR / "raw"
PROC = DATA_DIR / "processed"
for d in (RAW_FUT, RAW_SPOT, PROC):
    d.mkdir(parents=True, exist_ok=True)

SPOT_BASE = "https://data.binance.vision/data/spot/monthly"
FUT_BASE = "https://data.binance.vision/data/futures/um/monthly"

# Liquid core with long perp history. Extend freely; carry wants breadth.
DEFAULT_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT",
    "LINKUSDT", "LTCUSDT", "TRXUSDT", "ETCUSDT", "SOLUSDT", "AVAXUSDT",
    "DOTUSDT", "MATICUSDT", "ATOMUSDT", "UNIUSDT", "FILUSDT", "AAVEUSDT",
    "VETUSDT", "CHZUSDT",
]
KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "num_trades", "taker_buy_base", "taker_buy_quote", "ignore",
]


def month_range(start, end):
    s, e = pd.Period(start, "M"), pd.Period(end, "M")
    p = s
    out = []
    while p <= e:
        out.append((p.year, p.month))
        p += 1
    return out


def download(url, save_path, session):
    """Cache-guarded, exception-safe download. Returns path or None."""
    if save_path.exists():
        return save_path
    try:
        r = session.get(url, timeout=30)
    except requests.RequestException:
        return None
    if r.status_code != 200:
        return None
    save_path.write_bytes(r.content)
    return save_path


def read_zip_csv(path, names=None, header=None):
    with zipfile.ZipFile(path) as z:
        with z.open(z.namelist()[0]) as f:
            return pd.read_csv(f, header=header, names=names)


def to_utc(ms):
    ms = pd.to_numeric(ms, errors="coerce")
    unit = "us" if ms.max() > 10**14 else "ms"
    return pd.to_datetime(ms, unit=unit, utc=True)


def fetch_funding(symbol, months, session):
    frames = []
    for y, m in months:
        fn = f"{symbol}-fundingRate-{y}-{m:02d}.zip"
        p = download(f"{FUT_BASE}/fundingRate/{symbol}/{fn}", RAW_FUT / fn, session)
        if p is None:
            continue
        try:
            df = read_zip_csv(p, header=0)
        except Exception:
            continue
        if "calc_time" not in df.columns or "last_funding_rate" not in df.columns:
            continue
        s = pd.Series(df["last_funding_rate"].values, index=to_utc(df["calc_time"]))
        frames.append(s)
    if not frames:
        return None
    s = pd.concat(frames).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s.resample("1D").sum()              # daily funding = sum of 8h payments


def fetch_perp_close(symbol, interval, months, session):
    frames = []
    for y, m in months:
        fn = f"{symbol}-{interval}-{y}-{m:02d}.zip"
        p = download(f"{FUT_BASE}/klines/{symbol}/{interval}/{fn}", RAW_FUT / fn, session)
        if p is None:
            continue
        try:
            df = read_zip_csv(p, names=KLINE_COLS, header=None)
        except Exception:
            continue
        df = df[pd.to_numeric(df["open_time"], errors="coerce").notna()]
        c = pd.Series(pd.to_numeric(df["close"]).values, index=to_utc(df["open_time"]))
        frames.append(c)
    if not frames:
        return None
    c = pd.concat(frames).sort_index()
    return c[~c.index.duplicated(keep="last")]


def fetch_spot_close(symbol, interval, months, session):
    frames = []
    for y, m in months:
        fn = f"{symbol}-{interval}-{y}-{m:02d}.zip"
        p = download(f"{SPOT_BASE}/klines/{symbol}/{interval}/{fn}", RAW_SPOT / fn, session)
        if p is None:
            continue
        try:
            df = read_zip_csv(p, names=KLINE_COLS, header=None)
        except Exception:
            continue
        df = df[pd.to_numeric(df["open_time"], errors="coerce").notna()]
        c = pd.Series(pd.to_numeric(df["close"]).values, index=to_utc(df["open_time"]))
        frames.append(c)
    if not frames:
        return None
    c = pd.concat(frames).sort_index()
    return c[~c.index.duplicated(keep="last")]


def build_matrix(series_by_symbol):
    series_by_symbol = {k: v for k, v in series_by_symbol.items() if v is not None}
    if not series_by_symbol:
        return None
    return pd.DataFrame(series_by_symbol).sort_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="*", default=DEFAULT_SYMBOLS)
    ap.add_argument("--start", default="2019-09")      # USDT-M futures launch
    ap.add_argument("--end", default="2026-05")
    ap.add_argument("--spot-from", default=None,
                    help="If set (e.g. 2017-01), also extend spot daily history from here.")
    args = ap.parse_args()

    fut_months = month_range(args.start, args.end)
    session = requests.Session()
    print(f"Fetching funding + perp for {len(args.symbols)} symbols, "
          f"{args.start}..{args.end} ({len(fut_months)} months each)")

    funding, perp = {}, {}
    for i, sym in enumerate(args.symbols, 1):
        t0 = time.time()
        funding[sym] = fetch_funding(sym, fut_months, session)
        perp[sym] = fetch_perp_close(sym, "1d", fut_months, session)
        fok = "ok" if funding[sym] is not None else "--"
        pok = "ok" if perp[sym] is not None else "--"
        print(f"  [{i:>2}/{len(args.symbols)}] {sym:12s} funding:{fok} perp:{pok} "
              f"({time.time()-t0:.1f}s)")

    fmat = build_matrix(funding)
    pmat = build_matrix(perp)
    if fmat is not None:
        fmat.to_parquet(PROC / "funding_daily.parquet")
        print(f"Saved funding_daily.parquet {fmat.shape} "
              f"({fmat.index.min().date()}..{fmat.index.max().date()})")
    if pmat is not None:
        pmat.to_parquet(PROC / "perp_close_1d.parquet")
        print(f"Saved perp_close_1d.parquet {pmat.shape}")

    # Basis = perp/spot - 1, where spot exists in the processed set.
    spot_path = PROC / "close_1d.parquet"
    if pmat is not None and spot_path.exists():
        spot = pd.read_parquet(spot_path)
        common = [c for c in pmat.columns if c in spot.columns]
        if common:
            idx = pmat.index.intersection(spot.index)
            basis = (pmat.loc[idx, common] / spot.loc[idx, common]) - 1.0
            basis.to_parquet(PROC / "basis_1d.parquet")
            print(f"Saved basis_1d.parquet {basis.shape} ({len(common)} overlapping symbols)")

    if args.spot_from:
        spot_months = month_range(args.spot_from, args.end)
        print(f"\nExtending spot daily history from {args.spot_from} for {len(args.symbols)} symbols")
        ext = {}
        for i, sym in enumerate(args.symbols, 1):
            ext[sym] = fetch_spot_close(sym, "1d", spot_months, session)
            print(f"  [{i:>2}/{len(args.symbols)}] {sym:12s} "
                  f"spot:{'ok' if ext[sym] is not None else '--'}")
        emat = build_matrix(ext)
        if emat is not None:
            emat.to_parquet(PROC / "spot_close_1d_extended.parquet")
            print(f"Saved spot_close_1d_extended.parquet {emat.shape} "
                  f"({emat.index.min().date()}..{emat.index.max().date()})")

    print("\nDone.")


if __name__ == "__main__":
    main()
