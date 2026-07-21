"""Market-data snapshot for the calibration notebook — a SINGLE free option-chain
snapshot from Yahoo Finance (`yfinance`), cleaned and **cached to disk on first
pull** so every later run is reproducible and offline. A snapshot calibration is
not a backtest; it is one photograph of the surface on one day, and the notebook
says so.

Cleaning: drop zero-bid / crossed / stale quotes, use the mid, keep out-of-the-
money options (the liquid side), filter to a sane moneyness and maturity band, and
back out the market-implied forward per expiry via put-call parity (so the surface
is built on the forward the market is actually pricing, not an assumed dividend).
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

from .config import CACHE


def _rate_from_irx(default=0.05):
    """Continuous risk-free proxy from the 13-week T-bill yield (^IRX, in percent)."""
    try:
        import yfinance as yf
        y = yf.Ticker("^IRX").history(period="5d")["Close"].dropna()
        return float(y.iloc[-1]) / 100.0 if len(y) else default
    except Exception:
        return default


def _implied_forward(calls, puts, r, T):
    """Market-implied forward from put-call parity: on strikes quoted on both sides,
    ``C - P = e^{-rT}(F - K)``, so a line through ``(K, C-P)`` gives ``F``."""
    m = calls.merge(puts, on="strike", suffixes=("_c", "_p"))
    m = m[(m["mid_c"] > 0) & (m["mid_p"] > 0)]
    if len(m) < 3:
        return np.nan
    slope, intercept = np.polyfit(m["strike"], m["mid_c"] - m["mid_p"], 1)   # C-P = -e^{-rT} K + e^{-rT} F
    disc = -slope
    return float(intercept / disc) if disc > 0 else np.nan


def option_chain_snapshot(ticker="SPY", n_expiries=8, moneyness=(0.80, 1.20),
                          min_days=7, max_days=400, refresh=False, verbose=True):
    """Return ``(chain_df, meta)``. ``chain_df`` has columns ``T, K, price, kind, F, q``;
    ``meta`` has ``spot, r, asof, ticker``. Cached under ``_cache/``; set
    ``refresh=True`` to re-pull. Raises with a clear message if there is no cache
    and the network is unavailable."""
    CACHE.mkdir(parents=True, exist_ok=True)
    fp = CACHE / f"{ticker}_chain.parquet"
    mp = CACHE / f"{ticker}_meta.json"
    if fp.exists() and mp.exists() and not refresh:
        if verbose:
            print(f"Loading cached {ticker} snapshot from {fp} (offline, reproducible).")
        return pd.read_parquet(fp), json.loads(mp.read_text())

    try:
        import yfinance as yf
    except Exception as e:  # pragma: no cover
        raise RuntimeError("yfinance is required for a fresh snapshot: pip install yfinance") from e

    try:
        tk = yf.Ticker(ticker)
        spot = float(tk.history(period="5d")["Close"].dropna().iloc[-1])
        r = _rate_from_irx()
        asof = pd.Timestamp.utcnow().tz_localize(None).normalize()
        # Select expiries spread GEOMETRICALLY across the maturity band, so the
        # snapshot is a proper term structure (short-dated to ~1y), not just the
        # nearest weeklies.
        avail = [(e, (pd.Timestamp(e) - asof).days) for e in tk.options]
        avail = [(e, d) for e, d in avail if min_days <= d <= max_days]
        targets = np.geomspace(min_days, max_days, n_expiries)
        chosen, seen = [], set()
        for td in targets:
            e, d = min(avail, key=lambda x: abs(x[1] - td))
            if e not in seen:
                seen.add(e)
                chosen.append(e)
        rows = []
        for exp in chosen:
            T = (pd.Timestamp(exp) - asof).days / 365.0
            oc = tk.option_chain(exp)
            calls, puts = oc.calls.copy(), oc.puts.copy()
            for df in (calls, puts):
                df["mid"] = (df["bid"] + df["ask"]) / 2
            F = _implied_forward(calls, puts, r, T)
            if not np.isfinite(F):
                F = spot * np.exp(r * T)
            q = r - np.log(F / spot) / T
            lo, hi = moneyness[0] * F, moneyness[1] * F
            for kind, df in (("call", calls), ("put", puts)):
                keep = (df["bid"] > 0) & (df["ask"] > df["bid"]) & (df["strike"] >= lo) & (df["strike"] <= hi)
                otm = (df["strike"] >= F) if kind == "call" else (df["strike"] < F)   # liquid OTM wing
                sel = df[keep & otm]
                for K, mid in zip(sel["strike"], sel["mid"]):
                    rows.append({"T": T, "K": float(K), "price": float(mid), "kind": kind, "F": F, "q": q})
        chain = pd.DataFrame(rows)
        meta = {"spot": spot, "r": r, "asof": str(asof.date()), "ticker": ticker}
        chain.to_parquet(fp)
        mp.write_text(json.dumps(meta, indent=1))
        if verbose:
            print(f"Pulled + cached {ticker} snapshot as of {meta['asof']}: {len(chain)} quotes, "
                  f"{chain['T'].nunique()} maturities, spot {spot:.2f}, r {r:.3%}.")
        return chain, meta
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            f"Could not fetch a live {ticker} snapshot ({type(e).__name__}: {e}). "
            "Yahoo may be rate-limiting; try again, or use a previously cached snapshot.") from e


def underlying_history(ticker="SPY", period="2y", refresh=False, verbose=True):
    """Cached daily close history of the underlying (for a realised-vol reference)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    fp = CACHE / f"{ticker}_history.parquet"
    if fp.exists() and not refresh:
        return pd.read_parquet(fp)["Close"]
    import yfinance as yf
    h = yf.Ticker(ticker).history(period=period)[["Close"]]
    h.index = h.index.tz_localize(None)
    h.to_parquet(fp)
    if verbose:
        print(f"Pulled + cached {ticker} history: {len(h)} days.")
    return h["Close"]
