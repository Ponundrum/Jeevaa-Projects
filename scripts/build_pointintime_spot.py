#!/usr/bin/env python3
"""Fetch daily klines for the 260 missing/delisted coins and merge into complete
point-in-time daily matrices (close/high/low/quote_volume/taker_buy_base).
Fixes survivorship bias: adds coins that traded 2020-2023 but later delisted
(LUNA, FTT, SRM, ...) or were simply never fetched.
"""
import io, zipfile, time, sys
from pathlib import Path
import numpy as np, pandas as pd, requests
import concurrent.futures as cf

DATA = Path("/Users/jeevaa/QRJeevaa/crypto_data")
RAW  = DATA / "raw_delisted"; RAW.mkdir(parents=True, exist_ok=True)
PROC = DATA / "processed"
SPOT = "https://data.binance.vision/data/spot/monthly/klines"
COLS = ["open_time","open","high","low","close","volume","close_time",
        "quote_volume","num_trades","taker_buy_base","taker_buy_quote","ignore"]
START, END = "2020-01", "2026-05"
syms = open("/tmp/inwindow.txt").read().split()

def months(s,e):
    p,pe=pd.Period(s,"M"),pd.Period(e,"M"); out=[]
    while p<=pe: out.append((p.year,p.month)); p+=1
    return out
MONTHS = months(START,END)

def to_utc(ms):
    ms=pd.to_numeric(ms,errors="coerce")
    return pd.to_datetime(ms, unit=("us" if ms.max()>10**14 else "ms"), utc=True)

sess = requests.Session()
def get_month(sym,y,m):
    fn=f"{sym}-1d-{y}-{m:02d}.zip"; fp=RAW/fn
    if not fp.exists():
        try:
            r=sess.get(f"{SPOT}/{sym}/1d/{fn}",timeout=20)
        except requests.RequestException: return None
        if r.status_code!=200: return None
        fp.write_bytes(r.content)
    try:
        with zipfile.ZipFile(fp) as z, z.open(z.namelist()[0]) as f:
            df=pd.read_csv(f,header=None,names=COLS)
    except Exception: return None
    df=df[pd.to_numeric(df["open_time"],errors="coerce").notna()]
    if df.empty: return None
    idx=to_utc(df["open_time"])
    return pd.DataFrame({c:pd.to_numeric(df[c]).values for c in
                         ["high","low","close","quote_volume","taker_buy_base"]}, index=idx)

def fetch_sym(sym):
    fr=[get_month(sym,y,m) for y,m in MONTHS]; fr=[x for x in fr if x is not None]
    if not fr: return sym,None
    d=pd.concat(fr).sort_index(); d=d[~d.index.duplicated(keep="last")]
    return sym, d

out={}
t0=time.time()
with cf.ThreadPoolExecutor(max_workers=24) as ex:
    for i,(sym,d) in enumerate(ex.map(fetch_sym,syms),1):
        if d is not None: out[sym]=d
        if i%20==0: print(f"  {i}/{len(syms)} fetched ok={len(out)} ({time.time()-t0:.0f}s)",flush=True)
print(f"fetched {len(out)}/{len(syms)} coins in {time.time()-t0:.0f}s",flush=True)

# Build new-coin matrices on a daily UTC index, then merge with existing 119
fields={"high":"high_1d","low":"low_1d","close":"close_1d",
        "quote_volume":"quote_volume_1d","taker_buy_base":"taker_buy_base_volume_1d"}
existing=pd.read_parquet(PROC/"close_1d.parquet")
full_idx=existing.index   # 2020-01-01..2026-05-31 daily, tz-naive
def mat(field):
    m=pd.DataFrame({s:d[field] for s,d in out.items()})
    m.index=m.index.normalize()              # keep UTC-aware to match existing index
    if full_idx.tz is None and m.index.tz is not None:
        m.index=m.index.tz_localize(None)
    m=m[~m.index.duplicated(keep="last")]
    return m.reindex(full_idx)

for field,fname in fields.items():
    new=mat(field)
    old=pd.read_parquet(PROC/f"{fname}.parquet")
    merged=pd.concat([old, new[[c for c in new.columns if c not in old.columns]]], axis=1)
    merged.to_parquet(PROC/f"{fname}_full.parquet")
    print(f"{fname}_full.parquet: {old.shape[1]} -> {merged.shape[1]} coins  ({field})",flush=True)
print("DONE building *_full.parquet matrices")
