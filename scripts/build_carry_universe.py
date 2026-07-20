#!/usr/bin/env python3
"""Broaden carry universe: fetch funding + perp daily close for ~70 more liquid
perps (point-in-time, incl. delisted LUNA/FTT/SRM), merge into *_full matrices."""
import zipfile, time
from pathlib import Path
import numpy as np, pandas as pd, requests
import concurrent.futures as cf

DATA=Path("/Users/jeevaa/QRJeevaa/crypto_data"); RAW=DATA/"raw_futures"; PROC=DATA/"processed"
RAW.mkdir(parents=True,exist_ok=True)
FUT="https://data.binance.vision/data/futures/um/monthly"
KCOLS=["open_time","open","high","low","close","volume","close_time","quote_volume","num_trades","taker_buy_base","taker_buy_quote","ignore"]
syms=open("/tmp/carry_new.txt").read().split()
def months(s,e):
    p,pe=pd.Period(s,"M"),pd.Period(e,"M"); o=[]
    while p<=pe: o.append((p.year,p.month)); p+=1
    return o
MO=months("2019-09","2026-05")
def to_utc(ms):
    ms=pd.to_numeric(ms,errors="coerce"); return pd.to_datetime(ms,unit=("us" if ms.max()>10**14 else "ms"),utc=True)
sess=requests.Session()
def dl(url,fp):
    if fp.exists(): return fp
    try: r=sess.get(url,timeout=20)
    except requests.RequestException: return None
    if r.status_code!=200: return None
    fp.write_bytes(r.content); return fp
def rd(fp,names=None,header=None):
    with zipfile.ZipFile(fp) as z, z.open(z.namelist()[0]) as f:
        return pd.read_csv(f,header=header,names=names)
def funding(sym):
    fr=[]
    for y,m in MO:
        fn=f"{sym}-fundingRate-{y}-{m:02d}.zip"; p=dl(f"{FUT}/fundingRate/{sym}/{fn}",RAW/fn)
        if p is None: continue
        try: df=rd(p,header=0)
        except: continue
        if "calc_time" not in df or "last_funding_rate" not in df: continue
        fr.append(pd.Series(df["last_funding_rate"].values,index=to_utc(df["calc_time"])))
    if not fr: return None
    s=pd.concat(fr).sort_index(); s=s[~s.index.duplicated(keep="last")]; return s.resample("1D").sum()
def perp(sym):
    fr=[]
    for y,m in MO:
        fn=f"{sym}-1d-{y}-{m:02d}.zip"; p=dl(f"{FUT}/klines/{sym}/1d/{fn}",RAW/fn)
        if p is None: continue
        try: df=rd(p,names=KCOLS,header=None)
        except: continue
        df=df[pd.to_numeric(df["open_time"],errors="coerce").notna()]
        fr.append(pd.Series(pd.to_numeric(df["close"]).values,index=to_utc(df["open_time"])))
    if not fr: return None
    c=pd.concat(fr).sort_index(); return c[~c.index.duplicated(keep="last")]
def one(sym): return sym, funding(sym), perp(sym)
fout,pout={}, {}
t0=time.time()
with cf.ThreadPoolExecutor(max_workers=20) as ex:
    for i,(sym,f,p) in enumerate(ex.map(one,syms),1):
        if f is not None: fout[sym]=f
        if p is not None: pout[sym]=p
        if i%15==0: print(f"  {i}/{len(syms)} f={len(fout)} p={len(pout)} ({time.time()-t0:.0f}s)",flush=True)
print(f"fetched funding={len(fout)} perp={len(pout)} in {time.time()-t0:.0f}s",flush=True)

oldf=pd.read_parquet(PROC/"funding_daily.parquet"); oldp=pd.read_parquet(PROC/"perp_close_1d.parquet")
idx=oldf.index
def build(d,old):
    m=pd.DataFrame(d); m.index=m.index.normalize()
    m=m[~m.index.duplicated(keep="last")].reindex(idx)
    add=[c for c in m.columns if c not in old.columns]
    return pd.concat([old,m[add]],axis=1)
fm=build(fout,oldf); pm=build(pout,oldp)
fm.to_parquet(PROC/"funding_daily_full.parquet"); pm.to_parquet(PROC/"perp_close_1d_full.parquet")
print(f"funding_daily_full: {oldf.shape[1]} -> {fm.shape[1]}",flush=True)
print(f"perp_close_1d_full: {oldp.shape[1]} -> {pm.shape[1]}",flush=True)
print("DONE carry expansion")
