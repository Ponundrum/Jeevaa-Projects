"""Self-contained data layer.

On first use this downloads ~6 years of DAILY history (spot OHLCV + perpetual
close + funding) straight from Binance's PUBLIC archive (``data.binance.vision``
— no API key, no account) and caches parquet under ``crypto_data/processed/``.
Every later run loads from disk in seconds.

The universe is POINT-IN-TIME: it includes coins that were liquid during
2020-2026 but have since been delisted or died (LUNA, FTT, SRM, ...), so it is
free of survivorship bias. ``Dataset.load()`` returns one object holding every
frame the strategies need, plus the liquidity / short-feasibility masks and the
BTC-residualised returns.
"""
import io
import zipfile
import concurrent.futures as cf
from urllib.request import urlopen

import numpy as np
import pandas as pd

from .config import DATA, ARCHIVE, TRAIN, seg

# --- 379-coin point-in-time spot pool (incl. since-delisted names) ---------
SPOT_SYMBOLS = [s + "USDT" for s in (
    "BTC ETH BNB XRP DOGE ADA SOL SHIB DOT MATIC LINK LTC AVAX TRX FTM FIL ETC EOS SAND VET CHZ "
    "BCH GALA MANA ATOM AXS NEAR XLM UNI SXP ICP SUSHI AAVE ALGO RUNE CRV GRT CAKE OMG ALICE HOT "
    "ONE EGLD ENJ WAVES ZIL LRC DYDX ZEC TLM XTZ YFI ONT CHR DENT ROSE XMR DASH IOST 1INCH QTUM "
    "KAVA REEF HBAR FET BAKE KSM C98 IOTA IOTX ANKR SNX JASMY COTI CELR MASK BAT INJ COMP OGN RVN "
    "LINA DAR ALPHA RSR STORJ CELO BAND MINA AR CTSI STX LIT ICX ARPA TWT OCEAN MKR RNDR TRB SC "
    "REN STMX KNC DODO FLOW ZRX ZEN SKL MTL NKN SUPER FLM BEL KEY WAXP DUSK QNT POND ACA ACH ACM "
    "ADX AERGO AGIX AGLD AION AKRO ALCX ALPACA ALPINE AMB AMP ANC ANT APE API3 APT ARB ARDR ASR "
    "ASTR AST ATA ATM AUCTION AUDIO AUTO AVA BADGER BAL BAR BEAM BETA BETH BICO BIFI BLZ BNT BNX "
    "BOND BSW BTCST BTG BTS BTTC BTT BURGER BZRX CFX CHESS CITY CKB CLV COCOS COMBO COS CTK CTXC "
    "CVC CVP CVX DATA DCR DEGO DEXE DF DGB DIA DNT DOCK DREP EDU ELF ENS EPS EPX ERD ERN FARM "
    "FIDA FIO FIRO FIS FLOKI FLUX FORTH FOR FRONT FTT FUN FXS GAL GAS GHST GLMR GLM GMT GMX GNO "
    "GNS GTC GTO GXS HARD HC HFT HIFI HIGH HIVE HNT HOOK IDEX ID ILV IMX IRIS JOE JST JUV KDA "
    "KEEP KLAY KMD KP3R LAZIO LDO LEND LEVER LOKA LOOM LPT LQTY LSK LTO LUNA LUNC MAGIC MAV MBL "
    "MBOX MCO MC MDT MDX MFT MIR MITH MLN MOB MOVR MULTI NANO NBS NBT NEO NEXO NMR NPXS NULS NU "
    "OAX OG OM ONG OOKI OP ORN OSMO OXT PEOPLE PEPE PERL PERP PHA PHB PLA PNT POLS POLY POLYX "
    "PORTO POWR PROM PROS PSG PUNDIX PYR QI QKC QUICK RAD RAMP RARE RAY RDNT REI REP REQ RIF RLC "
    "RPL SANTOS SCRT SFP SLP SNT SPELL SRM SSV STEEM STG STORM STPT STRAT STRAX SUI SUN SYN SYS "
    "TCT TFUEL THETA TKO TOMO TORN TRIBE TROY TRU T TVK UFT UMA UNFI USDS USTC UTK VGX VIB VIDT "
    "VITE VOXEL VTHO WAN WBTC WING WIN WNXM WOO WRX WTC XEC XEM XNO XVG XVS XZC YFII YGG ").split()]

# --- 90 perps that ever had funding history (carry universe) ---------------
CARRY_SYMBOLS = [s + "USDT" for s in (
    "BTC ETH BNB XRP DOGE ADA LINK LTC TRX ETC SOL AVAX DOT MATIC ATOM UNI FIL AAVE VET CHZ ARB "
    "OP APT SUI BCH FTM EOS LUNA APE NEAR SAND GMT CRV XMR RUNE SUSHI ALGO LDO XLM AGIX ZEC WAVES "
    "SRM GRT GALA EGLD XTZ ID MANA AXS SNX MAGIC KEEP FTT ZIL THETA ENS NEO RDNT HOOK ANC BTT "
    "DASH UNFI HNT PEOPLE SXP KAVA BZRX ROSE LEND DYDX SLP ICP FET YFI HBAR COMP MAV LQTY IMX EDU "
    "SSV INJ STG ONT LRC CAKE NU RSR ").split()]

# --- every spot coin that has (or had) a tradeable USDT perp -> shortable ---
PERP_SYMBOLS = set(s + "USDT" for s in (
    "1INCH AAVE ACH ADA AERGO AGIX AGLD AKRO ALGO ALICE ALPACA ALPHA ALPINE AMB ANC ANKR ANT APE API3 APT "
    "AR ARB ARPA ASR ASTR ATA ATOM AUCTION AUDIO AVA AVAX AXS BADGER BAKE BAL BAND BAT BCH BEL BICO BLZ BNB "
    "BNT BNX BOND BSW BTC BTCST BTS BTT BTTC BZRX C98 CAKE CELO CELR CFX CHESS CHR CHZ CKB COCOS COMBO COMP "
    "COS COTI CRV CTK CTSI CVC CVX DAR DASH DEGO DENT DEXE DF DGB DIA DODO DOGE DOT DUSK DYDX EDU EGLD ENJ "
    "ENS EOS ETC ETH FET FIDA FIL FIO FIS FLM FLOKI FLOW FLUX FORTH FRONT FTM FTT FUN FXS GAL GALA GAS GHST "
    "GLM GLMR GMT GMX GRT GTC HBAR HFT HIFI HIGH HIVE HNT HOOK HOT ICP ICX ID IDEX ILV IMX INJ IOST IOTA "
    "IOTX JASMY JOE JST KAVA KDA KEEP KEY KLAY KNC KSM LDO LEND LEVER LINA LINK LIT LOKA LOOM LPT LQTY LRC "
    "LSK LTC LUNA LUNC MAGIC MANA MASK MATIC MAV MBL MBOX MDT MINA MKR MLN MOVR MTL NEAR NEO NKN NMR NU NULS "
    "OCEAN OG OGN OM OMG ONE ONG ONT OP OXT PEOPLE PEPE PERP PHA PHB POLYX POWR PROM PUNDIX QNT QTUM QUICK "
    "RAD RARE RAY RDNT REEF REI REN RIF RLC RNDR ROSE RPL RSR RUNE RVN SAND SANTOS SC SCRT SFP SHIB SKL SLP "
    "SNT SNX SOL SPELL SRM SSV STEEM STG STMX STORJ STPT STRAX STX SUI SUN SUPER SUSHI SXP SYN SYS T THETA "
    "TLM TOMO TRB TROY TRU TRX TWT UMA UNFI UNI USTC VET VIDT VOXEL VTHO WAVES WAXP WOO XEM XLM XMR XRP XTZ "
    "XVG XVS YFI YFII YGG ZEC ZEN ZIL ZRX").split())

_SPOT = "https://data.binance.vision/data/spot/monthly/klines"
_FUT = "https://data.binance.vision/data/futures/um/monthly"
_KC = ["open_time", "open", "high", "low", "close", "volume", "close_time",
       "quote_volume", "n", "tbb", "tbq", "ig"]
_SMON = [(p.year, p.month) for p in pd.period_range("2020-01", "2026-05", freq="M")]
_FMON = [(p.year, p.month) for p in pd.period_range("2019-09", "2026-05", freq="M")]

_NEED = ["close_1d_full", "high_1d_full", "low_1d_full", "quote_volume_1d_full",
         "taker_buy_base_volume_1d_full", "funding_daily_full", "perp_close_1d_full"]


def _utc(ms):
    ms = pd.to_numeric(ms, errors="coerce")
    return pd.to_datetime(ms, unit=("us" if ms.max() > 10 ** 14 else "ms"), utc=True)


def _grab(url, fn, cache):
    fp = cache / fn
    if fp.exists():
        b = fp.read_bytes()
        return b if b else None
    try:
        b = urlopen(url, timeout=30).read()
    except Exception:
        fp.write_bytes(b"")
        return None
    fp.write_bytes(b)
    return b


def _csv(b, names=None, header=None):
    with zipfile.ZipFile(io.BytesIO(b)) as z, z.open(z.namelist()[0]) as f:
        return pd.read_csv(f, header=header, names=names)


def _spot_one(sym):
    fr = []
    for y, m in _SMON:
        b = _grab(f"{_SPOT}/{sym}/1d/{sym}-1d-{y}-{m:02d}.zip", f"{sym}-1d-{y}-{m:02d}.zip", ARCHIVE)
        if not b:
            continue
        try:
            d = _csv(b, names=_KC)
        except Exception:
            continue
        d = d[pd.to_numeric(d["open_time"], errors="coerce").notna()]
        if len(d):
            fr.append(pd.DataFrame({k: pd.to_numeric(d[k]).values for k in
                      ["high", "low", "close", "quote_volume", "tbb"]}, index=_utc(d["open_time"])))
    if not fr:
        return sym, None
    x = pd.concat(fr).sort_index()
    x = x[~x.index.duplicated(keep="last")]
    x.index = x.index.normalize()
    return sym, x


def _fut_one(sym):
    fr = []
    for y, m in _FMON:
        b = _grab(f"{_FUT}/fundingRate/{sym}/{sym}-fundingRate-{y}-{m:02d}.zip",
                  f"{sym}-fundingRate-{y}-{m:02d}.zip", ARCHIVE)
        if not b:
            continue
        try:
            d = _csv(b, header=0)
        except Exception:
            continue
        if "calc_time" in d and "last_funding_rate" in d:
            fr.append(pd.Series(d["last_funding_rate"].values, index=_utc(d["calc_time"])))
    fund = (pd.concat(fr).sort_index().pipe(lambda s: s[~s.index.duplicated(keep="last")])
            .resample("1D").sum()) if fr else None
    pr = []
    for y, m in _FMON:
        b = _grab(f"{_FUT}/klines/{sym}/1d/{sym}-1d-{y}-{m:02d}.zip", f"{sym}-fut1d-{y}-{m:02d}.zip", ARCHIVE)
        if not b:
            continue
        try:
            d = _csv(b, names=_KC)
        except Exception:
            continue
        d = d[pd.to_numeric(d["open_time"], errors="coerce").notna()]
        if len(d):
            pr.append(pd.Series(pd.to_numeric(d["close"]).values, index=_utc(d["open_time"])))
    perp = (pd.concat(pr).sort_index().pipe(lambda c: c[~c.index.duplicated(keep="last")])) if pr else None
    return sym, fund, perp


def download_if_needed(verbose=True):
    """Download + cache the daily dataset if the parquet files are not present."""
    DATA.mkdir(parents=True, exist_ok=True)
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    if all((DATA / f"{n}.parquet").exists() for n in _NEED):
        if verbose:
            print("Data cache present: loading from crypto_data/processed/ (no download needed).")
        return
    if verbose:
        print(f"First run: downloading daily data for {len(SPOT_SYMBOLS)} spot + "
              f"{len(CARRY_SYMBOLS)} perp symbols from data.binance.vision (~5-8 min once, then cached).")
    IDX = pd.date_range("2020-01-01", "2026-05-31", freq="D", tz="UTC")
    sd = {}
    with cf.ThreadPoolExecutor(max_workers=16) as ex:
        for i, (s, x) in enumerate(ex.map(_spot_one, SPOT_SYMBOLS), 1):
            if x is not None:
                sd[s] = x
            if verbose and i % 60 == 0:
                print(f"  spot {i}/{len(SPOT_SYMBOLS)}")
    for field, name in [("close", "close_1d_full"), ("high", "high_1d_full"), ("low", "low_1d_full"),
                        ("quote_volume", "quote_volume_1d_full"), ("tbb", "taker_buy_base_volume_1d_full")]:
        pd.DataFrame({s: x[field] for s, x in sd.items()}).reindex(IDX).to_parquet(DATA / f"{name}.parquet")
    fd, pp = {}, {}
    with cf.ThreadPoolExecutor(max_workers=16) as ex:
        for i, (s, f, p) in enumerate(ex.map(_fut_one, CARRY_SYMBOLS), 1):
            if f is not None:
                fd[s] = f
            if p is not None:
                pp[s] = p
    pd.DataFrame(fd).reindex(IDX).to_parquet(DATA / "funding_daily_full.parquet")
    pd.DataFrame(pp).reindex(IDX).to_parquet(DATA / "perp_close_1d_full.parquet")
    if verbose:
        print(f"Done: built {len(sd)} spot + {len(fd)} funding / {len(pp)} perp series, cached.")


class Dataset:
    """Everything the strategies read. Build with :meth:`Dataset.load`.

    Attributes
    ----------
    close_all, qvol_all, high_all, low_all, tbb_all : DataFrame
        Full point-in-time pool (379 coins), daily.
    ret_all, resid_all, beta_all : DataFrame
        Simple returns; BTC-residualised (idiosyncratic) returns; rolling beta.
    liq60, shortable, short_ok : DataFrame(bool)
        Point-in-time liquidity mask; has-a-perp mask; shortable = liquid & perp.
    UNIV : list[str]
        Top-15 tradeable universe (top by TRAIN-window $-volume).
    close, qvol, high, low, ret : DataFrame
        The same frames restricted to ``UNIV``.
    btc, ew : Series
        Market proxy / benchmark and equal-weight benchmark daily returns.
    funding, perp : DataFrame
        Daily funding and perp close for the carry universe.
    """

    def __init__(self, **kw):
        self.__dict__.update(kw)

    @classmethod
    def load(cls, top_n=15, liq_usd=5e6, beta_window=60, verbose=True):
        download_if_needed(verbose=verbose)
        close_all = pd.read_parquet(DATA / "close_1d_full.parquet")
        qvol_all = pd.read_parquet(DATA / "quote_volume_1d_full.parquet")
        high_all = pd.read_parquet(DATA / "high_1d_full.parquet")
        low_all = pd.read_parquet(DATA / "low_1d_full.parquet")
        tbb_all = pd.read_parquet(DATA / "taker_buy_base_volume_1d_full.parquet")
        ret_all = close_all.pct_change(fill_method=None)

        # Universe: top-N by TRAIN-window $-volume only (full-sample volume would be look-ahead).
        UNIV = list(seg(qvol_all, TRAIN).median().sort_values(ascending=False).head(top_n).index)
        btc = ret_all["BTCUSDT"]
        ew = ret_all[UNIV].mean(axis=1)

        # Point-in-time liquidity mask (trailing-60d median $-vol over threshold, lagged).
        liq60 = (qvol_all.rolling(60).median().shift(1) > liq_usd)

        # Short feasibility: a coin can only be shorted with a tradeable USDT perp.
        shortable = pd.DataFrame(False, index=close_all.index, columns=close_all.columns)
        shortable[[c for c in close_all.columns if c in PERP_SYMBOLS]] = True
        short_ok = liq60 & shortable

        # Idiosyncratic (BTC-residualised) returns: strip each coin's beta component (beta lagged).
        beta_all = ret_all.rolling(beta_window).cov(btc).div(btc.rolling(beta_window).var(), axis=0)
        resid_all = ret_all - beta_all.shift(1).mul(btc, axis=0)

        funding = pd.read_parquet(DATA / "funding_daily_full.parquet")
        perp = pd.read_parquet(DATA / "perp_close_1d_full.parquet")

        return cls(
            close_all=close_all, qvol_all=qvol_all, high_all=high_all, low_all=low_all, tbb_all=tbb_all,
            ret_all=ret_all, resid_all=resid_all, beta_all=beta_all,
            liq60=liq60, shortable=shortable, short_ok=short_ok,
            UNIV=UNIV, close=close_all[UNIV], qvol=qvol_all[UNIV], high=high_all[UNIV], low=low_all[UNIV],
            ret=ret_all[UNIV], btc=btc, ew=ew, funding=funding, perp=perp,
        )

    # --- optional intraday pulls used by specific analyses --------------------
    def hourly_basis(self, symbols, verbose=True):
        """Fetch hourly spot+perp for ``symbols`` and return the per-coin hourly
        basis frame ``(perp/spot - 1)``. Cached under ``crypto_data/processed/_h1basis``.
        Used to measure carry's intraday basis risk."""
        return _hourly_basis(symbols, verbose=verbose)

    def fetch_4h(self, symbols, verbose=True):
        """Fetch 4-hour spot closes for ``symbols`` -> DataFrame of 4h returns.
        Used to show that intraday-rebalanced books die on turnover."""
        return _fetch_4h(symbols, verbose=verbose)


# ---------------------------------------------------------------------------
# Intraday helpers (hourly basis, 4h bars) — cached, best-effort
# ---------------------------------------------------------------------------
_SPOT1H = "https://data.binance.vision/data/spot/monthly/klines"
_FUT1H = "https://data.binance.vision/data/futures/um/monthly/klines"
_MON1H = [(p.year, p.month) for p in pd.period_range("2020-01", "2026-05", freq="M")]


def _bar_series(base, sym, tag, freq, cache):
    fr = []
    for y, m in _MON1H:
        fp = cache / f"{tag}-{sym}-{y}-{m:02d}.zip"
        if fp.exists():
            b = fp.read_bytes()
            b = b if b else None
        else:
            try:
                b = urlopen(f"{base}/{sym}/{freq}/{sym}-{freq}-{y}-{m:02d}.zip", timeout=30).read()
            except Exception:
                b = None
            fp.write_bytes(b or b"")
        if not b:
            continue
        try:
            with zipfile.ZipFile(io.BytesIO(b)) as z, z.open(z.namelist()[0]) as f:
                d = pd.read_csv(f, header=None, usecols=[0, 4], names=["t", "c"])
        except Exception:
            continue
        d = d[pd.to_numeric(d["t"], errors="coerce").notna()]
        if len(d):
            t = pd.to_numeric(d["t"])
            idx = pd.to_datetime(t, unit=("us" if t.max() > 1e14 else "ms"), utc=True)
            fr.append(pd.Series(pd.to_numeric(d["c"]).values, index=idx))
    if not fr:
        return None
    s = pd.concat(fr).sort_index()
    return s[~s.index.duplicated(keep="last")]


def _hourly_basis(symbols, verbose=True):
    cache = DATA / "_h1basis"
    cache.mkdir(parents=True, exist_ok=True)
    if verbose:
        print("Fetching hourly spot+perp for the carry majors (cached after first run)...")
    sp, pp = {}, {}
    with cf.ThreadPoolExecutor(max_workers=16) as ex:
        fs = {ex.submit(_bar_series, _SPOT1H, s, "s", "1h", cache): s for s in symbols}
        fpp = {ex.submit(_bar_series, _FUT1H, s, "p", "1h", cache): s for s in symbols}
        for fu in cf.as_completed(fs):
            r = fu.result()
            if r is not None:
                sp[fs[fu]] = r
        for fu in cf.as_completed(fpp):
            r = fu.result()
            if r is not None:
                pp[fpp[fu]] = r
    common = [s for s in symbols if s in sp and s in pp]
    SP = pd.DataFrame({s: sp[s] for s in common})
    PP = pd.DataFrame({s: pp[s] for s in common})
    ix = SP.index.union(PP.index)
    SP, PP = SP.reindex(ix), PP.reindex(ix)
    return PP / SP - 1


def _fetch_4h(symbols, verbose=True):
    cache = DATA / "_h4"
    cache.mkdir(parents=True, exist_ok=True)
    if verbose:
        print("Fetching 4h spot bars (cached after first run)...")
    out = {}
    with cf.ThreadPoolExecutor(max_workers=16) as ex:
        futs = {ex.submit(_bar_series, _SPOT1H, s, "s4", "4h", cache): s for s in symbols}
        for fu in cf.as_completed(futs):
            r = fu.result()
            if r is not None:
                out[futs[fu]] = r
    px = pd.DataFrame(out).sort_index()
    return px.pct_change(fill_method=None)
