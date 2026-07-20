"""Regenerate the committed figures in docs/ (P3.2), so the READMEs can show
results without opening a notebook. Run with `make figures`."""
import warnings; warnings.filterwarnings("ignore")
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd, matplotlib.pyplot as plt, matplotlib.ticker as mtick
from qsa import Dataset, apply_style, config as C
from qsa.engine import dn_weights, backtest, backtest_carry, sharpe, maxdd
from qsa import signals as S

apply_style()
DOCS = pathlib.Path(__file__).resolve().parents[1] / "docs"; DOCS.mkdir(exist_ok=True)
ds = Dataset.load()
seg, TRAIN, VAL, FULL, ANN = C.seg, C.TRAIN, C.VAL, C.FULL, C.ANN
btc, ew, ret_all = ds.btc, ds.ew, ds.ret_all

mom, _ = backtest(dn_weights(S.near_high(ds, 90), ds), ret_all, rebal=7)
bab, _ = backtest(dn_weights(S.idio_vol(ds, 60), ds, topN=50), ret_all, rebal=7)
MAJ = [s for s in ["BTCUSDT","ETHUSDT","BNBUSDT","XRPUSDT","DOGEUSDT","ADAUSDT","LINKUSDT","LTCUSDT","TRXUSDT",
    "ETCUSDT","SOLUSDT","AVAXUSDT","DOTUSDT","MATICUSDT","ATOMUSDT","UNIUSDT","FILUSDT","AAVEUSDT","VETUSDT","CHZUSDT"]
    if s in ds.perp.columns and s in ds.close_all.columns]
carry, _ = backtest_carry(ds.close_all, ds.perp, ds.funding, MAJ, capture=0.85, drag=0.02, rebal=7)
sl = pd.DataFrame({"m": mom, "b": bab, "c": carry}).loc[TRAIN[0]:FULL[1]].dropna()
vt = seg(sl, TRAIN).std() * ANN; w = (1/vt)/(1/vt).sum(); w["c"] = min(w["c"], 0.5)
w[["m", "b"]] *= (1-w["c"]) / w[["m", "b"]].sum()
comb = (sl * w).sum(1)

def gr(s): return (1+seg(s, FULL).fillna(0)).cumprod()
def dd(s): c = gr(s); return c/c.cummax()-1

# 1) combined book vs buy-and-hold (the hero figure)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
for nm, s, c, lw in [("Combined book", comb, "#1b1b1b", 2.0), ("BTC (buy & hold)", btc, "#ff7f0e", 1.3),
                     ("Equal-weight", ew, "#8c8c8c", 1.3)]:
    gr(s).plot(ax=ax[0], label=nm, color=c, lw=lw); dd(s).plot(ax=ax[1], label=nm, color=c, lw=lw)
ax[0].set_yscale("log"); ax[0].set_title("Growth of \\$1 (log) — 2020-2026, net of 20 bps"); ax[0].set_ylabel("growth of \\$1"); ax[0].legend()
ax[1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0)); ax[1].set_title("Drawdown — combined book vs buy-and-hold"); ax[1].set_ylabel("drawdown"); ax[1].legend()
for a in ax: a.axvline(pd.Timestamp(VAL[0]), ls="--", c="grey", lw=.8); a.set_xlabel("")
plt.tight_layout(); fig.savefig(DOCS/"combined_book.png", dpi=120, bbox_inches="tight"); plt.close(fig)

# 2) in-sample vs out-of-sample Sharpe bar (the whole project in one view)
fig, ax = plt.subplots(figsize=(9, 4.2))
order = [("Momentum", mom), ("Low-vol", bab), ("Carry", carry), ("Combined", comb), ("BTC", btc), ("Equal-weight", ew)]
y = np.arange(len(order)); h = 0.38
ax.barh(y+h/2, [sharpe(seg(s, TRAIN)) for _, s in order], h, label="In-sample (2020-23)", color="#bdbdbd")
ax.barh(y-h/2, [sharpe(seg(s, VAL)) for _, s in order], h, label="Out-of-sample (2024-26)", color="#1b1b1b")
ax.set_yticks(y); ax.set_yticklabels([n for n, _ in order]); ax.invert_yaxis(); ax.axvline(0, c="k", lw=.8)
ax.set_xlabel("Sharpe ratio"); ax.set_title("Sharpe by strategy — in-sample vs out-of-sample"); ax.legend(loc="lower right")
plt.tight_layout(); fig.savefig(DOCS/"scoreboard.png", dpi=120, bbox_inches="tight"); plt.close(fig)

print(f"Wrote {DOCS}/combined_book.png and {DOCS}/scoreboard.png")
print(f"(combined OOS Sharpe {sharpe(seg(comb, VAL)):+.2f}, MaxDD {maxdd(seg(comb, FULL)):.0%})")
