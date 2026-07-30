"""Regenerate the committed figures in ``docs/``. The offline simulator figures always
run; the data figures need the futures aggTrades + bookTicker caches and are skipped with
a note if absent. Run from the project root: ``python scripts/make_figures.py``.

The parameters here — calibrated from real BTCUSDT **futures** quotes in notebook 02 — are
the single source of truth for the numbers quoted in the README and echoed in the notebooks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mmlab import calibrate, data, markout, metrics, quotes, simulate, strategies
from mmlab.config import get_rng
from mmlab.plotting import CLR, apply_style

DOCS = __import__("pathlib").Path(__file__).resolve().parent.parent / "docs"
DATE = "2024-01-15"
HORIZONS = [1, 5, 10, 30, 60, 300]

# --- simulator params, calibrated from the true futures mid (notebook 02) ------
SIM = dict(S0=42646.0, sigma=3.184, A=3.47, kappa=0.484, T=600.0, dt=1.0, n_paths=3000)
GAMMA = 7e-4                            # re-chosen so the two AS spread terms are comparable
NAIVE_HALF = 0.5 * float(strategies.optimal_spread(GAMMA, SIM["sigma"], SIM["kappa"], SIM["T"]))
ADVERSE = 0.298e-4 * SIM["S0"]          # measured 60s markout vs the true mid, in price units


def _as():
    return strategies.AvellanedaStoikov(GAMMA, SIM["sigma"], SIM["kappa"], SIM["T"])


def _run(strat, seed, adverse=0.0):
    return simulate.run(strat, rng=get_rng(seed), adverse=adverse, **SIM)


def fig_inventory_paths():
    """HERO figure: inventory paths, naive vs AS."""
    apply_style()
    rn = _run(strategies.Naive(NAIVE_HALF), 7)
    ra = _run(_as(), 7)
    t = np.arange(rn.inventory.shape[1]) * SIM["dt"] / 60.0
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, res, name in [(axL, rn, "Naive"), (axR, ra, "AS")]:
        for i in range(24):
            ax.plot(t, res.inventory[i], color=CLR[name], alpha=0.25, lw=0.8)
        std = metrics.inventory_std_over_time(res)
        ax.plot(t, std, color=CLR[name], lw=2.2, label="±1 cross-path std")
        ax.plot(t, -std, color=CLR[name], lw=2.2)
        ax.axhline(0, color="#888", lw=0.8)
        ax.set_title(f"{name}: inventory paths", color=CLR[name])
        ax.set_xlabel("minutes")
        ax.legend(loc="upper left")
    axL.set_ylabel("inventory (units)")
    fig.suptitle("Naive inventory random-walks; Avellaneda-Stoikov mean-reverts to flat",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(DOCS / "inventory_paths.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  inventory_paths.png  (naive std {rn.terminal_inventory.std():.1f} vs AS {ra.terminal_inventory.std():.2f})")


def fig_pnl_decomposition():
    apply_style()
    labels, spread, inv = [], [], []
    for adv, tag in [(0.0, "no adv"), (ADVERSE, "adv")]:
        for strat, name in [(strategies.Naive(NAIVE_HALF), "Naive"), (_as(), "AS")]:
            _, s, iv = metrics.decompose_pnl(_run(strat, 7, adverse=adv))
            labels.append(f"{name}\n({tag})")
            spread.append(s.mean())
            inv.append(iv.mean())
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(x, spread, 0.6, label="spread PnL", color="#2ca02c")
    ax.bar(x, inv, 0.6, bottom=0, label="inventory PnL", color="#d62728", alpha=0.85)
    ax.plot(x, np.array(spread) + np.array(inv), "ko", ms=7, label="total PnL")
    ax.axhline(0, color="#333", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("mean PnL (USDT)")
    ax.set_title("Adverse selection dents inventory PnL for both; AS carries far less inventory risk")
    ax.legend()
    fig.tight_layout()
    fig.savefig(DOCS / "pnl_decomposition.png", bbox_inches="tight")
    plt.close(fig)
    print("  pnl_decomposition.png")


def _load_futures():
    tr = quotes.load_futures_aggtrades("BTCUSDT", [DATE], verbose=False)
    q = quotes.load_bookticker("BTCUSDT", DATE, verbose=False)
    return tr, q


def fig_data_figures():
    """The three data figures: markout proxy-vs-true, kappa fit, touch-spread — all on
    the same futures tape (proxy mid vs real bookTicker mid)."""
    try:
        tr, q = _load_futures()
    except Exception as e:
        print(f"  [skip data figures] {e}")
        return
    apply_style()
    ttime = tr["time"].to_numpy(); price = tr["price"].to_numpy(); ibm = tr["is_buyer_maker"].to_numpy()
    tq = q["time"].to_numpy(); tmid = quotes.quote_mid(q)
    S0 = float(np.median(tmid))
    dur = float(ttime.max() - ttime.min())

    # proxy mid (trade-price, 1s causal) and true mid markouts
    tpm, pm = data.mid_grid(tr, step=1.0, smooth=3)
    m_proxy, _ = markout.markout_curve(ttime, price, ibm, tpm, pm, HORIZONS)
    m_true, _ = markout.markout_curve(ttime, price, ibm, tq, tmid, HORIZONS)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(HORIZONS, [m_proxy[h] for h in HORIZONS], "o--", color=CLR["markout"], lw=1.8,
            label="trade-price proxy (1s)")
    ax.plot(HORIZONS, [m_true[h] for h in HORIZONS], "o-", color=CLR["fit"], lw=2.2,
            label="true bookTicker mid")
    ax.axhline(0, color="#333", lw=0.9)
    ax.annotate("proxy 1s point:\nlag artefact (+)", (1, m_proxy[1]),
                textcoords="offset points", xytext=(12, -6), fontsize=8, color="#555")
    ax.set_xscale("log"); ax.set_xticks(HORIZONS); ax.set_xticklabels([str(h) for h in HORIZONS])
    ax.set_xlabel("horizon after fill (seconds)"); ax.set_ylabel("passive markout (bps)")
    ax.set_title("Real quotes remove the proxy's 1s lag artefact; adverse selection ~ -0.3 bps")
    ax.legend()
    fig.tight_layout()
    fig.savefig(DOCS / "markout_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  markout_comparison.png  (proxy 1s {m_proxy[1]:+.2f} -> true 1s {m_true[1]:+.2f} bps)")

    # kappa fit against the true mid
    bd, ak = quotes.quote_at(q, ttime); qm = 0.5 * (bd + ak); ok = np.isfinite(qm)
    bidd, askd = calibrate.trade_depths(price[ok], qm[ok], ibm[ok])
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for d, name, c in [(bidd, "bid", CLR["bid"]), (askd, "ask", CLR["ask"])]:
        A, k, x, lam = calibrate.fit_intensity(d, dur, calibrate.default_delta_grid(d))
        ax.scatter(x, np.log(lam), s=18, color=c, label=f"{name}: A={A:.2f}, kappa={k:.2f}")
        ax.plot(x, np.log(A) - k * x, color=c, lw=1.8)
    ax.set_xlabel("quote distance delta from true mid (USDT)"); ax.set_ylabel("log lambda(delta)")
    ax.set_title("Fill intensity vs distance from the REAL mid: kappa ~ 0.48/USDT (proxy gave ~0.16)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(DOCS / "kappa_fit.png", bbox_inches="tight")
    plt.close(fig)
    print("  kappa_fit.png")

    # touch half-spread distribution
    ths = quotes.touch_half_spread(q)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.hist(np.clip(ths, 0, 0.5), bins=60, color=CLR["AS"], alpha=0.85)
    ax.axvline(ths.mean(), color="#d62728", lw=1.8, label=f"mean {ths.mean():.3f} USDT = {1e4*ths.mean()/S0:.4f} bps")
    ax.set_xlabel("touch half-spread (ask-bid)/2  (USDT)"); ax.set_ylabel("quote updates")
    ax.set_title("Observable touch half-spread: what a maker at the touch can capture per fill")
    ax.legend()
    fig.tight_layout()
    fig.savefig(DOCS / "touch_spread.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  touch_spread.png  (mean {1e4*ths.mean()/S0:.4f} bps)")


def main():
    DOCS.mkdir(exist_ok=True)
    print("regenerating docs/ figures:")
    fig_inventory_paths()
    fig_pnl_decomposition()
    fig_data_figures()
    print("done.")


if __name__ == "__main__":
    main()
