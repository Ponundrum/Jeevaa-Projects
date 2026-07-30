"""Regenerate the committed figures in ``docs/``. Offline figures (the simulator
ones) always run; the data figures need the aggTrades cache and are skipped with a
note if it is absent. Run from the project root: ``python scripts/make_figures.py``
(or ``make figures``).

The analysis parameters here are the single source of truth for the numbers quoted
in the README and echoed in the notebooks.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from mmlab import calibrate, data, markout, metrics, simulate, strategies
from mmlab.config import get_rng
from mmlab.plotting import CLR, apply_style

DOCS = __import__("pathlib").Path(__file__).resolve().parent.parent / "docs"
DATES = ["2024-01-15", "2024-01-16", "2024-01-17"]
HORIZONS = [1, 5, 10, 30, 60, 300]

# --- simulator params, matched between the two strategies ------------------
SIM = dict(S0=42730.0, sigma=2.14, A=2.6, kappa=0.576, T=600.0, dt=1.0, n_paths=3000)
GAMMA = 0.0008
NAIVE_HALF = 1.1                       # ~ where real fills land (mean depth), in USDT
ADVERSE = 0.666e-4 * SIM["S0"]         # measured 60s markout, in price units


def _run(strat, seed, adverse=0.0):
    return simulate.run(strat, rng=get_rng(seed), adverse=adverse,
                        **{k: v for k, v in SIM.items()})


def fig_inventory_paths():
    """HERO figure: inventory paths, naive vs AS. Naive random-walks; AS is pinned
    near zero by the reservation-price skew."""
    apply_style()
    rn = _run(strategies.Naive(NAIVE_HALF), 7)
    ra = _run(strategies.AvellanedaStoikov(GAMMA, SIM["sigma"], SIM["kappa"], SIM["T"]), 7)
    t = np.arange(rn.inventory.shape[1]) * SIM["dt"] / 60.0     # minutes
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
    print(f"  inventory_paths.png  (naive term-inv std {rn.terminal_inventory.std():.1f} "
          f"vs AS {ra.terminal_inventory.std():.2f})")


def fig_pnl_decomposition():
    """Spread PnL vs inventory PnL for both strategies, with and without the
    data-measured adverse drift."""
    apply_style()
    labels, spread, inv = [], [], []
    for adv, tag in [(0.0, "no adv"), (ADVERSE, "adv")]:
        for strat, name in [(strategies.Naive(NAIVE_HALF), "Naive"),
                            (strategies.AvellanedaStoikov(GAMMA, SIM["sigma"], SIM["kappa"], SIM["T"]), "AS")]:
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
    ax.set_title("Adverse selection turns inventory PnL sharply negative — naive is hit hardest")
    ax.legend()
    fig.tight_layout()
    fig.savefig(DOCS / "pnl_decomposition.png", bbox_inches="tight")
    plt.close(fig)
    print("  pnl_decomposition.png")


def _load_real():
    trades = data.load_aggtrades("BTCUSDT", DATES, verbose=False)
    t_mid, mid = data.mid_grid(trades, step=1.0, smooth=3)
    return trades, t_mid, mid


def fig_markout_and_kappa():
    """The two data figures: the kappa fit (with points) and the markout curve."""
    try:
        trades, t_mid, mid = _load_real()
    except Exception as e:
        print(f"  [skip data figures] {e}")
        return
    apply_style()
    mid_at = markout._mid_at(t_mid, mid, trades.time.to_numpy())
    ok = np.isfinite(mid_at)
    bid_d, ask_d = calibrate.trade_depths(trades.price.to_numpy()[ok], mid_at[ok],
                                          trades.is_buyer_maker.to_numpy()[ok])
    dur = float(trades.time.max() - trades.time.min())

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for d, name, c in [(bid_d, "bid", CLR["bid"]), (ask_d, "ask", CLR["ask"])]:
        grid = calibrate.default_delta_grid(d)
        A, k, x, lam = calibrate.fit_intensity(d, dur, grid)
        ax.scatter(x, np.log(lam), s=18, color=c, label=f"{name}: kappa={k:.2f}, A={A:.2f}")
        ax.plot(x, np.log(A) - k * x, color=c, lw=1.8)
    ax.set_xlabel("quote distance delta from mid (USDT)")
    ax.set_ylabel("log fill intensity  log lambda(delta)")
    ax.set_title("Fill intensity is exponential in quote distance: log-linear fit of lambda = A e^{-kappa delta}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(DOCS / "kappa_fit.png", bbox_inches="tight")
    plt.close(fig)
    print("  kappa_fit.png")

    means, _ = markout.markout_curve(trades.time.to_numpy(), trades.price.to_numpy(),
                                     trades.is_buyer_maker.to_numpy(), t_mid, mid, HORIZONS)
    hs_bps = markout.half_spread_captured_bps(bid_d, ask_d, float(np.median(mid)))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(HORIZONS, [means[h] for h in HORIZONS], "o-", color=CLR["markout"], lw=2, label="passive markout")
    ax.axhline(0, color="#333", lw=0.9)
    ax.axhline(-hs_bps, color=CLR["fit"], ls="--", lw=1.6, label=f"half-spread captured (+{hs_bps:.2f} bps, shown as -)")
    ax.fill_between(HORIZONS, [means[h] for h in HORIZONS], 0, color=CLR["markout"], alpha=0.12)
    ax.set_xscale("log")
    ax.set_xticks(HORIZONS)
    ax.set_xticklabels([str(h) for h in HORIZONS])
    ax.set_xlabel("horizon after fill (seconds)")
    ax.set_ylabel("markout (bps)")
    ax.set_title("The market moves against the passive quote — adverse selection dwarfs the captured spread")
    ax.legend()
    fig.tight_layout()
    fig.savefig(DOCS / "markout_curve.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  markout_curve.png  (60s markout {means[60]:.2f} bps vs +{hs_bps:.2f} bps spread)")


def main():
    DOCS.mkdir(exist_ok=True)
    print("regenerating docs/ figures:")
    fig_inventory_paths()
    fig_pnl_decomposition()
    fig_markout_and_kappa()
    print("done.")


if __name__ == "__main__":
    main()
