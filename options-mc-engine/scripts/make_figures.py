"""Regenerate the committed figures in docs/ (so GitHub viewers see results without
running anything). Run with `make figures`. Uses the cached SPY snapshot."""
import warnings; warnings.filterwarnings("ignore")
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np, matplotlib.pyplot as plt
from qmc import get_rng, apply_style, CLR
from qmc.analytic import bs_price, bs_implied_vol
from qmc import payoffs, processes, iv
from qmc.engine import mc_price, convergence, convergence_slope
from qmc.data import option_chain_snapshot
from qmc.calibration import calibrate_heston, calibrate_rough_bergomi

apply_style()
DOCS = pathlib.Path(__file__).resolve().parents[1] / "docs"; DOCS.mkdir(exist_ok=True)
S0, r, q, sig = 100.0, 0.05, 0.0, 0.20
rng = get_rng()

# 1) MC convergence to Black-Scholes
bs = bs_price(S0, 100, 1.0, r, sig, q, "call")
sim = lambda N, g: processes.simulate_gbm(S0, r, q, sig, 1.0, 1, N, g)
Ns, rmse = convergence(sim, payoffs.european("call", 100), r, 1.0,
                       [1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000], bs, rng, n_reps=60)
fig, ax = plt.subplots(figsize=(7.5, 4.6))
ax.loglog(Ns, rmse, "o-", color=CLR["MC"], label=f"measured (slope {convergence_slope(Ns, rmse):+.3f})")
ax.loglog(Ns, rmse[0]*(Ns/Ns[0])**-0.5, "--", color=CLR["theory"], label=r"theoretical $O(N^{-1/2})$")
ax.set_xlabel("paths N"); ax.set_ylabel("RMSE vs Black-Scholes")
ax.set_title("Monte Carlo converges to the closed form at the theoretical rate"); ax.legend()
plt.tight_layout(); fig.savefig(DOCS/"convergence.png", dpi=120, bbox_inches="tight"); plt.close(fig)

# 2) SPY implied-vol smiles (cached snapshot)
chain, meta = option_chain_snapshot("SPY", verbose=False)
S, r_m = meta["spot"], meta["r"]
surf = iv.implied_vol_surface(chain, S, r_m)
Ts = sorted(surf["T"].unique())
fig, ax = plt.subplots(figsize=(8, 4.6))
for T, c in zip(Ts, plt.cm.viridis(np.linspace(0, 0.9, len(Ts)))):
    g = surf[surf["T"] == T].sort_values("k")
    ax.plot(g["k"], g["iv"], "-", color=c, lw=1.3, label=f"{T*365:.0f}d")
ax.set_xlabel("log-moneyness k = ln(K/F)"); ax.set_ylabel("implied volatility")
ax.set_title(f"SPY implied-vol surface (snapshot {meta['asof']}) — the equity skew"); ax.legend(ncol=2, fontsize=8)
plt.tight_layout(); fig.savefig(DOCS/"vol_surface.png", dpi=120, bbox_inches="tight"); plt.close(fig)

# 3) Hero: short-dated smile, market vs Heston vs rough Bergomi
cal = surf[surf["k"].abs() < 0.12]
cal = cal[cal["T"].isin([Ts[1], Ts[3], Ts[5], Ts[7]])].groupby("T", group_keys=False).apply(
    lambda g: g.iloc[np.linspace(0, len(g)-1, min(7, len(g))).astype(int)])
hp, _, _ = calibrate_heston(cal, S, r_m, n_paths=30_000, max_nfev=50, verbose=False)
short = surf[surf["k"].abs() < 0.12]
short = short[short["T"].isin(Ts[1:4])].groupby("T", group_keys=False).apply(
    lambda g: g.iloc[np.linspace(0, len(g)-1, min(7, len(g))).astype(int)])
rb, _, _ = calibrate_rough_bergomi(short, S, r_m, n_paths=15_000, steps_per_year=150, max_nfev=30, verbose=False)
Tsh, q0 = Ts[1], chain["q"].iloc[0]
g = surf[(surf["T"] == Tsh) & (surf["k"] > -0.13) & (surf["k"] < 0.05)].sort_values("k")
def smile(paths):
    return np.array([bs_implied_vol(mc_price(paths, payoffs.european("put" if kk < 0 else "call", K), r_m, Tsh).price,
                                    S, K, Tsh, r_m, q0, "put" if kk < 0 else "call") for K, kk in zip(g["K"], g["k"])])
sm_h = smile(processes.simulate_heston(S, hp["v0"], hp["kappa"], hp["theta"], hp["xi"], hp["rho"], r_m, q0, Tsh, 150, 400_000, get_rng(1)))
sm_r = smile(processes.simulate_rough_bergomi(S, rb["xi0"], rb["eta"], rb["rho"], rb["H"], r_m, q0, Tsh, 200, 150_000, get_rng(2)))
fig, ax = plt.subplots(figsize=(8, 4.8))
ax.plot(g["k"], g["iv"], "o", color=CLR["Market"], ms=6, label="market")
ax.plot(g["k"], sm_h, "-", color=CLR["Heston"], lw=1.8, label="Heston")
ax.plot(g["k"], sm_r, "-", color=CLR["rBergomi"], lw=1.8, label=f"rough Bergomi (H={rb['H']:.2f})")
ax.set_xlabel("log-moneyness k"); ax.set_ylabel("implied vol")
ax.set_title(f"Short-dated SPY smile ({Tsh*365:.0f}d): rough volatility matches the steeper skew"); ax.legend()
plt.tight_layout(); fig.savefig(DOCS/"rough_vol_smile.png", dpi=120, bbox_inches="tight"); plt.close(fig)

print(f"Wrote convergence.png, vol_surface.png, rough_vol_smile.png (Heston rho={hp['rho']:+.2f}, rBergomi H={rb['H']:.3f})")
