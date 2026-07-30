# Quantitative Finance Projects

Three self-contained projects across the arc of quant — **the research, the pricing, and the trading**. One
finds and validates market-neutral *alpha* in real crypto data; one builds and *proves* a derivatives-pricing
engine from first principles against closed-form mathematics; one shows, from real trade data, why naive
market making loses to adverse selection and what quoting rule survives. Each folder runs end-to-end from a
clean checkout, with its own tests, CI, and honest-limitations write-up.

## 1 · Crypto Statistical Arbitrage &nbsp;·&nbsp; [`crypto-stat-arb/`](crypto-stat-arb/)

A market-neutral crypto book built from three uncorrelated edges — **momentum**, **low-volatility**, and
**funding carry** — with the full signal search behind it. Designed on 2020–2023, tested out-of-sample on
2024–2026, net of a realistic turnover cost, with shorts restricted to borrowable names.

**Out-of-sample Sharpe +1.6 at a −15% maximum drawdown — versus roughly −77% for buy-and-hold BTC.**

| Strategy | Train | **Val (OOS)** | Val Sortino | Max DD | DSR |
|---|---:|---:|---:|---:|---:|
| **Combined book** | +1.05 | **+1.57** | +2.21 | **−15%** | 0.42 |
| Momentum (proximity-to-high) | +0.68 | +0.65 | +0.87 | −49% | 0.05 |
| Low-volatility (idiosyncratic) | +0.36 | +1.92 | +2.75 | −60% | 0.63 |
| Carry (funding, daily marks) | +4.16 | +2.98 | +7.60 | −6% | 0.99 |
| BTC (benchmark) | +1.00 | +0.72 | +1.09 | −77% | 0.06 |

A small Python package (`qsa`) does the plumbing; three notebooks tell the story — the combined book and its
robustness, the research trail of every rejected signal, and a walk-forward that re-fits weights on past-only
data. &nbsp;→ **[Read the project](crypto-stat-arb/)**

![Combined book vs buy-and-hold](crypto-stat-arb/docs/combined_book.png)

## 2 · Monte Carlo Derivatives Pricing & Rough Volatility &nbsp;·&nbsp; [`options-mc-engine/`](options-mc-engine/)

A from-scratch Monte Carlo option-pricing engine — **proven correct against closed-form Black–Scholes to within
Monte Carlo error** — then extended to exotics, the Heston model, and **rough Bergomi** (rough volatility
simulated by covariance decomposition). The complementary skill set to project 1: stochastic calculus,
numerical methods, variance reduction, and model calibration, where correctness is provable against
mathematics rather than a dataset.

**On a live SPY option-chain snapshot, rough volatility (calibrated Hurst H ≈ 0.07) matches the steep
short-dated skew that classical Heston structurally flattens.** European MC matches Black–Scholes within 3
standard errors; the control-variate estimator cuts Asian-option variance ~1300×; simulated rough paths
verifiably recover their Hurst exponent.

A Python package (`qmc`) implements every model from scratch (no QuantLib); two notebooks tell the story — the
engine proven correct, and the volatility surface with rough-vol calibration. &nbsp;→ **[Read the project](options-mc-engine/)**

![Short-dated smile: rough volatility matches the steeper skew](options-mc-engine/docs/rough_vol_smile.png)

## 3 · Market-Making Laboratory &nbsp;·&nbsp; [`mm-lab/`](mm-lab/)

The trading side: how a liquidity *provider* actually loses or survives. A minimal Poisson-fill simulator,
matched exactly to the **Avellaneda–Stoikov** quoting model, plus a data pipeline that measures **adverse
selection** from Binance trades and feeds it back into the sim. Deliberately whiteboard-simple — no RL, no
order-book reconstruction, no queue model — with every headline checked against a closed form.

**On BTCUSDT trades, a passive fill is marked out ~0.65 bps against the maker within ~5 seconds, then flat —
real, fast adverse selection.** Fed back into the simulator it costs naive and AS a *comparable* amount per
fill; AS wins on a ~6× smaller PnL variance, because inventory control removes the mark-to-market swings — a
variance claim, not a cost-avoidance one. (An earlier lookahead bug in the mid proxy had produced a tidier
"naive loses" headline; a causal fix overturned it, and the corrected result is reported instead.)

A Python package (`mmlab`) holds the simulator, the two strategies, calibration and markouts; two notebooks
tell the story — the model proven correct, then adverse selection measured and closed back into the loop.
&nbsp;→ **[Read the project](mm-lab/)**

![Inventory paths: naive random-walks, Avellaneda-Stoikov mean-reverts to flat](mm-lab/docs/inventory_paths.png)
