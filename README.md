# Projects

Quantitative research and data-science projects. Each folder is self-contained — its own code, data
pipeline, and write-up — and runs end-to-end from a clean checkout.

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

---

*More projects coming.*
