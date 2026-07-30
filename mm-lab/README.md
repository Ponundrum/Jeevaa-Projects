# A Market-Making Laboratory

[![CI](https://github.com/Ponundrum/Jeevaa-Projects/actions/workflows/ci.yml/badge.svg)](https://github.com/Ponundrum/Jeevaa-Projects/actions/workflows/ci.yml)

A market maker who quotes symmetrically around the mid *thinks* they earn the spread. This
project builds a minimal simulator to state that intuition precisely, then measures — from
real Binance trades — the **adverse selection** it ignores: the market moves against the
passive side of essentially every fill. That measured adverse selection is then fed back
into the simulator to see what it does to naive vs inventory-aware (Avellaneda–Stoikov)
quoting.

Everything is built from fewer than five moving parts that can be drawn on a whiteboard —
no reinforcement learning, no order-book reconstruction, no queue model. Where a closed
form exists (the AS quoting rule, the `γ → 0` limit, the PnL accounting identity), the code
is checked against it.

**On BTCUSDT trades, a passive fill is marked out ~0.65 bps against the maker within ~5
seconds, then flat to 300s — adverse selection is real, fast, and bounded in time.** Fed
back into the simulator it costs the naive and the AS strategy a *comparable* amount per
fill (~−10% vs −7% of PnL); AS's advantage is that inventory control removes the
mark-to-market **variance** (PnL std ~6× smaller), not that it dodges the adverse move — a
variance claim, honestly, not a cost-avoidance one.

> **Reported honestly.** An earlier version measured the mid with a *centred* smoother that
> peeked one step into the future, producing a tidy "naive loses ~2.5× the spread" headline.
> With a strictly causal mid (`tests/test_data.py` pins it), that result did not survive:
> naive stays profitable, and AS's edge is variance reduction. The corrected story is the
> one below — see notebook 02.

![Inventory paths: naive random-walks, AS mean-reverts to flat](docs/inventory_paths.png)

## The one idea, in three layers

Each layer is verifiable before the next is built (the self-test enforces it):

1. **The simulator** *(no data)* — an arithmetic-Brownian mid and a Poisson fill engine
   (`λ(δ) = A e^{-κδ}`), matched *exactly* to the Avellaneda–Stoikov assumptions so the
   closed-form quoting rule is ground truth against it.
2. **The strategies** — naive symmetric quoting (the strawman) vs Avellaneda–Stoikov, which
   quotes around an inventory-skewed reservation price. The head-to-head deliverable is a
   **PnL decomposition** into *spread PnL* and *inventory PnL* — the single most
   illuminating output, and the reason AS matters.
3. **Real data** — calibrate `σ, A, κ` from Binance `aggTrades` (against a strictly causal
   mid proxy), measure the **markout curve** (the adverse selection), then **close the loop**
   by feeding it back into the sim and reporting a sensitivity over how the drift is injected.

![Passive markout: adverse selection realized within ~5s, then flat to 300s](docs/markout_curve.png)

## What's validated (the definition of done)

`self_test()` runs first in every notebook and as a unit test — seven checks, nothing
loosened to pass:

| Check | What it pins |
|---|---|
| 1. `γ → 0` limit | edge term `→ 2/κ`, reservation price `→` mid (analytic) |
| 2. Zero inventory | `q = 0` ⟹ reservation price `==` mid, exactly |
| 3. Accounting identity | `total PnL == spread PnL + inventory PnL` per path, to 1e-8 |
| 4. No fills | `A = 0` ⟹ zero fills, inventory, PnL |
| 5. Fill rate | measured fill rate matches `A e^{-κδ}` within MC error |
| 6. **Estimator recovery** | the `A`/`κ` calibration recovers known values from a synthetic tape |
| 7. AS beats naive on **inventory** | matched-parameter terminal-inventory std strictly lower for AS |

Check 7 pins the *inventory* claim, not a PnL claim — AS does **not** reliably beat naive on
mean PnL, and a test that asserted it would be a lie that eventually fails.

## What's in this repo

Read it in this order:

1. **[`notebooks/00_intuition.md`](notebooks/00_intuition.md)** — the whole project in plain
   language: the ten questions an interviewer asks (why symmetric quoting loses, what the
   reservation price is, why you can't backtest a market maker, …), answered without
   equations beyond the two AS formulas.
2. **[`notebooks/01_model_and_simulator.ipynb`](notebooks/01_model_and_simulator.ipynb)** —
   theory → simulator → strategy comparison → PnL decomposition → the inventory figure.
3. **[`notebooks/02_adverse_selection.ipynb`](notebooks/02_adverse_selection.ipynb)** —
   calibration on real BTCUSDT trades, the `κ` fit, the markout curve, and closing the loop.
4. **[`mmlab/`](mmlab/)** — the ~800-line package the notebooks import: `simulate.py`
   (mid + Poisson fills), `strategies.py` (naive, Avellaneda–Stoikov), `metrics.py` (PnL
   decomposition), `calibrate.py` (`σ, A, κ`), `markout.py`, `data.py` (cached Binance
   loader), `plotting.py`, `selftest.py`.

## Running it

```bash
pip install -e ".[dev]"      # numpy / scipy / pandas / matplotlib
python -c "from mmlab import self_test; self_test()"   # trust the lab first (~2s)
pytest                       # 31 unit tests, no network (~3s)
jupyter lab                  # notebooks/01 ... then notebooks/02
```

Every result is reproducible from the single seed in `mmlab/config.py` (`SEED = 20240101`).
Notebook 01 needs no data. Notebook 02 pulls a few days of BTCUSDT `aggTrades` from Binance's
public archive (`data.binance.vision`, ~15 MB/day) and caches them under `_cache/`; the
loader retries with backoff and degrades gracefully if the network is down.

## Honest limitations

- **Fills are a Poisson model, not a matching engine.** No queue position — in reality,
  being at the back of the queue means being filled precisely when you least want to be,
  which would make naive look *worse*, not better.
- **No latency, no cancellations, no order-size distribution** — every fill is one unit.
- **No market impact:** the simulated maker's own quotes do not move the price.
- **Arithmetic Brownian mid** has no fat tails, no volatility clustering, no jumps — the
  three things that actually kill market makers.
- **Mid proxy — and what it cannot measure.** The futures `bookTicker` (true bid/ask) exists
  but is ~10× heavier (~188 MB/day), so notebook 02 uses a strictly-causal **trade-price mid
  proxy**. At 1-second resolution it cannot resolve BTC's true spread (one tick ≈ 0.01 USDT ≈
  0.002 bps); the "penetration depths" it measures are dominated by ~1s of volatility, which
  makes the fitted `κ` (~0.19) implausibly low. So the absolute fill economics are *not*
  reliable, and this repo does **not** quote a spread-capture-vs-markout ratio — that would
  need real quotes. The robust results are the markout magnitude/shape and the simulator's
  inventory-control behaviour. The proxy also lags ~1s, which is why the 1s markout point is
  positive (an artefact, not maker profit).
- **`σ` is biased a little low.** Median-smoothing the mid removes high-frequency variation,
  so the realized `σ` (~28% annualised) understates true vol; since `σ` enters the AS spread
  quadratically, the inventory-risk term is understated too.
- **The markout is measured on *all* tape trades**, not on this strategy's own fills, so it
  estimates the adverse selection facing a *typical* passive quote; and the feedback injects
  it as a first-order per-fill drift whose *timing* is a modelling choice (the notebook reports
  a sensitivity over `adverse_steps`), not a microstructurally exact coupling.
- **AS quotes can cross the mid at high inventory.** The skew can push a half-spread negative,
  where `λ(δ) = A e^{−κδ}` is extrapolated outside `δ ≥ 0` and stops meaning anything (it does
  not bite at the inventories reached here, but `AvellanedaStoikov(min_half_spread=…)` floors
  it; a production maker would floor the quote and impose a hard inventory limit).
- **The stationary variant is the pragmatic one.** The `(T−t)` clock is frozen at a fixed
  risk horizon (what production systems effectively do). The principled alternative is the
  Guéant–Lehalle–Fernandez-Tapia asymptotic solution, which gives stationary quotes and a
  hard inventory bound; it is *not* implemented here because its closed form was not sourced
  and verified, and writing it from memory would violate the project's "check against the
  math" rule.
- **One symbol, one window.** BTCUSDT, a few days in January 2024. No claim of generality —
  the point is to understand `κ`, not to survey coins.

MIT licensed. Sibling projects: **[`../crypto-stat-arb`](../crypto-stat-arb)** (empirical
crypto alpha research) and **[`../options-mc-engine`](../options-mc-engine)** (Monte Carlo
derivatives pricing).
