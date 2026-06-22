# The Wall Street Quants Course Project: Statistical Arbitrage in Cryptocurrencies

> Markdown transcription of `ClassProject_7.docx` for easy reference in-repo.

## Project Goal

Statistical arbitrage is a class of strategies that tries to discover price-volume
patterns that predict returns. It is one of the most popular and successful
quantitative hedge-fund strategies.

Cryptocurrency markets are still relatively new and should be fertile grounds for
finding market inefficiencies using statistical arbitrage techniques. The two main
patterns exploited in statistical arbitrage are **momentum** and **reversal**.

**The goal of this project is to research profitable momentum and/or reversal
strategies in crypto.**

## Research Outline

### How to find momentum
- **Time Horizon** — Longer time horizons generally lead to momentum. Test different
  time horizons and see where momentum might exist.
- **New Information / Activity** — Times of heightened activity coupled with new
  information favor momentum. Apply indicators of activity / new info (e.g. Twitter
  activity, trading volume) to find stronger momentum.
- **Seasonality** — Similar seasons tend to show momentum. What are the relevant
  seasons in crypto and do they show momentum? Explore times when institutions vs.
  retail trade (e.g. weekdays vs. weekends, day vs. night).
- **Investment Themes** — Investment "themes" or "styles" tend to show momentum.
- **Technical Plays** — Sometimes predictable mechanical rebalancing leads to
  momentum. Given institutional crypto players and their trading schedules (usually
  during work-hours), is it possible to front-run them?

### How to find reversal
- **Time Horizon** — Shorter time horizons generally lead to reversal. Test different
  time horizons and see where reversal might exist.
- **Uninformed Trading** — Uninformed (liquidity-driven) trades reverse more. Apply
  indicators of activity / new info and isolate cases of lower activity / info to
  strengthen reversal. Draw on the "Fire Sale" crypto idea, which relied on uninformed
  trading during liquidations to find reversal.
- **Correlation** — `Security A - (Something Correlated to It)` is more mean-reverting.
  Find either correlated pairs or baskets of crypto assets.
- **Macro** — There is more reversal in times when there's more volatility and
  dislocation in the macro environment. Test using indicators of volatility /
  dislocation (implied volatility, realized volatility, return dispersion, pairwise
  correlation).

## Data
Cryptocurrency price-volume data is freely available (see the "PriceData" lecture).

## Backtesting
To start, use the **"unconstrained"** style of backtests introduced in the
backtesting section of the course. If needed, move onto other types of backtests later.

## Execution / Slippage
Cryptocurrencies can have commissions of ~7 bps. Total slippage is unknown and depends
on the trader's volume; assume another 13 bps. So total all-in execution costs will be
**20 bps for market orders**. **Limit orders will just have the 7 bps** of commissions.

## Weighting
You may find more than one compelling strategy (e.g. 1 momentum + 1 reversal). Combine
them appropriately (see the weighting videos).

## Performance Evaluation
Provide the key performance metrics: **returns, volatility, Sharpes, max drawdowns, and
alpha / beta.**
