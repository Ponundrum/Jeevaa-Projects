"""Backtest engine, portfolio-construction rules, and the metrics stack.

Design notes that make the numbers trustworthy:

* **No look-ahead.** ``backtest`` lags weights one bar, so today's P&L only ever
  uses yesterday's positions. ``self_test`` asserts this.
* **Costs are real.** Turnover (sum of absolute weight changes) is charged at
  ``TCOST`` (flat) or a per-coin square-root market-impact cost (``backtest_lc``).
* **Shorts must be borrowable.** ``dn_weights(screen=True)`` restricts the short
  leg to names with a tradeable, liquid perpetual and re-centres so the book
  stays dollar-neutral on the names it actually holds.

The weight builders and the liquidity-cost model need the market context
(liquidity mask, dollar volume, short-feasibility), so they take a ``Dataset``
``ds``. The metrics are pure functions of a return series and a window.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .config import TCOST, ANN, RF, seg


# ---------------------------------------------------------------------------
# Portfolio construction
# ---------------------------------------------------------------------------
def eq_weights(signal):
    """Directional long-only: weight proportional to a non-negative conviction
    signal; gross leverage 1."""
    w = signal.clip(lower=0)
    return w.divide(w.abs().sum(1).replace(0, np.nan), axis=0).fillna(0.0)


def cs_weights(signal):
    """Cross-sectional dollar-neutral: demean then scale to gross leverage 1.

    Unscreened — used only to *show* that raw cross-sectional signals fail; the
    books that survive use ``dn_weights`` (liquidity + short screened)."""
    s = signal.subtract(signal.mean(1), axis=0)
    return s.divide(s.abs().sum(1).replace(0, np.nan), axis=0).fillna(0.0)


def dn_weights(signal, ds, topN=None, screen=True):
    """Liquidity-screened dollar-neutral weights — the workhorse construction.

    1. Keep only names that clear the point-in-time liquidity mask (``ds.liq60``),
       optionally the ``topN`` most liquid.
    2. Demean (long > 0, short < 0) and scale to gross 1.
    3. If ``screen``: restrict the SHORT leg to shortable names (a tradeable,
       liquid perp), iterating drop-infeasible-short -> re-centre to a fixed
       point so residual short weight on non-shortable names is ~0.
    """
    s = signal.where(ds.liq60)
    if topN is not None:
        rk = ds.qvol_all.rolling(60).median().shift(1).rank(axis=1, ascending=False)
        s = s.where(rk <= topN)
    s = s.subtract(s.mean(1), axis=0)                       # + long, - short
    if screen:
        for _ in range(4):
            s = s.where((s >= 0) | ds.short_ok)            # drop infeasible shorts
            s = s.subtract(s.mean(1), axis=0)              # restore dollar-neutrality
        s = s.where((s >= 0) | ds.short_ok, 0.0)           # final hard mask
        s = s.where(s.notna(), 0.0)
    return s.divide(s.abs().sum(1).replace(0, np.nan), axis=0).fillna(0.0)


def backtest(weights, rets, tcost=TCOST, rebal=1):
    """Hold weights ``rebal`` days; lag 1 bar; charge ``tcost`` on turnover.

    Returns ``(net_daily_pnl, daily_turnover)``.

    Convention (P0.4): between rebalances the target weights are held CONSTANT and
    P&L is computed on those constant weights; turnover is charged only at
    rebalance points (the jump from last period's weights to this period's). The
    small daily trades that would be needed to *hold* weights constant as prices
    drift are not charged here — a standard simplification for low-turnover weekly
    books. ``maintenance_turnover`` estimates that omitted cost, and §4 shows it
    barely moves the headline (Appendix A documents the choice). With ``rebal=1``
    (the engine self-test) every bar is a rebalance, so the question doesn't arise.
    """
    if rebal > 1:
        mask = pd.Series(np.arange(len(weights)) % rebal == 0, index=weights.index)
        weights = weights.where(mask, np.nan).ffill()
    pnl = (weights.shift(1) * rets).sum(1)
    turn = (weights.shift(1) - weights.shift(2)).abs().sum(1).fillna(0.0)
    return pnl - tcost * turn, turn


def backtest_carry(spot, perp, funding, symbols, capture=0.85, drag=0.02, rebal=7, tcost=TCOST):
    """Funding-carry sleeve, under the same tested roof as the other books (P2.6).

    Long spot / short perpetual, equal-weighting the pairs available each day,
    rebalanced every ``rebal`` days, with turnover charged on BOTH legs plus a
    financing ``drag`` (annual). ``capture`` is the fraction of funding actually
    realised. Returns ``(net_daily_pnl, daily_turnover)``. With ``capture=1,
    drag=0`` it is the idealised full-capture version. Funding is contemporaneous
    (the §4 integrity check confirms lagging it one day barely moves the result)."""
    sr = spot[symbols].pct_change(fill_method=None)
    pr = perp[symbols].pct_change(fill_method=None)
    avail = (spot[symbols].notna() & perp[symbols].notna()).astype(float)
    w = avail.div(avail.sum(1).replace(0, np.nan), axis=0)
    mask = pd.Series(np.arange(len(w)) % rebal == 0, index=w.index)
    w = w.where(mask, np.nan).ffill()
    turn = (w.shift(1) - w.shift(2)).abs().sum(1).fillna(0.0)
    pnl = (w.shift(1) * (sr - pr + capture * funding[symbols])).sum(1) - 2 * tcost * turn - drag / 365
    return pnl, turn


def maintenance_turnover(weights, rets, rebal):
    """Estimate the daily turnover needed to HOLD target weights constant as prices
    drift between rebalances (the cost ``backtest`` omits — see its docstring).

    On a non-rebalance day each position's weight drifts by ~``w_i*(ret_i - r_book)``;
    trading it back to target costs that much turnover. Zero on rebalance days (that
    trade is already counted in ``backtest``'s turnover). Use it to price the
    worst-case extra cost and show the low-turnover sleeves survive it."""
    mask = pd.Series(np.arange(len(weights)) % rebal == 0, index=weights.index)
    w = weights.where(mask, np.nan).ffill()
    r_book = (w.shift(1) * rets).sum(1)                         # book return each day
    drift = w.shift(1).mul(rets).sub(w.shift(1).mul(r_book, axis=0))   # per-name weight drift
    mt = drift.abs().sum(1)
    mt[mask.values] = 0.0                                        # folded into rebalance turnover
    return mt.fillna(0.0)


def liq_cost_frame(ds, base=TCOST, cap=8.0):
    """Per-coin turnover cost = ``base * sqrt(reference $-vol / coin $-vol)``,
    capped at ``cap``x base. Thin names pay several times what the majors pay."""
    medv = ds.qvol_all.rolling(60).median().shift(1)               # lagged trailing $-vol
    ref = medv[ds.UNIV].median(axis=1)                             # reference = median major
    cf = base * np.sqrt(np.divide(ref.values[:, None], medv.values,
            out=np.full(medv.shape, cap), where=medv.values > 0)).clip(1, cap)
    return pd.DataFrame(cf, index=medv.index, columns=medv.columns)


def backtest_lc(weights, rets, costframe, rebal=1):
    """Like ``backtest`` but charges a PER-COIN liquidity-aware turnover cost."""
    if rebal > 1:
        mask = pd.Series(np.arange(len(weights)) % rebal == 0, index=weights.index)
        weights = weights.where(mask, np.nan).ffill()
    pnl = (weights.shift(1) * rets).sum(1)
    turn = (weights.shift(1) - weights.shift(2)).abs()
    cost = (turn * costframe.reindex_like(turn).fillna(TCOST * 8)).sum(1).fillna(0.0)
    return pnl - cost, turn.sum(1)


# ---------------------------------------------------------------------------
# Metrics (pure functions of a return series + evaluation window)
# ---------------------------------------------------------------------------
def maxdd(x):
    c = (1 + x.fillna(0)).cumprod()
    return (c / c.cummax() - 1).min()


# Std convention (P0.3): sample standard deviation (ddof=1) is used for EVERY Sharpe
# in the project — the headline, its bootstrap CI, and its deflated version — so the
# CI and DSR correspond exactly to the point estimate they annotate. pandas .std()
# defaults to ddof=1; the NumPy paths below pass ddof=1 explicitly to match.
def sharpe(x, rf=RF):
    """Annualised Sharpe. ``rf`` is the annual risk-free rate, defaulting to
    ``config.RF`` (0) — see the note there: the books are dollar-neutral and
    self-funding, so excess return ~= raw return."""
    x = x.dropna()
    if len(x) < 5 or x.std() == 0:
        return np.nan
    return (x.mean() - rf / 365) / x.std() * ANN


def sortino(x, mar=0.0):
    """Downside-risk analogue of Sharpe: annualised mean over the RMS of
    below-target returns (target 0)."""
    x = x.dropna()
    if len(x) < 5:
        return np.nan
    dsd = np.sqrt((np.minimum(x - mar, 0.0) ** 2).mean())
    return np.nan if dsd == 0 else (x.mean() - mar) / dsd * ANN


def alpha_beta(y, x, window, hac_lags=21):
    """OLS alpha/beta with Newey-West (HAC) errors -> autocorrelation-robust t."""
    d = pd.concat([seg(y, window), seg(x, window)], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    d = d[d.abs().sum(axis=1) > 0]
    m = sm.OLS(d.iloc[:, 0], sm.add_constant(d.iloc[:, 1])).fit(
        cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    return dict(alpha_ann=m.params.iloc[0] * 365, beta=m.params.iloc[1], alpha_t=m.tvalues.iloc[0])


def hac_ols(y, X, window, hac_lags=21):
    """Multi-factor OLS of return series ``y`` on a DataFrame of factor returns
    ``X`` (P1.4), with Newey-West (HAC) errors. Returns the annualised alpha and
    its t-stat plus each factor's beta and t. Lets you check the book's alpha
    survives controlling for more than just BTC beta."""
    d = pd.concat([seg(y, window).rename("y"), seg(X, window)], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    d = d[d.abs().sum(axis=1) > 0]
    m = sm.OLS(d["y"], sm.add_constant(d.drop(columns="y"))).fit(
        cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    out = {"alpha_ann": m.params["const"] * 365, "alpha_t": m.tvalues["const"]}
    for f in X.columns:
        out[f"{f} beta"] = m.params[f]
        out[f"{f} t"] = m.tvalues[f]
    return out


def hac_tstat(net, window, lags=21):
    """Newey-West t-stat that the mean return differs from zero."""
    s = seg(net, window).dropna()
    return sm.OLS(s.values, np.ones(len(s))).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags}).tvalues[0]


def metrics(net, window, bench=None):
    s = seg(net, window).dropna()
    out = {"Ann.Return": s.mean() * 365, "Ann.Vol": s.std() * ANN,
           "Sharpe": sharpe(s), "MaxDD": maxdd(s)}
    if bench is not None:
        out.update({k.title().replace("_", " "): v for k, v in alpha_beta(net, bench, window).items()})
    return out


def bootstrap_sharpe_ci(net, window, n=2000, seed=0, block=10):
    """Moving-block bootstrap (block ~10d) CI for the Sharpe — resamples blocks,
    not days, so it respects autocorrelation."""
    s = seg(net, window).dropna().values
    T = len(s)
    if T < block + 5:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(T / block))
    starts = rng.integers(0, T - block + 1, size=(n, nb))
    bs = []
    for row in starts:
        v = np.concatenate([s[i:i + block] for i in row])[:T]
        bs.append(v.mean() / v.std(ddof=1) * ANN)                 # ddof=1 to match sharpe()
    return np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def sharpe_diff_test(a, b, window, n=2000, seed=0, block=10):
    """Is strategy ``a``'s Sharpe *significantly* higher than benchmark ``b``'s?
    (P1.3) Paired moving-block bootstrap of ``sharpe(a) - sharpe(b)`` over
    ``window``, resampling the SAME block indices from both series so the pairing
    (and their correlation) is preserved. Returns the observed annualised
    difference, its 95% CI, and a two-sided bootstrap p-value that the true
    difference is zero."""
    d = pd.concat([seg(a, window), seg(b, window)], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    A, B = d.iloc[:, 0].values, d.iloc[:, 1].values
    T = len(A)
    if T < block + 5:
        return {"diff_ann": np.nan, "ci": (np.nan, np.nan), "p": np.nan}
    _sr = lambda v: v.mean() / v.std(ddof=1) * ANN
    obs = _sr(A) - _sr(B)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(T / block))
    diffs = np.empty(n)
    for k in range(n):
        starts = rng.integers(0, T - block + 1, size=nb)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:T]
        diffs[k] = _sr(A[idx]) - _sr(B[idx])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())     # two-sided
    return {"diff_ann": obs, "ci": (lo, hi), "p": min(p, 1.0)}


def deflated_sharpe(net, window, n_trials):
    """Probability the true Sharpe > 0 after deflating for ``n_trials`` and
    non-normality (Bailey / Lopez de Prado)."""
    from scipy.stats import norm, skew, kurtosis
    s = seg(net, window).dropna().values
    if len(s) < 30:
        return np.nan
    sr = s.mean() / s.std(ddof=1)                                # ddof=1 to match sharpe()
    T = len(s)
    g3, g4 = skew(s), kurtosis(s, fisher=False)
    emax = (1 - np.euler_gamma) * norm.ppf(1 - 1 / n_trials) + np.euler_gamma * norm.ppf(1 - 1 / (n_trials * np.e))
    sr0 = emax / np.sqrt(T)
    denom = np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2)
    return float(norm.cdf((sr - sr0) * np.sqrt(T - 1) / denom))


def self_test():
    """Assert the engine has no look-ahead and actually charges cost. Cheap; run
    it in a notebook to earn trust in every downstream number."""
    _r = pd.DataFrame({"A": [0.0, 0.10, -0.05, 0.02]})
    _net, _ = backtest(pd.DataFrame({"A": [1., 1, 1, 1]}), _r, tcost=0.0)
    assert abs(_net.iloc[1:].sum() - _r["A"].iloc[1:].sum()) < 1e-12, "lag/identity broken"
    _n0, _ = backtest(pd.DataFrame({"A": [0, 1, 1, 0]}), _r, tcost=0.0)
    _n1, _ = backtest(pd.DataFrame({"A": [0, 1, 1, 0]}), _r, tcost=0.01)
    assert (_n0.sum() - _n1.sum()) > 0, "cost not charged on turnover"
    assert abs(backtest(pd.DataFrame({"A": [1., 1, 0, 0]}), _r)[0].iloc[1]
               - backtest(pd.DataFrame({"A": [1., 1, 1, 1]}), _r)[0].iloc[1]) < 1e-12, "lookahead detected"
    return "Self-tests passed: weights lag one bar (no look-ahead), turnover is charged, buy-hold identity holds."
