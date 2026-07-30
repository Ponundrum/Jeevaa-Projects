"""The quote join must be causal — a trade sees the book strictly before it (no network)."""
import numpy as np
import pandas as pd

from mmlab import quotes


def _quotes():
    # quotes at integer seconds 0..9, widening spread so each is distinguishable
    t = np.arange(10, dtype=float)
    bid = 100.0 - 0.1 * t
    ask = 100.0 + 0.1 * t
    return pd.DataFrame({"time": t, "bid": bid, "ask": ask,
                         "bid_qty": np.ones(10), "ask_qty": np.ones(10)})


def test_quote_at_is_strictly_prior():
    q = _quotes()
    # A query EXACTLY on a quote timestamp must return the PREVIOUS quote, not that one.
    b, a = quotes.quote_at(q, np.array([5.0]))
    assert b[0] == q["bid"].iloc[4] and a[0] == q["ask"].iloc[4]
    # A query between quotes returns the last quote strictly before it.
    b, a = quotes.quote_at(q, np.array([5.5]))
    assert b[0] == q["bid"].iloc[5] and a[0] == q["ask"].iloc[5]
    # Before the first quote -> NaN (no prior quote exists).
    b, a = quotes.quote_at(q, np.array([-1.0]))
    assert np.isnan(b[0]) and np.isnan(a[0])


def test_quote_at_shock_is_causal():
    # Perturbing quotes strictly after t0 must not change any join result at or before t0.
    q = _quotes()
    q2 = q.copy()
    q2.loc[q2["time"] > 5.0, "bid"] -= 50.0
    query = np.array([3.0, 5.0])                 # both <= t0 = 5
    b1, a1 = quotes.quote_at(q, query)
    b2, a2 = quotes.quote_at(q2, query)
    assert np.allclose(b1, b2) and np.allclose(a1, a2)


def test_touch_half_spread_and_mid():
    q = _quotes()
    assert np.allclose(quotes.quote_mid(q), 100.0)
    assert np.allclose(quotes.touch_half_spread(q), 0.1 * np.arange(10))


def test_microprice_leans_to_thin_side():
    # More size on the bid -> price should sit closer to the ask (bid is likely to hold).
    q = pd.DataFrame({"time": [0.0], "bid": [99.0], "ask": [101.0],
                      "bid_qty": [9.0], "ask_qty": [1.0]})
    assert quotes.microprice(q)[0] > 100.0
