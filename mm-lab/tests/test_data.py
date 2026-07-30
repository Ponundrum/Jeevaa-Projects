"""The mid proxy must be causal — the property the whole Layer-3 measurement rests on."""
import numpy as np
import pandas as pd

from mmlab.data import mid_grid


def _tape(seed=0, n=4000, span=400.0):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0.0, span, n))
    price = 100.0 + np.cumsum(rng.standard_normal(n) * 0.05)
    ibm = rng.random(n) < 0.5
    return pd.DataFrame({"time": t, "price": price, "is_buyer_maker": ibm})


def test_mid_proxy_is_causal():
    # Perturbing the tape strictly AFTER t0 must not change any mid at or before t0.
    a = _tape()
    b = a.copy()
    t0 = 200.0
    b.loc[b["time"] > t0, "price"] += 500.0          # a large, obvious shock to the future
    t_a, m_a = mid_grid(a)
    t_b, m_b = mid_grid(b)
    past = t_a <= t0
    assert past.sum() > 10                            # the assertion is not vacuous
    assert np.allclose(m_a[past], m_b[past])          # fails with a centred / current-bin mid


def test_mid_grid_shapes_and_finite():
    t_grid, mid = mid_grid(_tape())
    assert len(t_grid) == len(mid)
    assert np.all(np.isfinite(mid))
    assert np.all(np.diff(t_grid) > 0)
