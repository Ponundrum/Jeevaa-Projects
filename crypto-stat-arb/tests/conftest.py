"""Tiny synthetic Dataset so the suite runs in <1s with no network."""
import numpy as np
import pandas as pd
import pytest

from qsa.data import Dataset


@pytest.fixture
def ds():
    """A 6-coin, 150-day synthetic dataset with the attributes the engine reads.
    The last coin is deliberately non-shortable, to test the short screen."""
    n_days, cols = 150, [f"C{i}USDT" for i in range(6)]
    rng = np.random.default_rng(0)
    idx = pd.date_range("2021-01-01", periods=n_days, freq="D", tz="UTC")
    ret = pd.DataFrame(rng.normal(0, 0.03, (n_days, len(cols))), index=idx, columns=cols)
    close = (1 + ret).cumprod() * 100
    qvol = pd.DataFrame(rng.uniform(1e7, 1e8, (n_days, len(cols))), index=idx, columns=cols)
    liq60 = pd.DataFrame(True, index=idx, columns=cols)
    short_ok = liq60.copy()
    short_ok[cols[-1]] = False                          # one non-shortable name
    btc = ret[cols[0]]
    beta = ret.rolling(20).cov(btc).div(btc.rolling(20).var(), axis=0)
    resid = ret - beta.shift(1).mul(btc, axis=0)
    return Dataset(close_all=close, qvol_all=qvol, ret_all=ret, resid_all=resid,
                   liq60=liq60, short_ok=short_ok, UNIV=cols[:4], btc=btc, ew=ret.mean(1),
                   NONSHORT=cols[-1])
