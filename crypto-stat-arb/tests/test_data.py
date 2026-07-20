"""Data layer: timestamp parsing and signal shape/alignment (no network)."""
import numpy as np
import pandas as pd

from qsa.data import _to_utc
from qsa import signals as S


def test_to_utc_ms_and_us():
    # Binance uses both millisecond and (since ~2025) microsecond epochs; both must parse to the same date.
    ms = pd.Series([1577836800000, 1580515200000])          # 2020-01-01, 2020-02-01 in ms
    us = pd.Series([1577836800000000, 1580515200000000])    # same instants in microseconds
    dms = pd.DatetimeIndex(np.asarray(_to_utc(ms)))
    dus = pd.DatetimeIndex(np.asarray(_to_utc(us)))
    assert dms[0].year == 2020 and dms[1].month == 2
    assert (dms == dus).all()


def test_to_utc_rejects_bad_unit():
    # A microsecond value parsed as ms would land in year ~51969; the guard must not return that.
    idx = pd.DatetimeIndex(np.asarray(_to_utc(pd.Series([1700000000000000]))))
    assert 2017 <= idx[0].year <= 2027


def test_signals_shape_and_causality(ds):
    for fn in (lambda d: S.near_high(d, 30), lambda d: S.idio_vol(d, 20), lambda d: S.reversal(d, 3)):
        sig = fn(ds)
        assert sig.shape == ds.close_all.shape                # aligned to the pool
        assert list(sig.columns) == list(ds.close_all.columns)
        # rolling signals must have NaNs at the very start, i.e. no forward-fill from the future
        assert sig.iloc[0].isna().all()
