"""Universe sizes are quoted in the README, so pin them here (no network)."""
from qsa.data import SPOT_SYMBOLS, PERP_SYMBOLS


def test_spot_pool_size_and_uniqueness():
    assert len(SPOT_SYMBOLS) == len(set(SPOT_SYMBOLS))
    assert len(SPOT_SYMBOLS) == 379          # quoted in README


def test_shortable_count():
    assert len(set(SPOT_SYMBOLS) & PERP_SYMBOLS) == 277   # quoted in README
