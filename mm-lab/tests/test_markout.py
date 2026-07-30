"""Markout sign and direction on a controlled synthetic tape."""
import numpy as np

from mmlab import markout


def test_adverse_selection_shows_as_negative_markout():
    # Two passive fills: the maker BUYS (is_buyer_maker=True) at t=0 and t=10, and
    # the mid then drifts DOWN afterwards -> the passive buyer's markout is negative.
    t_mid = np.arange(0, 100, 1.0)
    p_mid = 100.0 - 0.01 * t_mid                        # steadily falling mid
    t_trade = np.array([0.0, 10.0])
    p_fill = np.array([100.0, 99.9])
    ibm = np.array([True, True])                        # maker bought
    means, counts = markout.markout_curve(t_trade, p_fill, ibm, t_mid, p_mid, [5, 30])
    assert means[5] < 0 and means[30] < means[5]        # negative and worsening
    assert counts[30] == 2


def test_favourable_move_is_positive_markout():
    t_mid = np.arange(0, 100, 1.0)
    p_mid = 100.0 + 0.01 * t_mid                        # rising mid
    means, _ = markout.markout_curve(np.array([0.0]), np.array([100.0]),
                                     np.array([True]), t_mid, p_mid, [10])
    assert means[10] > 0                                # bought, mid rose -> good


def test_seller_side_sign_flips():
    # Same falling mid, but the maker SOLD (is_buyer_maker=False) -> good for a seller.
    t_mid = np.arange(0, 100, 1.0)
    p_mid = 100.0 - 0.01 * t_mid
    means, _ = markout.markout_curve(np.array([0.0]), np.array([100.0]),
                                     np.array([False]), t_mid, p_mid, [10])
    assert means[10] > 0


def test_horizon_past_tape_end_is_dropped():
    t_mid = np.arange(0, 20, 1.0)
    p_mid = np.full_like(t_mid, 100.0)
    means, counts = markout.markout_curve(np.array([15.0]), np.array([100.0]),
                                          np.array([True]), t_mid, p_mid, [10])
    assert counts[10] == 0 and np.isnan(means[10])      # t+h beyond the tape
