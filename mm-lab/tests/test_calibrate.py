"""Parameter estimation, validated against ground truth (check 6)."""
import numpy as np

from mmlab import calibrate, simulate
from mmlab.config import get_rng


def test_realized_sigma_recovers_input():
    sigma, dt = 0.3, 0.5
    mid = simulate.simulate_mid(100.0, sigma, T=8000.0, dt=dt, n_paths=1, rng=get_rng(1))[0]
    assert abs(calibrate.realized_sigma(mid, dt) - sigma) / sigma < 0.05


def test_intensity_recovery():
    A_true, k_true = 2.0, 1.5
    depths, dur = calibrate.synthetic_depth_tape(A_true, k_true, 8000.0, get_rng(2))
    grid = calibrate.default_delta_grid(depths)
    A_hat, k_hat, x, lam = calibrate.fit_intensity(depths, dur, grid)
    assert abs(A_hat - A_true) / A_true < 0.10
    assert abs(k_hat - k_true) / k_true < 0.10
    assert len(x) == len(lam) >= 2


def test_trade_depths_split_by_side():
    # is_buyer_maker=True -> market sell -> hits resting bids at (mid - price).
    price = np.array([99.0, 101.0, 98.0])
    mid = np.array([100.0, 100.0, 100.0])
    ibm = np.array([True, False, True])
    bid_d, ask_d = calibrate.trade_depths(price, mid, ibm)
    assert np.allclose(sorted(bid_d), [1.0, 2.0])   # mid - price for the two sells
    assert np.allclose(ask_d, [1.0])                # price - mid for the one buy


def test_fit_is_close_to_a_line_in_log_space():
    depths, dur = calibrate.synthetic_depth_tape(3.0, 1.0, 10000.0, get_rng(3))
    grid = calibrate.default_delta_grid(depths)
    A_hat, k_hat, x, lam = calibrate.fit_intensity(depths, dur, grid)
    resid = np.log(lam) - (np.log(A_hat) - k_hat * x)
    assert np.std(resid) < 0.1                      # exponential model fits by construction
