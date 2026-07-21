"""Path simulators: correct moments, martingale property, and — for rough Bergomi
— that the simulated paths actually have the intended roughness."""
import numpy as np
import pytest

from qmc.config import get_rng
from qmc.analytic import bs_price
from qmc import processes, payoffs
from qmc.engine import mc_price

S, r, q, SIG, T = 100.0, 0.05, 0.0, 0.2, 1.0


def test_gbm_terminal_moments():
    rng = get_rng(1)
    paths = processes.simulate_gbm(S, r, q, SIG, T, 1, 500_000, rng)
    ST = paths[:, -1]
    assert ST.mean() == pytest.approx(S * np.exp((r - q) * T), rel=2e-3)          # forward
    var_theory = S ** 2 * np.exp(2 * (r - q) * T) * (np.exp(SIG ** 2 * T) - 1)
    assert ST.var() == pytest.approx(var_theory, rel=2e-2)


def test_gbm_is_exact_for_one_step():
    # GBM log is exactly normal, so even n_steps=1 prices the European exactly (to SE)
    rng = get_rng(2)
    paths = processes.simulate_gbm(S, r, q, SIG, T, 1, 400_000, rng, antithetic=True)
    res = mc_price(paths, payoffs.european("call", 100), r, T)
    assert abs(res.price - bs_price(S, 100, T, r, SIG, q, "call")) < 3 * res.std_error


def test_antithetic_pairs_are_negated():
    rng = get_rng(3)
    Z = processes._draw_normals(6, 4, rng, antithetic=True)
    assert np.allclose(Z[:3], -Z[3:])


def test_heston_martingale_and_bs_limit():
    rng = get_rng(4)
    # xi -> 0 is deterministic vol -> Black-Scholes at sqrt(v0)
    p = processes.simulate_heston(S, 0.04, 1.5, 0.04, 1e-4, -0.7, r, q, T, 100, 100_000, rng)
    res = mc_price(p, payoffs.european("call", 100), r, T)
    assert abs(res.price - bs_price(S, 100, T, r, np.sqrt(0.04), q, "call")) < 3 * res.std_error
    assert np.exp(-r * T) * p[:, -1].mean() == pytest.approx(S, rel=3e-3)          # martingale


def test_rough_bergomi_martingale_and_hurst_recovery():
    rng = get_rng(5)
    S_paths, v = processes.simulate_rough_bergomi(S, 0.04, 1.5, -0.7, 0.1, r, q, T, 400, 4000, rng, return_v=True)
    assert np.exp(-r * T) * S_paths[:, -1].mean() == pytest.approx(S, rel=1e-2)    # martingale
    # estimate Hurst from the log-variance structure function; should recover H=0.1
    lv = np.log(v)
    lags = np.arange(1, 21)
    m = [np.mean((lv[:, lag:] - lv[:, :-lag]) ** 2) for lag in lags]
    H_hat = np.polyfit(np.log(lags), np.log(m), 1)[0] / 2
    assert abs(H_hat - 0.1) < 0.05
