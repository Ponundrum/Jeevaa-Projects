"""Monte Carlo Greeks: all three methods match the analytic Greeks, and the
likelihood-ratio estimator handles the discontinuous digital that pathwise cannot."""
import pytest

from qmc.config import get_rng
from qmc.analytic import bs_greeks, digital_delta
from qmc.greeks import mc_greeks, mc_digital_delta

S, K, T, r, SIG, Q = 100.0, 100.0, 1.0, 0.05, 0.2, 0.0


def test_all_three_methods_match_bs_greeks():
    g_bs = bs_greeks(S, K, T, r, SIG, Q, "call")
    g_mc = mc_greeks(S, K, T, r, SIG, Q, "call", n_paths=400_000, rng=get_rng(9))
    for method in ("pathwise", "likelihood_ratio", "finite_diff"):
        assert g_mc[method]["delta"] == pytest.approx(g_bs["delta"], abs=0.01)
        assert g_mc[method]["vega"] == pytest.approx(g_bs["vega"], abs=0.6)
    for method in ("likelihood_ratio", "finite_diff"):        # pathwise gives no vanilla gamma
        assert g_mc[method]["gamma"] == pytest.approx(g_bs["gamma"], abs=0.003)


def test_digital_delta_lr_matches_pathwise_is_zero():
    exact = digital_delta(S, K, T, r, SIG, Q, "call")
    mc = mc_digital_delta(S, K, T, r, SIG, Q, "call", n_paths=600_000, rng=get_rng(3))
    assert mc["likelihood_ratio"] == pytest.approx(exact, abs=0.002)   # LR is right ...
    assert mc["pathwise"] == 0.0                                       # ... pathwise is structurally zero
