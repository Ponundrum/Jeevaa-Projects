"""``self_test()`` — the engine's green light. Cheap, assert-based checks that the
Monte Carlo machinery agrees with the closed-form mathematics, run first in every
notebook so no result is trusted before the engine is. Parallel to the sibling
project's ``qsa.engine.self_test``.
"""
from __future__ import annotations


from .config import get_rng
from .analytic import bs_price, bs_greeks, geometric_asian_price, put_call_parity_gap, digital_delta
from . import payoffs, processes
from .engine import mc_price, convergence, convergence_slope
from .greeks import mc_greeks, mc_digital_delta


def self_test(verbose=True):
    """Run the trust checks; raise AssertionError on any failure. Returns a summary
    string. Tolerances are in Monte Carlo standard errors, never eyeballed."""
    S, K, T, r, sigma, q = 100.0, 100.0, 1.0, 0.05, 0.2, 0.0
    rng = get_rng(2024)

    # 1) European MC == Black-Scholes, within 3 standard errors (call and put).
    paths1 = processes.simulate_gbm(S, r, q, sigma, T, 1, 200_000, rng, antithetic=True)
    for kind in ("call", "put"):
        res = mc_price(paths1, payoffs.european(kind, K), r, T)
        bs = bs_price(S, K, T, r, sigma, q, kind)
        assert abs(res.price - bs) < 3 * res.std_error, f"MC vs BS {kind}: {res.price} vs {bs}"

    # 2) Put-call parity holds on the engine's own MC prices (to MC error).
    c = mc_price(paths1, payoffs.european("call", K), r, T)
    p = mc_price(paths1, payoffs.european("put", K), r, T)
    assert abs(put_call_parity_gap(c.price, p.price, S, K, T, r, q)) < 3 * (c.std_error + p.std_error)

    # 3) Geometric-Asian MC == its closed form (exact path-dependent benchmark).
    pa = processes.simulate_gbm(S, r, q, sigma, T, 50, 200_000, rng, antithetic=True)
    ga = mc_price(pa, payoffs.asian_geometric("call", K), r, T)
    ga_cf = geometric_asian_price(S, K, T, r, sigma, q, 50, "call")
    assert abs(ga.price - ga_cf) < 3 * ga.std_error, f"geo-Asian: {ga.price} vs {ga_cf}"

    # 4) Convergence rate ~ O(N^-1/2). The log-log slope is a NOISY statistic, so this
    # check is deliberately generous: a fresh deterministic seed, many replications, a
    # wide path ladder (dropping the small-N point whose finite-sample RMSE biases the
    # slope), and a band that confirms "~ -1/2" rather than pinning two decimals.
    sim = lambda N, g: processes.simulate_gbm(S, r, q, sigma, T, 1, N, g)
    Ns, rmse = convergence(sim, payoffs.european("call", K), r, T,
                           [8000, 16000, 32000, 64000, 128000, 256000],
                           bs_price(S, K, T, r, sigma, q), get_rng(4), n_reps=80)
    slope = convergence_slope(Ns, rmse)
    assert -0.75 < slope < -0.30, f"convergence slope {slope}"

    # 5) MC Greeks agree with closed form — ALL THREE methods, delta/vega/gamma.
    g_bs = bs_greeks(S, K, T, r, sigma, q, "call")
    g_mc = mc_greeks(S, K, T, r, sigma, q, "call", n_paths=400_000, rng=rng)
    for method in ("pathwise", "likelihood_ratio", "finite_diff"):
        assert abs(g_mc[method]["delta"] - g_bs["delta"]) < 0.01, f"{method} delta"
        assert abs(g_mc[method]["vega"] - g_bs["vega"]) < 0.6, f"{method} vega"
    for method in ("likelihood_ratio", "finite_diff"):
        assert abs(g_mc[method]["gamma"] - g_bs["gamma"]) < 0.003, f"{method} gamma"

    # 5b) Likelihood-ratio delta of a DIGITAL matches its closed form, where pathwise
    #     is structurally zero — the case LR exists for.
    dig = mc_digital_delta(S, K, T, r, sigma, q, "call", n_paths=400_000, rng=rng)
    assert abs(dig["likelihood_ratio"] - digital_delta(S, K, T, r, sigma, q, "call")) < 0.002
    assert abs(dig["pathwise"]) < 1e-9

    msg = (f"Self-tests passed: MC==BS within 3 SE, put-call parity holds, geometric-Asian "
           f"matches closed form, convergence slope {slope:.2f} (~ -0.5), Greeks agree "
           f"(pathwise/LR/FD, incl. LR digital delta).")
    if verbose:
        print(msg)
    return msg
