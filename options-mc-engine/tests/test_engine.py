"""Engine: MC prices agree with closed form, variance reduction works, convergence
rate is O(N^-1/2). Small, seeded, deterministic."""


from qmc.config import get_rng
from qmc.analytic import bs_price, geometric_asian_price, put_call_parity_gap, barrier_price, lookback_floating_price
from qmc import payoffs, processes
from qmc.engine import mc_price, variance_reduction_factor, convergence, convergence_slope, bgk_barrier_shift

S, K, T, r, SIG, Q = 100.0, 100.0, 1.0, 0.05, 0.2, 0.0


def test_european_mc_matches_bs_within_3se():
    rng = get_rng(1)
    paths = processes.simulate_gbm(S, r, Q, SIG, T, 1, 200_000, rng, antithetic=True)
    for kind in ("call", "put"):
        res = mc_price(paths, payoffs.european(kind, K), r, T)
        assert abs(res.price - bs_price(S, K, T, r, SIG, Q, kind)) < 3 * res.std_error


def test_put_call_parity_on_mc_prices():
    rng = get_rng(2)
    paths = processes.simulate_gbm(S, r, Q, SIG, T, 1, 200_000, rng, antithetic=True)
    c = mc_price(paths, payoffs.european("call", K), r, T)
    p = mc_price(paths, payoffs.european("put", K), r, T)
    assert abs(put_call_parity_gap(c.price, p.price, S, K, T, r, Q)) < 3 * (c.std_error + p.std_error)


def test_geometric_asian_mc_matches_closed_form():
    rng = get_rng(3)
    pa = processes.simulate_gbm(S, r, Q, SIG, T, 50, 200_000, rng, antithetic=True)
    res = mc_price(pa, payoffs.asian_geometric("call", K), r, T)
    assert abs(res.price - geometric_asian_price(S, K, T, r, SIG, Q, 50, "call")) < 3 * res.std_error


def test_control_variate_reduces_variance():
    rng = get_rng(4)
    pa = processes.simulate_gbm(S, r, Q, SIG, T, 50, 100_000, rng)
    control = (payoffs.asian_geometric("call", K), geometric_asian_price(S, K, T, r, SIG, Q, 50, "call"))
    vrf = variance_reduction_factor(pa, payoffs.asian_arithmetic("call", K), r, T, control)
    assert vrf > 20                                                    # geometric control is very effective


def test_barrier_mc_matches_bgk_corrected_closed_form():
    # discrete MC with the Broadie-Glasserman-Kou barrier shift matches the
    # continuous Reiner-Rubinstein closed form; raw discrete does not
    rng = get_rng(6)
    B, m = 90.0, 200
    paths = processes.simulate_gbm(S, r, Q, SIG, T, m, 200_000, rng, antithetic=True)
    cf = barrier_price(S, K, B, T, r, SIG, Q, "call", "down-and-out")
    Badj = bgk_barrier_shift(B, SIG, T / m, up=False)
    cor = mc_price(paths, payoffs.barrier("call", K, Badj, "down-and-out"), r, T)
    raw = mc_price(paths, payoffs.barrier("call", K, B, "down-and-out"), r, T)
    assert abs(cor.price - cf) < 4 * cor.std_error                     # corrected agrees
    assert abs(raw.price - cf) > 4 * raw.std_error                     # raw is biased


def test_lookback_mc_matches_closed_form():
    rng = get_rng(7)
    paths = processes.simulate_gbm(S, r, Q, SIG, T, 1000, 40_000, rng)
    cf = lookback_floating_price(S, T, r, SIG, Q, "call")
    mc = mc_price(paths, payoffs.lookback("call"), r, T)
    assert abs(mc.price - cf) / cf < 0.05                              # within 5% at a fine grid
    assert mc.price < cf                                               # discrete underestimates the extremum


def test_convergence_slope_near_minus_half():
    # The log-log slope is noisy; use a wide ladder + many replications + a generous
    # band so this asserts "~ -1/2", not two decimals (see selftest check 4).
    rng = get_rng(5)
    sim = lambda N, g: processes.simulate_gbm(S, r, Q, SIG, T, 1, N, g)
    Ns, rmse = convergence(sim, payoffs.european("call", K), r, T,
                           [8000, 16000, 32000, 64000, 128000, 256000],
                           bs_price(S, K, T, r, SIG, Q), rng, n_reps=60)
    assert -0.75 < convergence_slope(Ns, rmse) < -0.30
