"""
Verification suite. Three kinds of checks, each testing something different:

1. Known-value checks: prices computed by hand/reference against a textbook
   example (Hull, "Options, Futures and Other Derivatives"), so there's a
   ground truth independent of this codebase.
2. Put-call parity: internal consistency, must hold to float precision.
3. Finite-difference cross-check: every analytic Greek is checked against a
   numerical (bump-and-reprice) derivative. This is the check that actually
   catches sign errors and algebra mistakes in the closed-form Greeks --
   the most common way a from-scratch BS implementation is subtly wrong.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import replace
from black_scholes.engine import (
    OptionSpec, price, delta, gamma, vega, theta, rho,
    put_call_parity_residual,
)
from black_scholes.implied_vol import implied_vol, ImpliedVolError

TOL = 1e-6
FD_TOL = 1e-3  # finite-difference vs analytic tolerance (FD has its own error)


def test_known_value_hull_example():
    # Hull, 11th ed., Example 15.6: S=42, K=40, r=0.10, sigma=0.20, T=0.5, q=0
    # Reference call price ~= 4.76
    spec = OptionSpec(S=42, K=40, T=0.5, r=0.10, q=0.0, sigma=0.20, is_call=True)
    p = price(spec)
    assert abs(p - 4.76) < 0.01, f"Expected ~4.76, got {p:.4f}"


def test_put_call_parity_holds_exactly():
    spec = OptionSpec(S=105, K=100, T=0.75, r=0.03, q=0.02, sigma=0.28, is_call=True)
    call_p = price(spec)
    put_p = price(replace(spec, is_call=False))
    residual = put_call_parity_residual(call_p, put_p, spec)
    assert abs(residual) < TOL, f"Put-call parity violated: residual={residual}"


def test_deep_itm_call_converges_to_intrinsic_forward():
    # deep ITM call: N(d1), N(d2) -> 1, price -> S*e^-qT - K*e^-rT
    spec = OptionSpec(S=1000, K=10, T=1.0, r=0.03, q=0.01, sigma=0.20, is_call=True)
    p = price(spec)
    from math import exp
    expected = spec.S * exp(-spec.q * spec.T) - spec.K * exp(-spec.r * spec.T)
    assert abs(p - expected) < 0.05


def test_zero_vol_matches_deterministic_forward_payoff():
    spec = OptionSpec(S=100, K=90, T=1.0, r=0.05, q=0.0, sigma=0.0, is_call=True)
    p = price(spec)
    from math import exp
    fwd = spec.S * exp(spec.r * spec.T)
    expected = exp(-spec.r * spec.T) * max(fwd - spec.K, 0)
    assert abs(p - expected) < 1e-6


def _finite_diff(spec, attr, bump, func=price):
    up = replace(spec, **{attr: getattr(spec, attr) + bump})
    down = replace(spec, **{attr: getattr(spec, attr) - bump})
    return (func(up) - func(down)) / (2 * bump)


def test_delta_matches_finite_difference():
    spec = OptionSpec(S=95, K=100, T=0.4, r=0.02, q=0.01, sigma=0.25, is_call=True)
    analytic = delta(spec)
    numeric = _finite_diff(spec, "S", 0.01)
    assert abs(analytic - numeric) < FD_TOL, f"delta: analytic={analytic:.6f} numeric={numeric:.6f}"


def test_gamma_matches_finite_difference_of_delta():
    spec = OptionSpec(S=95, K=100, T=0.4, r=0.02, q=0.01, sigma=0.25, is_call=False)
    analytic = gamma(spec)
    numeric = _finite_diff(spec, "S", 0.01, func=delta)
    assert abs(analytic - numeric) < FD_TOL, f"gamma: analytic={analytic:.6f} numeric={numeric:.6f}"


def test_vega_matches_finite_difference():
    spec = OptionSpec(S=95, K=100, T=0.4, r=0.02, q=0.01, sigma=0.25, is_call=True)
    analytic = vega(spec)
    numeric = _finite_diff(spec, "sigma", 0.0001) / 1  # per-1.00-vol units, matches engine convention
    assert abs(analytic - numeric) < FD_TOL, f"vega: analytic={analytic:.6f} numeric={numeric:.6f}"


def test_theta_matches_finite_difference():
    spec = OptionSpec(S=95, K=100, T=0.4, r=0.02, q=0.01, sigma=0.25, is_call=True)
    analytic = theta(spec)
    # theta is dPrice/dt = -dPrice/dT; bump T and negate
    numeric = -_finite_diff(spec, "T", 0.0005)
    assert abs(analytic - numeric) < FD_TOL, f"theta: analytic={analytic:.6f} numeric={numeric:.6f}"


def test_rho_matches_finite_difference():
    spec = OptionSpec(S=95, K=100, T=0.4, r=0.02, q=0.01, sigma=0.25, is_call=False)
    analytic = rho(spec)
    numeric = _finite_diff(spec, "r", 0.0001)
    assert abs(analytic - numeric) < FD_TOL, f"rho: analytic={analytic:.6f} numeric={numeric:.6f}"


def test_implied_vol_recovers_known_sigma():
    true_spec = OptionSpec(S=100, K=105, T=0.6, r=0.03, q=0.01, sigma=0.27, is_call=True)
    market_price = price(true_spec)
    recovered = implied_vol(replace(true_spec, sigma=0.0), market_price, initial_guess=0.2)
    assert abs(recovered - 0.27) < 1e-4, f"Expected 0.27, recovered {recovered:.6f}"


def test_implied_vol_rejects_arbitrage_violating_price():
    spec = OptionSpec(S=100, K=105, T=0.5, r=0.03, q=0.0, sigma=0.0, is_call=True)
    try:
        implied_vol(spec, market_price=-1.0)  # nonsensical negative price
        assert False, "Expected ImpliedVolError for arbitrage-violating price"
    except ImpliedVolError:
        pass


def run_all():
    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {passed + failed}")
    return failed == 0


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
