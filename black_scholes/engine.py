"""
black_scholes.engine
=====================
Closed-form Black-Scholes-Merton pricing engine (European options, continuous
dividend yield q).

PREMISE (stated explicitly, not left implicit):
This module prices options under the assumptions of Black-Scholes-Merton:
    - underlying follows geometric Brownian motion with constant volatility sigma
    - constant, known risk-free rate r and continuous dividend yield q
    - no transaction costs, frictionless short-selling, continuous trading
    - European exercise only
Under these assumptions, and only these assumptions, the prices below are the
unique no-arbitrage prices. Any divergence from observed market prices is not
"model error" in the naive sense -- it is information about which assumption
the market is pricing as false (see surface.py and screener.py, which turn
that divergence into the actual analytical output of this project).

All formulas here are the standard closed forms (Black-Scholes-Merton 1973/1973
generalisation with continuous yield, Merton 1973). Nothing is approximated
unless explicitly noted (see implied_vol.py for the one place a numerical
solver is required, because there is no closed-form inverse for implied vol).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log, sqrt, exp, pi

from scipy.stats import norm

N = norm.cdf   # standard normal CDF
n = norm.pdf   # standard normal PDF


@dataclass(frozen=True)
class OptionSpec:
    """
    Fully specifies a single European option contract for pricing.

    S     : spot price of the underlying (Bloomberg: PX_LAST)
    K     : strike price
    T     : time to expiry, in years (ACT/365 by convention here; see data.py
            for how this is derived from a Bloomberg expiry date)
    r     : continuously-compounded risk-free rate, decimal (e.g. Bloomberg
            USSOFR / GBP SONIA curve point matched to T)
    q     : continuous dividend yield, decimal (Bloomberg: EQY_DVD_YLD_EST,
            converted from discrete to continuous -- see data.py)
    sigma : annualised volatility, decimal
    is_call: True for call, False for put
    """
    S: float
    K: float
    T: float
    r: float
    q: float
    sigma: float
    is_call: bool = True

    def __post_init__(self):
        # Fail loudly rather than silently producing nonsense prices.
        if self.S <= 0:
            raise ValueError(f"Spot must be positive, got {self.S}")
        if self.K <= 0:
            raise ValueError(f"Strike must be positive, got {self.K}")
        if self.T < 0:
            raise ValueError(f"Time to expiry cannot be negative, got {self.T}")
        if self.sigma < 0:
            raise ValueError(f"Volatility cannot be negative, got {self.sigma}")


def _d1_d2(o: OptionSpec) -> tuple[float, float]:
    """
    The two standardised distance measures at the core of the model.

    d1 measures (roughly) how far in-the-money the option is, in units of
    volatility-adjusted standard deviations, drift-adjusted for the cost of
    carry (r - q). d2 = d1 - sigma*sqrt(T) is the same measure under the
    risk-neutral probability that the option actually finishes in the money
    (as opposed to the "share-weighted" probability d1 corresponds to).

    Edge case handled explicitly: T == 0 (expiry). d1/d2 are undefined in the
    limit unless S==K; pricing at T=0 is handled separately as intrinsic
    value in price(), so this function is never called with T==0 from price().
    """
    if o.T <= 0:
        raise ValueError("d1/d2 undefined at T=0; use intrinsic value instead")
    vol_sqrt_t = o.sigma * sqrt(o.T)
    if vol_sqrt_t == 0:
        # sigma == 0: no randomness, deterministic forward. d1/d2 -> +/-inf
        # depending on sign of (forward - K); handle as a limit.
        fwd = o.S * exp((o.r - o.q) * o.T)
        return (float("inf") if fwd > o.K else float("-inf"),) * 2
    d1 = (log(o.S / o.K) + (o.r - o.q + 0.5 * o.sigma ** 2) * o.T) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return d1, d2


def price(o: OptionSpec) -> float:
    """No-arbitrage Black-Scholes-Merton price."""
    if o.T == 0:
        # At expiry the option is worth exactly intrinsic value -- this is
        # not a BS formula output, it's a boundary condition BS must satisfy.
        return max(o.S - o.K, 0.0) if o.is_call else max(o.K - o.S, 0.0)

    d1, d2 = _d1_d2(o)
    disc_q = exp(-o.q * o.T)
    disc_r = exp(-o.r * o.T)

    if o.is_call:
        return o.S * disc_q * N(d1) - o.K * disc_r * N(d2)
    else:
        return o.K * disc_r * N(-d2) - o.S * disc_q * N(-d1)


# ----------------------------------------------------------------------------
# GREEKS
# Each one is the closed-form analytic partial derivative, not a finite
# difference. Finite-difference cross-checks live in tests/test_engine.py --
# analytic and numerical derivatives are required to agree to 1e-4 or the
# test suite fails. This is the actual verification step, not a decorative one.
# ----------------------------------------------------------------------------

def delta(o: OptionSpec) -> float:
    """dPrice/dS. Sensitivity to a 1-unit move in the underlying."""
    if o.T == 0:
        itm = (o.S > o.K) if o.is_call else (o.S < o.K)
        return (1.0 if itm else 0.0) if o.is_call else (-1.0 if itm else 0.0)
    d1, _ = _d1_d2(o)
    disc_q = exp(-o.q * o.T)
    return disc_q * N(d1) if o.is_call else disc_q * (N(d1) - 1)


def gamma(o: OptionSpec) -> float:
    """d(Delta)/dS. Identical for calls and puts (put-call parity: their
    deltas differ by a constant, so second derivatives coincide)."""
    if o.T == 0 or o.sigma == 0:
        return 0.0
    d1, _ = _d1_d2(o)
    disc_q = exp(-o.q * o.T)
    return disc_q * n(d1) / (o.S * o.sigma * sqrt(o.T))


def vega(o: OptionSpec) -> float:
    """dPrice/dSigma. Identical for calls and puts. Convention: returned per
    1.00 (100%) change in vol; divide by 100 for the market convention of
    'per 1 vol point'."""
    if o.T == 0 or o.sigma == 0:
        return 0.0
    d1, _ = _d1_d2(o)
    disc_q = exp(-o.q * o.T)
    return o.S * disc_q * n(d1) * sqrt(o.T)


def theta(o: OptionSpec) -> float:
    """dPrice/dt (time decay). Returned per year; divide by 365 for the
    market convention of 'per calendar day'."""
    if o.T == 0:
        return 0.0
    d1, d2 = _d1_d2(o)
    disc_q = exp(-o.q * o.T)
    disc_r = exp(-o.r * o.T)
    term1 = -disc_q * o.S * n(d1) * o.sigma / (2 * sqrt(o.T))
    if o.is_call:
        term2 = -o.r * o.K * disc_r * N(d2)
        term3 = o.q * o.S * disc_q * N(d1)
    else:
        term2 = o.r * o.K * disc_r * N(-d2)
        term3 = -o.q * o.S * disc_q * N(-d1)
    return term1 + term2 + term3


def rho(o: OptionSpec) -> float:
    """dPrice/dr. Returned per 1.00 (100%) change in rates; divide by 100
    for 'per 1% rate move'."""
    if o.T == 0:
        return 0.0
    _, d2 = _d1_d2(o)
    disc_r = exp(-o.r * o.T)
    return (o.K * o.T * disc_r * N(d2) if o.is_call
            else -o.K * o.T * disc_r * N(-d2))


def vanna(o: OptionSpec) -> float:
    """d(Delta)/dSigma == d(Vega)/dS. Second-order cross-Greek: how delta-
    hedge ratios shift as vol moves. Included because it's the first thing
    that matters once you go beyond a single static hedge -- relevant to
    the volatility-surface work in surface.py."""
    if o.T == 0 or o.sigma == 0:
        return 0.0
    d1, d2 = _d1_d2(o)
    disc_q = exp(-o.q * o.T)
    return -disc_q * n(d1) * d2 / o.sigma


def volga(o: OptionSpec) -> float:
    """d(Vega)/dSigma. Convexity of price with respect to vol -- exposure to
    vol-of-vol, the reason a smile has curvature and not just slope."""
    if o.T == 0 or o.sigma == 0:
        return 0.0
    d1, d2 = _d1_d2(o)
    return vega(o) * d1 * d2 / o.sigma


def all_greeks(o: OptionSpec) -> dict:
    """Convenience bundle, in the units actually shown on a Bloomberg OMON
    screen (theta/day, vega/vol-pt, rho/1%) so outputs are directly
    comparable to IVOL_MID-derived Greeks on the terminal."""
    return {
        "price": price(o),
        "delta": delta(o),
        "gamma": gamma(o),
        "vega_per_vol_pt": vega(o) / 100,
        "theta_per_day": theta(o) / 365,
        "rho_per_pct": rho(o) / 100,
        "vanna": vanna(o),
        "volga": volga(o),
    }


def put_call_parity_residual(call_price: float, put_price: float, o: OptionSpec) -> float:
    """
    C - P - (S*e^-qT - K*e^-rT). Should be ~0 to float precision for any
    correctly-implemented pair. This is the primary internal-consistency
    check referenced in the project README: it validates the ENGINE, not
    the market. A non-zero residual here means a bug, not an opportunity.
    """
    fwd_value = o.S * exp(-o.q * o.T) - o.K * exp(-o.r * o.T)
    return (call_price - put_price) - fwd_value
