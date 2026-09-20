"""
black_scholes.implied_vol
===========================
Solves for the sigma that makes the BS price equal an observed market price.

WHY THIS IS THE ONLY NUMERICAL PIECE IN THE PROJECT:
Price = f(sigma) has no closed-form inverse. Everywhere else in this project,
"solve for X" is either closed-form (the Greeks) or a deliberate numerical
fit (surface.py, by design). Here it's a genuine root-find: given a market
price, find the sigma Black-Scholes would need to reproduce it.

METHOD CHOICE, JUSTIFIED:
Newton-Raphson using vega as the derivative, because vega is available in
closed form (engine.vega) and the price-vs-sigma function is smooth,
monotonic and has a well-behaved (positive) derivative almost everywhere for
options with real time value -- Newton-Raphson converges in ~4-6 iterations
for realistic option prices. It fails (or converges slowly) near expiry / deep
ITM-OTM where vega is close to zero, so a bisection fallback with a wide
bracket is used as a safety net rather than letting Newton-Raphson diverge
silently. This fallback is not decorative -- it is exercised routinely on
short-dated, deep OTM Bloomberg option chains.
"""

from __future__ import annotations

from dataclasses import replace

from .engine import OptionSpec, price, vega

MAX_NEWTON_ITER = 50
NEWTON_TOL = 1e-8
BISECT_TOL = 1e-6
BISECT_MAX_ITER = 200
VOL_LOWER_BOUND = 1e-4
VOL_UPPER_BOUND = 5.0  # 500% annualised vol -- generous upper bracket


class ImpliedVolError(RuntimeError):
    """Raised when no volatility in [VOL_LOWER_BOUND, VOL_UPPER_BOUND]
    reproduces the observed price. This is itself informative: it usually
    means the quoted market price violates a no-arbitrage bound (e.g. below
    intrinsic value), which happens on real, stale or illiquid Bloomberg
    quotes and should be surfaced, not swallowed."""


def _price_at_sigma(o: OptionSpec, sigma: float) -> float:
    return price(replace(o, sigma=sigma))


def _vega_at_sigma(o: OptionSpec, sigma: float) -> float:
    return vega(replace(o, sigma=sigma))


def _check_no_arbitrage_bounds(o: OptionSpec, market_price: float) -> None:
    """A market price below intrinsic value, or above the undiscounted
    underlying, cannot correspond to ANY volatility. Checking this explicitly
    (rather than letting the solver fail opaquely) is the difference between
    a script and a tool that tells you why it failed."""
    from math import exp
    disc_q = exp(-o.q * o.T)
    disc_r = exp(-o.r * o.T)
    if o.is_call:
        lower = max(o.S * disc_q - o.K * disc_r, 0.0)
        upper = o.S * disc_q
    else:
        lower = max(o.K * disc_r - o.S * disc_q, 0.0)
        upper = o.K * disc_r
    if not (lower - 1e-8 <= market_price <= upper + 1e-8):
        raise ImpliedVolError(
            f"Market price {market_price:.4f} violates no-arbitrage bounds "
            f"[{lower:.4f}, {upper:.4f}] for this contract -- check for a "
            f"stale/crossed Bloomberg quote before treating this as a model output."
        )


def implied_vol(o: OptionSpec, market_price: float, initial_guess: float = 0.2) -> float:
    """
    Solve for sigma given an observed market_price. `o.sigma` is ignored
    (it's overwritten during the search) -- pass any placeholder there.
    """
    _check_no_arbitrage_bounds(o, market_price)

    if o.T <= 0:
        raise ImpliedVolError("Cannot back out implied vol at/after expiry")

    # --- Newton-Raphson pass ---
    sigma = initial_guess
    for _ in range(MAX_NEWTON_ITER):
        model_price = _price_at_sigma(o, sigma)
        diff = model_price - market_price
        if abs(diff) < NEWTON_TOL:
            return sigma
        v = _vega_at_sigma(o, sigma)
        if v < 1e-10:
            break  # vega too flat -- hand off to bisection
        sigma = sigma - diff / v
        if sigma <= 0:
            sigma = 1e-4
            break  # stepped out of domain -- hand off to bisection

    # --- Bisection fallback (guaranteed to converge if a root exists in-bracket) ---
    lo, hi = VOL_LOWER_BOUND, VOL_UPPER_BOUND
    f_lo = _price_at_sigma(o, lo) - market_price
    f_hi = _price_at_sigma(o, hi) - market_price
    if f_lo * f_hi > 0:
        raise ImpliedVolError(
            "No sign change across [0.01%, 500%] vol -- solver cannot bracket "
            "a root even though the no-arbitrage bound check passed. Flag for "
            "manual review rather than trusting a numerical extrapolation."
        )
    for _ in range(BISECT_MAX_ITER):
        mid = 0.5 * (lo + hi)
        f_mid = _price_at_sigma(o, mid) - market_price
        if abs(f_mid) < BISECT_TOL or (hi - lo) < BISECT_TOL:
            return mid
        if f_lo * f_mid <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)  # best available estimate if tolerance never quite hit
