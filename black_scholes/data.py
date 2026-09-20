"""
black_scholes.data
====================
Data ingestion layer. This is deliberately isolated from engine.py and
implied_vol.py -- pricing logic must never know or care where a number came
from. That separation is what lets this run on a Bloomberg terminal in one
environment and on synthetic/offline data in another (e.g. this sandbox,
which has no network path to Bloomberg) without touching a single pricing
formula.

BLOOMBERG FIELD MAP (for use on your terminal via the Python API / Excel):
    Underlying spot          : PX_LAST
    Historical vol (30d)     : HIST_CALL_IMP_VOL / VOLATILITY_30D
    Dividend yield           : EQY_DVD_YLD_EST (discrete -> converted below)
    Risk-free rate           : matched off the relevant government curve
                                (e.g. USSOFR / GBP SONIA / GT curve) at tenor T
    Option chain             : OMON <GO> for the monitor; OVDV <GO> for the
                                full option chain with strikes/expiries
    Option mid IV            : IVOL_MID
    Option mid price         : PX_MID (or (PX_BID+PX_ASK)/2 if PX_MID sparse)
    Option delta (quoted)    : DELTA_MID  -- useful as a cross-check against
                                this engine's own delta() output

HOW TO CONNECT (on a machine with terminal + Desktop API running):
    pip install blpapi xbbg
    from xbbg import blp
    chain = blp.bds('AAPL US Equity', 'OPT_CHAIN')
    fields = blp.bdp(chain['security_des'], ['PX_LAST', 'IVOL_MID', 'DELTA_MID'])
This module wraps exactly that call pattern in `fetch_option_chain_bloomberg`.
It is NOT executed here (no Bloomberg network path in this environment) --
it is written to run unmodified once pointed at a real terminal session, and
is unit-tested here only via the synthetic path.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from .engine import OptionSpec


@dataclass
class MarketSnapshot:
    """A single, immutable pull of everything the model needs, timestamped.
    Keeping this as one object (rather than passing loose floats around)
    means every downstream calculation can be traced back to exactly which
    data pull it used -- important when a client/reviewer asks 'as of when?'"""
    ticker: str
    as_of: date
    spot: float
    risk_free_rate: float     # continuously-compounded, matched to relevant tenor
    dividend_yield: float     # continuous-equivalent
    chain: pd.DataFrame       # columns: strike, expiry, is_call, market_price, market_iv (opt)


def discrete_to_continuous_yield(discrete_yield: float, compounding_freq: int = 1) -> float:
    """
    Bloomberg's EQY_DVD_YLD_EST is a discrete annual yield. Black-Scholes
    needs the continuous-compounding equivalent q such that
    e^q = (1 + discrete_yield/freq)^freq. Using the discrete figure directly
    as q is a common but real source of small systematic mispricing on
    high-yield names -- worth stating explicitly rather than silently
    getting slightly wrong.
    """
    return compounding_freq * np.log(1 + discrete_yield / compounding_freq)


def year_fraction(as_of: date, expiry: date, convention: str = "ACT/365") -> float:
    """Time-to-expiry in years. ACT/365 matches Bloomberg's default OVME/OVDV
    day-count for equity options; flag if your specific product uses ACT/360
    (common for some rates-linked structures) and pass convention accordingly."""
    days = (expiry - as_of).days
    if convention == "ACT/365":
        return max(days, 0) / 365.0
    elif convention == "ACT/360":
        return max(days, 0) / 360.0
    else:
        raise ValueError(f"Unsupported convention: {convention}")


def fetch_option_chain_bloomberg(ticker: str, as_of: Optional[date] = None) -> MarketSnapshot:
    """
    Real Bloomberg pull. Requires blpapi + an active Bloomberg terminal
    session with the Desktop API service running (bbcomm.exe). This function
    is not exercised in this sandbox (no network path to Bloomberg here) --
    run it locally once you've cloned this project onto a machine with
    terminal access, and it will populate a MarketSnapshot identically to
    the synthetic path used for the demo/tests.
    """
    try:
        from xbbg import blp
    except ImportError as e:
        raise ImportError(
            "xbbg/blpapi not installed or no terminal session available. "
            "Run `pip install blpapi xbbg` on a machine with Bloomberg Desktop "
            "API access. This function is intentionally not runnable in a "
            "sandboxed/offline environment -- see data.synthetic_snapshot() "
            "for the offline path used to validate the rest of this project."
        ) from e

    as_of = as_of or date.today()
    spot = float(blp.bdp(f"{ticker} Equity", "PX_LAST").iloc[0, 0])
    div_yield_discrete = float(blp.bdp(f"{ticker} Equity", "EQY_DVD_YLD_EST").iloc[0, 0]) / 100
    q = discrete_to_continuous_yield(div_yield_discrete)

    chain_tickers = blp.bds(f"{ticker} Equity", "OPT_CHAIN")
    # NOTE on field names: mnemonics below are the standard equity-option
    # fields as of this writing. Bloomberg revises/renames fields and
    # coverage varies by asset class and terminal configuration -- before
    # relying on any of these, run FLDS <GO> and confirm the exact mnemonic
    # for your instrument rather than assuming this list is authoritative.
    raw = blp.bdp(
        chain_tickers.iloc[:, 0].tolist(),
        ["OPT_STRIKE_PX", "OPT_EXPIRE_DT", "OPT_PUT_CALL", "PX_MID", "IVOL_MID",
         "DELTA_MID", "GAMMA_MID", "VEGA_MID", "THETA_MID"],
    )
    raw = raw.rename(columns={
        "OPT_STRIKE_PX": "strike", "OPT_EXPIRE_DT": "expiry",
        "OPT_PUT_CALL": "put_call", "PX_MID": "market_price", "IVOL_MID": "bbg_iv",
        "DELTA_MID": "bbg_delta", "GAMMA_MID": "bbg_gamma",
        "VEGA_MID": "bbg_vega", "THETA_MID": "bbg_theta",
    })
    raw["is_call"] = raw["put_call"].str.upper().eq("CALL")
    raw["bbg_iv"] = raw["bbg_iv"] / 100  # Bloomberg quotes IVOL_MID in percentage points

    # Rate: matched off the relevant curve at each option's tenor. Left as a
    # single flat estimate here for simplicity; a production version would
    # interpolate a full OIS curve per-expiry via blp.bdh on the swap curve.
    r = float(blp.bdp("USOSFR10 Curncy", "PX_LAST").iloc[0, 0]) / 100

    keep = ["strike", "expiry", "is_call", "market_price", "bbg_iv",
            "bbg_delta", "bbg_gamma", "bbg_vega", "bbg_theta"]
    return MarketSnapshot(
        ticker=ticker, as_of=as_of, spot=spot,
        risk_free_rate=r, dividend_yield=q,
        chain=raw[keep],
    )


def synthetic_snapshot(
    ticker: str = "DEMO",
    spot: float = 100.0,
    r: float = 0.045,
    q: float = 0.015,
    true_vol_smile: bool = True,
    seed: int = 7,
) -> MarketSnapshot:
    """
    Offline stand-in for fetch_option_chain_bloomberg, used ONLY so this
    project can be demonstrated, unit-tested and its methodology validated
    without a live terminal connection (i.e. in this environment).

    It does not pretend to be real market data. It generates a chain by
    picking a *true* implied-vol surface with a realistic smile/skew shape
    (vol rising for OTM puts, flatter for OTM calls -- the standard equity
    skew), pricing each contract off that surface with the real engine, then
    adding small bid/ask-like noise. This lets every downstream module
    (surface fitting, screener, backtest) be tested against ground truth
    with a KNOWN answer, which is a stronger validation than testing against
    real data with an unknown true surface.
    """
    rng = np.random.default_rng(seed)
    as_of = date(2026, 9, 18)
    strikes = np.arange(0.7, 1.35, 0.05) * spot
    expiries_days = [30, 60, 90, 180, 365]

    rows = []
    for d in expiries_days:
        expiry = date.fromordinal(as_of.toordinal() + d)
        T = d / 365.0
        for K in strikes:
            moneyness = np.log(K / spot)
            if true_vol_smile:
                # simple parametric skew: higher vol for downside strikes,
                # decaying with tenor (short-dated skew is steeper) --
                # deliberately synthetic, deliberately labelled as such.
                base_vol = 0.22
                skew = -0.35 * moneyness / np.sqrt(T + 0.1)
                curvature = 0.15 * moneyness ** 2 / (T + 0.25)
                true_iv = max(base_vol + skew + curvature, 0.03)
            else:
                true_iv = 0.22  # flat surface, for testing the "no smile" null case

            for is_call in (True, False):
                spec = OptionSpec(S=spot, K=K, T=T, r=r, q=q, sigma=true_iv, is_call=is_call)
                from .engine import price as bs_price
                true_price = bs_price(spec)
                noise = rng.normal(0, 0.003 * spot)  # ~ bid/ask microstructure noise
                rows.append({
                    "strike": K, "expiry": expiry, "is_call": is_call,
                    "market_price": max(true_price + noise, 0.001),
                    "market_iv": true_iv,  # "true" IV kept for validation only
                })

    return MarketSnapshot(
        ticker=ticker, as_of=as_of, spot=spot,
        risk_free_rate=r, dividend_yield=q,
        chain=pd.DataFrame(rows),
    )


def synthetic_bloomberg_snapshot(
    ticker: str = "DEMO",
    spot: float = 100.0,
    r: float = 0.045,
    q: float = 0.015,
    american_exercise_bias: bool = True,
    seed: int = 11,
) -> MarketSnapshot:
    """
    Offline stand-in for what fetch_option_chain_bloomberg would return,
    used ONLY to test and demonstrate validate_bloomberg.py without a live
    terminal connection.

    This deliberately builds in the single most common REAL source of
    divergence between a from-scratch European BS engine and Bloomberg's
    own quoted Greeks: Bloomberg's default equity option model is typically
    American-exercise (binomial/trinomial tree), which carries a genuine
    early-exercise premium -- largest for puts and for calls on high-
    dividend names. That premium is not a bug in either model; it's the two
    models pricing two different contract features (European vs American
    exercise). Setting american_exercise_bias=False produces a chain with
    no such bias, so the validation module can be tested against both the
    "should match closely" and "should show a structural, explainable gap"
    cases.
    """
    rng = np.random.default_rng(seed)
    as_of = date(2026, 9, 18)
    strikes = np.arange(0.8, 1.25, 0.05) * spot
    expiries_days = [30, 90, 180]

    from .engine import OptionSpec as _Spec, price as _price, all_greeks as _greeks

    rows = []
    for d in expiries_days:
        expiry = date.fromordinal(as_of.toordinal() + d)
        T = d / 365.0
        for K in strikes:
            true_iv = 0.22 + 0.05 * abs(np.log(K / spot))  # mild smile, same shape logic as synthetic_snapshot
            for is_call in (True, False):
                spec = _Spec(S=spot, K=K, T=T, r=r, q=q, sigma=true_iv, is_call=is_call)
                euro_price = _price(spec)
                g = _greeks(spec)

                premium = 0.0
                if american_exercise_bias:
                    # rough, illustrative early-exercise premium: larger for
                    # puts (always some premium for American puts) and for
                    # ITM dividend-paying calls; NOT a rigorous American
                    # pricing model -- just enough structure to demonstrate
                    # what a genuine, explainable divergence looks like.
                    moneyness = K / spot
                    if not is_call:
                        premium = euro_price * (0.01 + 0.04 * max(1.1 - moneyness, 0))
                    elif q > 0:
                        premium = euro_price * q * T * max(moneyness - 1, 0) * 2

                noise = rng.normal(0, 0.002 * spot)
                bbg_price = max(euro_price + premium + noise, 0.001)

                rows.append({
                    "strike": K, "expiry": expiry, "is_call": is_call,
                    "market_price": bbg_price,
                    "bbg_iv": true_iv + rng.normal(0, 0.003),
                    "bbg_delta": g["delta"] + (0.01 if premium > 0 else 0) * (1 if is_call else -1),
                    "bbg_gamma": g["gamma"] * (1 + 0.02 * rng.standard_normal()),
                    "bbg_vega": g["vega_per_vol_pt"] * (1 + 0.02 * rng.standard_normal()),
                    "bbg_theta": g["theta_per_day"] * (1 + 0.03 * rng.standard_normal()),
                })

    return MarketSnapshot(
        ticker=ticker, as_of=as_of, spot=spot,
        risk_free_rate=r, dividend_yield=q,
        chain=pd.DataFrame(rows),
    )
