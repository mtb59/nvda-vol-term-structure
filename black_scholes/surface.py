"""
black_scholes.surface
=======================
This module runs the central test of the whole project (methodology point 2
from the README): does a SINGLE implied volatility explain an entire option
chain, as the BS model's constant-sigma assumption requires? Or does implied
vol vary systematically by strike (smile/skew) and tenor (term structure)?

The answer is empirically always "it varies" for any real equity chain.
That is not a bug in this project -- reproducing and quantifying that
failure IS the deliverable. A flat-vol BS model that fits nothing is a much
stronger piece of work, explained properly, than a model that silently
"fits" because it was only ever tested at one strike.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
from scipy.interpolate import griddata

from .engine import OptionSpec
from .implied_vol import implied_vol, ImpliedVolError
from .data import MarketSnapshot, year_fraction


def build_iv_surface(snap: MarketSnapshot) -> pd.DataFrame:
    """
    Back out the market-implied vol for every contract in the chain, using
    OTM options only (standard convention -- OTM options are more liquid and
    their IVs are what feeds a real vol surface; ITM options are typically
    priced off the OTM IV via put-call parity rather than independently
    quoted).
    """
    rows = []
    for _, row in snap.chain.iterrows():
        T = year_fraction(snap.as_of, row["expiry"])
        if T <= 0:
            continue
        moneyness = row["strike"] / snap.spot
        is_otm = (row["is_call"] and moneyness > 1.0) or (not row["is_call"] and moneyness < 1.0)
        if not is_otm:
            continue
        spec = OptionSpec(S=snap.spot, K=row["strike"], T=T, r=snap.risk_free_rate,
                           q=snap.dividend_yield, sigma=0.2, is_call=row["is_call"])
        try:
            iv = implied_vol(spec, row["market_price"])
        except ImpliedVolError:
            continue  # stale/crossed quote -- excluded and would be flagged in a real run
        rows.append({
            "strike": row["strike"], "expiry": row["expiry"], "T": T,
            "moneyness": moneyness, "log_moneyness": np.log(moneyness),
            "is_call": row["is_call"], "implied_vol": iv,
        })
    return pd.DataFrame(rows)


def test_flat_vol_null_hypothesis(iv_surface: pd.DataFrame) -> dict:
    """
    Formal test of "does one vol explain the chain": fits the single best
    constant vol (the chain-wide average, weighted equally) and reports the
    RMSE of that flat assumption against actual per-contract implied vols.

    A large RMSE relative to the vol level itself is the quantitative
    evidence that BS's constant-vol assumption is violated -- this is the
    number you'd actually quote in an interview, not just "there's a smile".
    """
    if iv_surface.empty:
        raise ValueError("Empty IV surface -- check chain data / OTM filter")
    flat_vol = iv_surface["implied_vol"].mean()
    residuals = iv_surface["implied_vol"] - flat_vol
    rmse = float(np.sqrt((residuals ** 2).mean()))
    return {
        "flat_vol_estimate": flat_vol,
        "rmse_vs_flat": rmse,
        "rmse_as_pct_of_level": rmse / flat_vol,
        "min_iv": iv_surface["implied_vol"].min(),
        "max_iv": iv_surface["implied_vol"].max(),
        "range_iv": iv_surface["implied_vol"].max() - iv_surface["implied_vol"].min(),
        "n_contracts": len(iv_surface),
    }


def skew_metric(iv_surface: pd.DataFrame, tenor_days_target: int = 30, tol_days: int = 15) -> dict:
    """
    Quantifies skew the way a trader actually talks about it: the difference
    in implied vol between a ~25-delta-equivalent downside strike (proxied
    here by moneyness ~0.9) and a ~25-delta-equivalent upside strike
    (moneyness ~1.1), for the tenor closest to tenor_days_target.

    Positive skew_25d (put IV > call IV) is the standard equity index/single-
    name pattern (crash-risk premium priced into downside puts). A value near
    zero would itself be a notable, reportable finding (would suggest a name
    the market isn't pricing tail risk into -- worth knowing which names do
    that and why, in a real write-up).
    """
    iv_surface = iv_surface.copy()
    iv_surface["T_days"] = iv_surface["T"] * 365
    near_tenor = iv_surface[(iv_surface["T_days"] - tenor_days_target).abs() <= tol_days]
    if near_tenor.empty:
        return {"skew_25d": None, "note": f"no contracts within {tol_days}d of {tenor_days_target}d tenor"}

    downside = near_tenor.iloc[(near_tenor["moneyness"] - 0.90).abs().argsort()[:1]]
    upside = near_tenor.iloc[(near_tenor["moneyness"] - 1.10).abs().argsort()[:1]]
    if downside.empty or upside.empty:
        return {"skew_25d": None, "note": "insufficient strike coverage near target moneyness"}

    skew = float(downside["implied_vol"].values[0] - upside["implied_vol"].values[0])
    return {
        "skew_25d": skew,
        "downside_strike_moneyness": float(downside["moneyness"].values[0]),
        "upside_strike_moneyness": float(upside["moneyness"].values[0]),
        "tenor_days_used": float(downside["T_days"].values[0]),
    }


def interpolate_surface_grid(iv_surface: pd.DataFrame, n_strikes: int = 40, n_tenors: int = 40):
    """
    Produces a regular (moneyness x tenor) grid of interpolated implied vols
    via cubic interpolation, for the 3D surface plot in visualize.py.
    Returns (moneyness_grid, tenor_grid, iv_grid) as 2D numpy arrays.
    Extrapolation outside the convex hull of quoted strikes is deliberately
    left as NaN rather than guessed -- a vol surface should never silently
    invent a number outside the range it has evidence for.
    """
    points = iv_surface[["moneyness", "T"]].values
    values = iv_surface["implied_vol"].values
    m_grid = np.linspace(iv_surface["moneyness"].min(), iv_surface["moneyness"].max(), n_strikes)
    t_grid = np.linspace(iv_surface["T"].min(), iv_surface["T"].max(), n_tenors)
    mm, tt = np.meshgrid(m_grid, t_grid)
    iv_grid = griddata(points, values, (mm, tt), method="cubic")
    return mm, tt, iv_grid
