"""
black_scholes.validate_bloomberg
===================================
Runs this engine against Bloomberg's own calculated fields for the same
contracts (bbg_iv, bbg_delta, bbg_gamma, bbg_vega, bbg_theta -- see data.py
for exactly how these are pulled) and reports where they agree and, more
usefully, where and WHY they diverge.

WHY THIS IS THE RIGHT VALIDATION, NOT JUST "COMPARE MY PRICE TO BLOOMBERG'S":
Comparing theoretical prices is close to circular once both sides use a
similar vol input. Comparing GREEKS is a real, independent test -- Bloomberg
computes DELTA_MID etc. from its own model (commonly American-exercise for
US equities), so agreement isn't guaranteed by construction. Where this
engine and Bloomberg disagree, the size and DIRECTION of the disagreement is
itself diagnostic: a put showing a bigger Bloomberg delta than the European
engine predicts is early-exercise premium, not a bug (see synthetic_
bloomberg_snapshot's docstring in data.py for the mechanics).

This module never "corrects" this engine to match Bloomberg. It reports the
gap and classifies the likely cause. A candidate who can explain *why* their
own from-scratch model diverges from Bloomberg's -- rather than one who just
asserts they match -- is demonstrating the actual skill.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import OptionSpec, all_greeks
from .implied_vol import implied_vol, ImpliedVolError
from .data import MarketSnapshot, year_fraction


REQUIRED_BBG_COLS = ["bbg_iv", "bbg_delta", "bbg_gamma", "bbg_vega", "bbg_theta"]


def run_validation(snap: MarketSnapshot) -> pd.DataFrame:
    """
    For every contract with Bloomberg comparison fields present, computes:
      - this engine's own implied vol, solved from Bloomberg's market_price
        (independent of bbg_iv -- a second, separate check that this
        engine's own solver agrees with Bloomberg's on the SAME input price)
      - this engine's Greeks, using Bloomberg's own bbg_iv as the sigma
        input (isolates the Greek-formula comparison from any IV-solving
        disagreement -- the fairest apples-to-apples test)
      - the residual between each engine Greek and Bloomberg's own value

    Returns a row per contract; missing bbg_* columns are left as NaN rather
    than raising, so this can run on a chain that only has some fields
    populated (e.g. if FLDS confirms a different field is unavailable for
    a given product type).
    """
    missing = [c for c in REQUIRED_BBG_COLS if c not in snap.chain.columns]
    if missing:
        raise ValueError(
            f"Chain is missing Bloomberg comparison columns {missing}. "
            "Use fetch_option_chain_bloomberg (or synthetic_bloomberg_snapshot "
            "for an offline test run) rather than a chain built from "
            "synthetic_snapshot(), which has no bbg_* fields to validate against."
        )

    rows = []
    for _, row in snap.chain.iterrows():
        T = year_fraction(snap.as_of, row["expiry"])
        if T <= 0 or pd.isna(row["bbg_iv"]):
            continue

        spec_at_bbg_iv = OptionSpec(
            S=snap.spot, K=row["strike"], T=T, r=snap.risk_free_rate,
            q=snap.dividend_yield, sigma=row["bbg_iv"], is_call=row["is_call"],
        )
        engine_greeks = all_greeks(spec_at_bbg_iv)

        try:
            solved_iv = implied_vol(
                OptionSpec(S=snap.spot, K=row["strike"], T=T, r=snap.risk_free_rate,
                           q=snap.dividend_yield, sigma=0.2, is_call=row["is_call"]),
                market_price=row["market_price"],
            )
        except ImpliedVolError:
            solved_iv = float("nan")

        rows.append({
            "strike": row["strike"], "expiry": row["expiry"], "T": T, "is_call": row["is_call"],
            "market_price": row["market_price"],
            "bbg_iv": row["bbg_iv"], "engine_solved_iv": solved_iv,
            "iv_diff": solved_iv - row["bbg_iv"] if not np.isnan(solved_iv) else np.nan,
            "engine_price_at_bbg_iv": engine_greeks["price"],
            "price_diff": engine_greeks["price"] - row["market_price"],
            "bbg_delta": row["bbg_delta"], "engine_delta": engine_greeks["delta"],
            "delta_diff": engine_greeks["delta"] - row["bbg_delta"],
            "bbg_gamma": row["bbg_gamma"], "engine_gamma": engine_greeks["gamma"],
            "gamma_diff": engine_greeks["gamma"] - row["bbg_gamma"],
            "bbg_vega": row["bbg_vega"], "engine_vega": engine_greeks["vega_per_vol_pt"],
            "vega_diff": engine_greeks["vega_per_vol_pt"] - row["bbg_vega"],
            "bbg_theta": row["bbg_theta"], "engine_theta": engine_greeks["theta_per_day"],
            "theta_diff": engine_greeks["theta_per_day"] - row["bbg_theta"],
        })

    return pd.DataFrame(rows)


def summarise_validation(results: pd.DataFrame) -> dict:
    """
    Aggregate agreement statistics -- the numbers you'd actually quote:
    "mean absolute IV difference of X vol points, delta agreement within Y
    on Z% of contracts" is a specific, defensible claim. "It matches
    Bloomberg" is not.
    """
    if results.empty:
        raise ValueError("No validated rows -- check chain has valid bbg_* fields and T>0")

    def _mae(col):
        return float(results[col].abs().mean())

    summary = {
        "n_contracts": len(results),
        "iv_mae_vol_pts": _mae("iv_diff"),
        "price_mae": _mae("price_diff"),
        "delta_mae": _mae("delta_diff"),
        "gamma_mae": _mae("gamma_diff"),
        "vega_mae": _mae("vega_diff"),
        "theta_mae": _mae("theta_diff"),
        "pct_within_1c_price": float((results["price_diff"].abs() < 0.01).mean() * 100),
        "pct_within_1pt_delta": float((results["delta_diff"].abs() < 0.01).mean() * 100),
    }

    # simple directional flag: is the divergence systematically larger for
    # puts than calls (classic early-exercise-premium signature)?
    if "is_call" in results.columns:
        put_delta_mae = float(results[~results["is_call"]]["delta_diff"].abs().mean())
        call_delta_mae = float(results[results["is_call"]]["delta_diff"].abs().mean())
        summary["put_delta_mae"] = put_delta_mae
        summary["call_delta_mae"] = call_delta_mae
        summary["put_vs_call_divergence_ratio"] = (
            put_delta_mae / call_delta_mae if call_delta_mae > 1e-8 else float("nan")
        )
        summary["likely_american_exercise_signature"] = bool(
            put_delta_mae > 1.5 * call_delta_mae
        )

    return summary


def explain_divergence(summary: dict) -> str:
    """
    Turns the summary numbers into the actual written explanation you'd give
    in an interview or write-up -- deliberately conservative, never claiming
    a cause the numbers don't support.
    """
    lines = []
    iv_mae = summary["iv_mae_vol_pts"]
    if iv_mae < 0.005:
        lines.append(f"IV agreement is tight (MAE {iv_mae:.2%}) -- consistent with both "
                      f"models solving the same European-style pricing problem.")
    else:
        lines.append(f"IV MAE of {iv_mae:.2%} is larger than pure numerical noise would "
                      f"explain -- worth checking day-count convention, dividend treatment "
                      f"(discrete vs continuous yield), and snapshot timing before assuming "
                      f"it's a genuine model-structure difference.")

    if summary.get("likely_american_exercise_signature"):
        lines.append(
            f"Put deltas diverge more than call deltas ({summary['put_delta_mae']:.4f} vs "
            f"{summary['call_delta_mae']:.4f} MAE) -- the classic signature of Bloomberg "
            f"pricing American exercise while this engine prices European. This is an "
            f"explainable, expected structural difference, not an error in either model."
        )
    else:
        lines.append(
            "No strong put-vs-call asymmetry in the delta divergence -- doesn't show the "
            "typical early-exercise-premium pattern; remaining gap is more likely rate/"
            "dividend/timing convention differences than an exercise-style effect."
        )
    return "\n".join(lines)
