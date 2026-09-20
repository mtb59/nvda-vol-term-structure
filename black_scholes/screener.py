"""
black_scholes.screener
========================
Ranks contracts in a chain by how far their market-implied vol sits from a
chosen benchmark vol (e.g. historical realized vol, or a smoothed/fitted
surface vol excluding that point). This is the "mispricing screener"
referenced on the CV -- stated honestly, not as an alpha-generation claim.

WHAT THIS ACTUALLY IS AND ISN'T:
It is a tool for surfacing which contracts have the largest IV-vs-benchmark
divergence, ranked and filtered by materiality (spread-adjusted). It is NOT
a claim that the divergence is exploitable -- a genuinely large divergence
is far more often explained by liquidity, an unpriced corporate event, or a
stale quote than by real mispricing. The screener's job is to surface
candidates for investigation, not to assert an edge. Any real write-up of
this project should say that explicitly, because overclaiming here is
exactly the kind of thing that gets a candidate's technical credibility
questioned in an interview.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .surface import build_iv_surface
from .data import MarketSnapshot


def screen_chain(
    snap: MarketSnapshot,
    benchmark_vol: float,
    min_abs_diff: float = 0.02,
    bid_ask_pct: float = 0.02,
) -> pd.DataFrame:
    """
    benchmark_vol: a single reference vol to compare each contract's implied
        vol against -- typically a trailing historical realized vol (see
        backtest.realized_vol) or a robust central estimate of the surface.
        Deliberately passed in rather than computed here: the choice of
        benchmark is an analytical decision, not something to bury inside
        a screener function.
    min_abs_diff: minimum |IV - benchmark| (in vol points, decimal) to be
        reported at all -- filters noise-level divergences.
    bid_ask_pct: assumed round-trip cost as a fraction of price, used to
        flag whether a divergence is large enough to plausibly survive
        transaction costs. This is a rough materiality filter, not a real
        transaction-cost model (no borrow cost, no margin, no market impact
        -- all explicitly out of scope and should be named as such in any
        write-up).
    """
    iv_surface = build_iv_surface(snap)
    if iv_surface.empty:
        return pd.DataFrame()

    merged = iv_surface.merge(
        snap.chain[["strike", "expiry", "is_call", "market_price"]],
        on=["strike", "expiry", "is_call"], how="left",
    )
    merged["vol_diff"] = merged["implied_vol"] - benchmark_vol
    merged["abs_vol_diff"] = merged["vol_diff"].abs()
    merged["signal"] = np.where(merged["vol_diff"] > 0, "IV rich vs benchmark", "IV cheap vs benchmark")
    merged["est_cost_vol_equiv"] = bid_ask_pct  # placeholder flag column, see docstring
    merged["passes_materiality_filter"] = merged["abs_vol_diff"] >= min_abs_diff

    result = merged[merged["passes_materiality_filter"]].sort_values("abs_vol_diff", ascending=False)
    return result[[
        "strike", "expiry", "T", "is_call", "moneyness",
        "implied_vol", "market_price", "vol_diff", "signal",
    ]].reset_index(drop=True)
