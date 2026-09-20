"""
black_scholes.visualize
=========================
Plotting utilities. Kept deliberately separate from analysis logic -- no
plot function computes anything; they only render arrays already produced
by engine/surface/backtest. This means every chart is directly traceable to
a numeric result printed elsewhere in the demo, not a chart doing silent
extra work.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- registers 3D projection

from .engine import OptionSpec, price, delta, gamma, vega, theta


def plot_payoff_and_price(S_range, K, T, r, q, sigma, is_call=True, ax=None):
    """Payoff at expiry vs. current theoretical price across a spot range --
    the standard chart for explaining time value vs intrinsic value."""
    ax = ax or plt.gca()
    payoff = np.maximum(S_range - K, 0) if is_call else np.maximum(K - S_range, 0)
    prices = [price(OptionSpec(S=s, K=K, T=T, r=r, q=q, sigma=sigma, is_call=is_call)) for s in S_range]
    ax.plot(S_range, payoff, "--", label="Payoff at expiry", color="gray")
    ax.plot(S_range, prices, label=f"Theoretical price (T={T:.2f}y)", color="C0")
    ax.axvline(K, color="black", linewidth=0.5, linestyle=":")
    ax.set_xlabel("Underlying price (S)")
    ax.set_ylabel("Value")
    ax.set_title(f"{'Call' if is_call else 'Put'} payoff vs. current price (K={K})")
    ax.legend()
    return ax


def plot_greeks_vs_spot(S_range, K, T, r, q, sigma, is_call=True):
    """4-panel Greeks sensitivity -- what actually gets asked in an
    interview walkthrough of this project."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    greeks_funcs = [("Delta", delta), ("Gamma", gamma), ("Vega", vega), ("Theta", theta)]
    for ax, (name, func) in zip(axes.flat, greeks_funcs):
        vals = [func(OptionSpec(S=s, K=K, T=T, r=r, q=q, sigma=sigma, is_call=is_call)) for s in S_range]
        ax.plot(S_range, vals, color="C1")
        ax.axvline(K, color="black", linewidth=0.5, linestyle=":")
        ax.set_title(name)
        ax.set_xlabel("Spot (S)")
    fig.suptitle(f"Greeks vs. spot -- {'Call' if is_call else 'Put'}, K={K}, T={T:.2f}y, sigma={sigma:.0%}")
    fig.tight_layout()
    return fig


def plot_iv_smile(iv_surface, tenor_days_target=30, tol_days=15, ax=None):
    """The single most important chart in the project: implied vol vs
    moneyness for one tenor, showing the smile/skew that a constant-sigma
    model cannot produce -- this IS the visual evidence for the flat-vol
    null-hypothesis test in surface.py."""
    ax = ax or plt.gca()
    df = iv_surface.copy()
    df["T_days"] = df["T"] * 365
    near = df[(df["T_days"] - tenor_days_target).abs() <= tol_days].sort_values("moneyness")
    ax.plot(near["moneyness"], near["implied_vol"], "o-", color="C2")
    ax.axhline(near["implied_vol"].mean(), linestyle="--", color="gray", label="flat-vol assumption")
    ax.set_xlabel("Moneyness (K/S)")
    ax.set_ylabel("Implied vol")
    ax.set_title(f"Implied vol smile, ~{tenor_days_target}d tenor")
    ax.legend()
    return ax


def plot_vol_surface_3d(mm, tt, iv_grid):
    """3D surface plot from surface.interpolate_surface_grid output."""
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(mm, tt * 365, iv_grid, cmap="viridis", edgecolor="none", alpha=0.9)
    ax.set_xlabel("Moneyness (K/S)")
    ax.set_ylabel("Tenor (days)")
    ax.set_zlabel("Implied vol")
    ax.set_title("Implied volatility surface")
    fig.colorbar(surf, shrink=0.5, aspect=10, label="Implied vol")
    return fig


def plot_iv_vs_hv_forecast(path, test_result, hv_forecast, iv_forecast, fwd_realized):
    """Time series overlay for the backtest -- shows visually whether IV or
    HV tracks the subsequently realized vol more closely."""
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(fwd_realized.index, fwd_realized, label="Forward realized vol (ground truth)",
            color="black", linewidth=1.5)
    ax.plot(hv_forecast.index, hv_forecast, label="Historical vol forecast", alpha=0.7)
    ax.plot(iv_forecast.index, iv_forecast, label="Implied vol forecast (synthetic)", alpha=0.7)
    ax.set_title(
        f"IV vs HV forecast quality -- winner: {test_result['winner']} "
        f"(IV MAE {test_result['iv_mae']:.4f} vs HV MAE {test_result['hv_mae']:.4f})"
    )
    ax.set_xlabel("Day")
    ax.set_ylabel("Annualised vol")
    ax.legend()
    fig.tight_layout()
    return fig
