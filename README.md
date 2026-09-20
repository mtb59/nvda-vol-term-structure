# Does NVDA's options market actually price in a vol premium ahead of earnings?

Tested it on real option chains. The naive expectation didn't hold.

**The question:** NVIDIA reports earnings on Nov 25, 2026. Standard options-trading
intuition says the option expiry that spans an earnings date should show
higher implied volatility than the expiry just before it — the market pricing
in the extra uncertainty of the event.

**What I found instead:** using real, live Yahoo Finance option chains for
both the Nov 20, 2026 expiry (before earnings) and the Dec 18, 2026 expiry
(spans earnings), pulled by hand on 18 Sep 2026 — implied vol at every
matched strike was flat to *slightly lower* in the post-earnings expiry.
Running a standard variance-decomposition to back out an implied earnings
move gives a **negative** number, meaning the naive assumption underlying
that decomposition doesn't hold for this pair of expiries.

**My read on why:** most likely explanation is term-structure backwardation,
not the absence of an earnings premium. NVDA's baseline implied vol right
now (mid-30s to low-40s across the chain) is already elevated — consistent
with the live "AI capex bubble" debate in markets generally — and when
current vol is already high relative to its longer-run level, front-dated
options often price *above* back-dated ones regardless of scheduled events,
because the market's expecting reversion rather than escalation. The
earnings-specific premium is probably still there, just small relative to
that already-elevated baseline. Full reasoning, including the weaker
alternative explanations I considered and ruled less likely, is in the
notebook.

**What this isn't:** a trading signal, a claim that Black-Scholes "found
alpha," or a finished, publication-grade result. It's real data, an honest
test of a specific hypothesis, and a documented case where the hypothesis
failed and the more interesting question is why. See Limitations below.

---

## Run it yourself

Open `notebooks/nvda_earnings_vol_analysis.ipynb` in Google Colab
(File → Upload notebook) and Run All. No API keys, no live network calls —
the option chain data is embedded exactly as pulled, so it reproduces the
same numbers every time.

## What's in this repo

```
black_scholes/           tested pricing engine (closed-form BS, all Greeks,
                          Newton-Raphson implied-vol solver with bisection
                          fallback), see engine.py and implied_vol.py
tests/                    unit tests: known textbook value, put-call parity,
                          finite-difference Greek verification (11/11 passing)
notebooks/                the actual analysis, self-contained, real data
```

## Methodology, briefly

Black-Scholes assumes constant volatility, so it has no way to price a
discrete earnings jump directly. What it's used for here is as a
**translation layer**: convert real market prices into implied vol so
numbers across different strikes and expiries become comparable, then
reason about *why* those implied vols differ. The interpretation is doing
the analytical work, not the pricing formula itself.

## Limitations, stated plainly

- Rate and dividend yield are flat estimates, not pulled from a real curve.
- This engine prices European exercise; NVDA options are American, which
  carries a real (if generally small) early-exercise premium not captured
  here.
- Yahoo's data is free, single-source, and ~15 minutes delayed — not
  institutional-grade. A cross-check against Bloomberg-quoted IV/Greeks for
  the same contracts is the natural next step; kept private given
  Bloomberg's data-licensing terms rather than published here.
- The variance-decomposition method assumes a flat baseline vol rate
  carried forward from the near expiry — a simplifying assumption that
  visibly doesn't hold in this result. A cleaner version would use a third,
  longer-dated reference expiry with no nearby catalyst to separate
  backwardation from a genuine event effect.
- Single ticker, single date. Whether this pattern is NVDA-specific or
  general to high-vol megacaps heading into earnings right now is an open
  question, not yet tested.
