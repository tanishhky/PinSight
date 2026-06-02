"""End-to-end RND extraction.

    raw chain  →  filter_chain  →  smile.fit  →  call_prices(grid)
              →  density_from_calls  →  attach_tails  →  RNDFit

The orchestrator stamps every fit with `as_of_ts` (the lookahead boundary)
and exposes a clean dataclass for downstream pricing / paper-trader use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from . import obs_compat as obs  # local re-export to avoid circular import
from .breeden_litzenberger import BLDensity, density_from_calls, moments
from .quality import CleanChain, QualityConfig, filter_chain
from .smile import SmileFit, SmileParams, call_prices, fit as fit_smile
from .tails import ExpTail, attach_tails


@dataclass(frozen=True)
class RNDFit:
    """Full risk-neutral density fit at a moment in time."""
    as_of_ts: str
    spot: float
    T: float                  # years to expiry
    r: float
    q: float
    # Smile diagnostics
    smile: SmileFit
    # Inner BL density (before tails)
    bl_inner: BLDensity
    # Stitched final density with GEV tails
    strikes: np.ndarray
    density: np.ndarray
    cdf: np.ndarray
    left_tail: ExpTail
    right_tail: ExpTail
    # Quality diagnostics
    n_input_strikes: int
    n_clean_strikes: int
    # Moments of the final density
    rnd_mean: float
    rnd_std: float
    rnd_skew: float
    rnd_kurtosis_excess: float
    # Sanity check: density should integrate to 1 within tolerance.
    integral: float

    def prob_above(self, K: float) -> float:
        """Pr(S_T > K). Used by pricing.fair as `prob_itm` for calls."""
        idx = int(np.searchsorted(self.strikes, K))
        if idx <= 0:
            return 1.0
        if idx >= len(self.strikes):
            return 0.0
        # Linear interp on CDF.
        K0, K1 = self.strikes[idx - 1], self.strikes[idx]
        F0, F1 = self.cdf[idx - 1], self.cdf[idx]
        cdf_at_K = F0 + (F1 - F0) * (K - K0) / (K1 - K0)
        return float(max(0.0, 1.0 - cdf_at_K))

    def prob_below(self, K: float) -> float:
        return 1.0 - self.prob_above(K)


def extract(df: pd.DataFrame, *,
            spot: float,
            as_of_ts: str,
            expiry_iso: str,
            r: float = 0.0,
            q: float = 0.0,
            grid_size: int = 500,
            quality_cfg: Optional[QualityConfig] = None) -> Optional[RNDFit]:
    """Run the full chain → RND pipeline.

    Returns None if the chain isn't fit-able (too few clean strikes, T<=0,
    or fit failure). Reasons are logged via obs.event.
    """
    # ── No-lookahead boundary check at the data-read step ──
    if "_snapshot_ts" in df.columns:
        max_snap = str(df["_snapshot_ts"].max())
        assert max_snap <= as_of_ts, (
            f"LOOKAHEAD VIOLATION: snapshot {max_snap} > as_of_ts {as_of_ts}"
        )

    # Stage 1: filter
    clean = filter_chain(df, spot=spot, as_of_ts=as_of_ts,
                          expiry_iso=expiry_iso, r=r, q=q,
                          cfg=quality_cfg)
    if clean.T <= 0 or len(clean.strikes) < 5:
        obs.event(channel="fit", kind="rnd.too_few_strikes",
                  level="WARNING", as_of_ts=as_of_ts,
                  n_clean=len(clean.strikes), T=clean.T)
        return None

    # Stage 2: smile fit
    try:
        smile = fit_smile(clean.strikes, clean.ivs,
                           spot=spot, T=clean.T, r=r, q=q)
    except Exception as exc:
        obs.event(channel="error", kind="rnd.smile_fail",
                  level="WARNING", as_of_ts=as_of_ts, err=str(exc))
        return None
    if smile.r_squared < 0.6:
        obs.event(channel="fit", kind="rnd.smile_low_r2",
                  level="WARNING", as_of_ts=as_of_ts,
                  r_squared=smile.r_squared)

    # Stage 3: dense strike grid + repriced calls
    K_lo, K_hi = float(clean.strikes.min()), float(clean.strikes.max())
    K_grid = np.linspace(K_lo, K_hi, grid_size)
    calls = call_prices(smile.params, K_grid, spot=spot, T=clean.T, r=r, q=q)

    # Stage 4: Breeden-Litzenberger second difference
    try:
        bl_inner = density_from_calls(K_grid, calls, r=r, T=clean.T)
    except Exception as exc:
        obs.event(channel="error", kind="rnd.bl_fail",
                  level="WARNING", as_of_ts=as_of_ts, err=str(exc))
        return None

    # Stage 5: GEV tail extrapolation (Figlewski 2010)
    K_full, q_full, left_tail, right_tail = attach_tails(
        bl_inner.strikes, bl_inner.density, bl_inner.cdf,
    )
    cdf_full = np.concatenate([[0.0],
                                np.cumsum((q_full[:-1] + q_full[1:]) / 2.0
                                          * np.diff(K_full))])
    if cdf_full[-1] > 0:
        cdf_full = cdf_full / cdf_full[-1]

    mom = moments(BLDensity(strikes=K_full, density=q_full, cdf=cdf_full,
                             raw_negative_mass=0.0, raw_total_mass=1.0))

    integral = float(np.trapz(q_full, K_full))

    obs.event(channel="fit", kind="rnd.fit_done", level="INFO",
              as_of_ts=as_of_ts, T_hours=clean.T * 8760,
              n_strikes=len(clean.strikes),
              r_squared=smile.r_squared,
              rnd_mean=mom["mean"], rnd_std=mom["std"],
              rnd_skew=mom["skew"], rnd_kurtosis=mom["kurtosis_excess"],
              integral=integral)

    return RNDFit(
        as_of_ts=as_of_ts, spot=spot, T=clean.T, r=r, q=q,
        smile=smile, bl_inner=bl_inner,
        strikes=K_full, density=q_full, cdf=cdf_full,
        left_tail=left_tail, right_tail=right_tail,
        n_input_strikes=clean.n_input,
        n_clean_strikes=len(clean.strikes),
        rnd_mean=mom["mean"], rnd_std=mom["std"],
        rnd_skew=mom["skew"], rnd_kurtosis_excess=mom["kurtosis_excess"],
        integral=integral,
    )
