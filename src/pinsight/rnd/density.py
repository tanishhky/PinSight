"""End-to-end RND extraction.

    raw chain  →  filter_chain  →  svi.fit  →  call_prices(grid)
              →  density_from_calls  →  attach_tails  →  RNDFit

Production pipeline (post-fix 2026-06-03):
  - Smile fit: Gatheral SVI (5-param) — replaces the legacy quadratic
  - Grid: 2,000 points (4× denser than legacy 500) for cleaner BL diff
  - Tails: GEV per Figlewski (2010), with exp-decay fallback if GEV
    fit fails
  - Lookahead: parsed-datetime comparison, not string lex sort
  - R² gate: returns None when fit quality is below threshold or NaN
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Union

import numpy as np
import pandas as pd

from . import obs_compat as obs
from .breeden_litzenberger import BLDensity, density_from_calls, moments
from .quality import CleanChain, QualityConfig, filter_chain
from .svi import SVIFit, SVIParams, call_price_from_params, fit as fit_svi
from .tails import Tail, attach_tails


_TRAPEZOID = getattr(np, "trapezoid", None) or np.trapz


def _parse_iso(ts: str) -> datetime:
    """Parse an ISO 8601 string to a tz-aware UTC datetime.

    Accepts both "+00:00" and "Z" suffixes. All return values are
    normalised to UTC.
    """
    s = str(ts).replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _smile_r_squared(svi: SVIFit, ivs_observed: np.ndarray) -> float:
    """R² of the SVI fit in IV space.

    SVI internally tracks w_data / w_fit (total variance). For the R² gate
    we want a comparison the user can interpret (IV space), so we recompute.
    """
    iv_fit = np.sqrt(np.maximum(svi.w_fit, 1e-12) /
                     max(svi.k_data.size and svi.w_data.size and 1.0, 1.0))
    # ↑ w = σ²·T, so σ = √(w/T). We don't have T directly here; use w_data
    # and ivs_observed to back out a scale.
    if ivs_observed.size < 2:
        return float("nan")
    # Use IVs in observed space directly:
    iv_obs = ivs_observed
    # Convert w_fit → IV_fit using same T inferred from observed:
    # iv_obs² · T = w_data ⇒ T = w_data / iv_obs²  ⇒ pick first element.
    T_inferred = float(svi.w_data[0] / max(iv_obs[0] ** 2, 1e-12))
    if T_inferred <= 0:
        return float("nan")
    iv_fit_real = np.sqrt(np.maximum(svi.w_fit, 1e-12) / T_inferred)
    residuals = iv_fit_real - iv_obs
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((iv_obs - iv_obs.mean()) ** 2))
    if ss_tot < 1e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot


@dataclass(frozen=True)
class RNDFit:
    """Full risk-neutral density fit at a moment in time."""
    as_of_ts: str
    spot: float
    T: float
    r: float
    q: float
    smile: SVIFit
    smile_r_squared: float
    bl_inner: BLDensity
    strikes: np.ndarray
    density: np.ndarray
    cdf: np.ndarray
    left_tail: Tail
    right_tail: Tail
    n_input_strikes: int
    n_clean_strikes: int
    rnd_mean: float
    rnd_std: float
    rnd_skew: float
    rnd_kurtosis_excess: float
    integral: float
    grid_size: int

    def prob_above(self, K: float) -> float:
        idx = int(np.searchsorted(self.strikes, K))
        if idx <= 0:
            return 1.0
        if idx >= len(self.strikes):
            return 0.0
        K0, K1 = self.strikes[idx - 1], self.strikes[idx]
        F0, F1 = self.cdf[idx - 1], self.cdf[idx]
        cdf_at_K = F0 + (F1 - F0) * (K - K0) / (K1 - K0)
        return float(max(0.0, 1.0 - cdf_at_K))

    def prob_below(self, K: float) -> float:
        return 1.0 - self.prob_above(K)


# R² floor below which the fit is rejected. SVI on a clean 0DTE chain
# routinely produces 0.95+; below 0.6 the recovered density is not
# interpretable.
R2_FLOOR = 0.60


def extract(df: pd.DataFrame, *,
            spot: float,
            as_of_ts: str,
            expiry_iso: str,
            r: float = 0.0,
            q: float = 0.0,
            grid_size: int = 2000,
            quality_cfg: Optional[QualityConfig] = None) -> Optional[RNDFit]:
    """Run the full chain → RND pipeline.

    Returns None if the chain isn't fit-able (too few clean strikes, T<=0,
    SVI fit failure, low R², or BL failure). Reasons logged via obs.event.
    """
    # ── No-lookahead boundary check (parsed-datetime, not string) ──
    if "_snapshot_ts" in df.columns:
        try:
            max_snap_str = str(df["_snapshot_ts"].max())
            max_snap_dt = _parse_iso(max_snap_str)
            as_of_dt = _parse_iso(as_of_ts)
        except Exception as exc:
            raise AssertionError(
                f"LOOKAHEAD CHECK FAILED to parse timestamps: "
                f"snapshot={df['_snapshot_ts'].max()!r} "
                f"as_of_ts={as_of_ts!r} err={exc}"
            ) from exc
        assert max_snap_dt <= as_of_dt, (
            f"LOOKAHEAD VIOLATION: snapshot {max_snap_dt.isoformat()} "
            f"> as_of_ts {as_of_dt.isoformat()}"
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

    # Stage 2: SVI smile fit
    try:
        svi = fit_svi(clean.strikes, clean.ivs,
                       spot=spot, T=clean.T, r=r, q=q)
    except Exception as exc:
        obs.event(channel="error", kind="rnd.svi_fail",
                  level="WARNING", as_of_ts=as_of_ts, err=str(exc))
        return None

    r_squared = _smile_r_squared(svi, clean.ivs)
    # R² gate: reject low or NaN. NaN passes `< 0.6` as False in Python,
    # so check explicitly.
    if not np.isfinite(r_squared) or r_squared < R2_FLOOR:
        obs.event(channel="fit", kind="rnd.smile_rejected",
                  level="WARNING", as_of_ts=as_of_ts,
                  r_squared=float(r_squared) if np.isfinite(r_squared) else None,
                  converged=svi.converged,
                  butterfly_violations=svi.butterfly_violations)
        return None

    if svi.butterfly_violations > 0:
        obs.event(channel="fit", kind="rnd.butterfly_warning",
                  level="WARNING", as_of_ts=as_of_ts,
                  violations=svi.butterfly_violations)

    # Stage 3: dense strike grid + repriced calls
    K_lo, K_hi = float(clean.strikes.min()), float(clean.strikes.max())
    K_grid = np.linspace(K_lo, K_hi, grid_size)
    calls = call_price_from_params(svi.params, K_grid,
                                     spot=spot, T=clean.T, r=r, q=q)

    # Stage 4: Breeden-Litzenberger second difference
    try:
        bl_inner = density_from_calls(K_grid, calls, r=r, T=clean.T)
    except Exception as exc:
        obs.event(channel="error", kind="rnd.bl_fail",
                  level="WARNING", as_of_ts=as_of_ts, err=str(exc))
        return None

    # Stage 5: GEV tail extrapolation with exp-decay fallback
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

    integral = float(_TRAPEZOID(q_full, K_full))

    obs.event(channel="fit", kind="rnd.fit_done", level="INFO",
              as_of_ts=as_of_ts, T_hours=clean.T * 8760,
              n_strikes=len(clean.strikes),
              r_squared=r_squared,
              svi_rmse=svi.rmse,
              left_tail_type=left_tail.__class__.__name__,
              right_tail_type=right_tail.__class__.__name__,
              rnd_mean=mom["mean"], rnd_std=mom["std"],
              rnd_skew=mom["skew"], rnd_kurtosis=mom["kurtosis_excess"],
              integral=integral, grid_size=grid_size)

    return RNDFit(
        as_of_ts=as_of_ts, spot=spot, T=clean.T, r=r, q=q,
        smile=svi, smile_r_squared=r_squared, bl_inner=bl_inner,
        strikes=K_full, density=q_full, cdf=cdf_full,
        left_tail=left_tail, right_tail=right_tail,
        n_input_strikes=clean.n_input,
        n_clean_strikes=len(clean.strikes),
        rnd_mean=mom["mean"], rnd_std=mom["std"],
        rnd_skew=mom["skew"], rnd_kurtosis_excess=mom["kurtosis_excess"],
        integral=integral, grid_size=grid_size,
    )
