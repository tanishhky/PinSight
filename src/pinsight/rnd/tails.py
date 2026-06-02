"""Tail extrapolation for the risk-neutral density.

Beyond the traded strike range, q(K) extracted via Breeden-Litzenberger is
unreliable (no data + smile extrapolation noise). Two approaches:

  v0 — **Exponential-decay tails**: q(K) = q(anchor) · exp(-|K-anchor|/λ),
       where λ matches the LOCAL log-slope of q at the anchor. Simple,
       always positive, smooth, no optimisation pathologies.

  v1 — **Figlewski (2010) GEV tails**: fit Generalised Extreme Value
       distributions to each wing using PDF + CDF + slope matching at
       the anchor. The literature-standard approach. Has subtle
       implementation traps with left-tail reflection — kept stubbed
       below for now (returns exp-decay fall-through).

The active implementation is exp-decay.  TODO #PIN-RND-1: implement GEV
properly with Figlewski's reflection convention.

Method:
  1. From the BL density inside the traded range, identify two anchor
     points α_L and α_R in each tail (Figlewski: α_L at the 5th percentile,
     α_R at the 95th).
  2. Fit a GEV tail in each wing such that:
       (a) the tail PDF matches q at the anchor (continuity),
       (b) the tail CDF matches the BL CDF at the anchor,
       (c) one additional moment condition (we use the local slope of the
           log-density) closes the system.
  3. Stitch: piecewise q = GEV_left for K < α_L, BL q for K ∈ [α_L, α_R],
     GEV_right for K > α_R. Renormalise the whole thing to 1.

GEV PDF:
    f(x; ξ, μ, σ) = (1/σ) · (1 + ξ(x-μ)/σ)^(-1/ξ - 1) · exp(-(1 + ξ(x-μ)/σ)^(-1/ξ))

For ξ > 0  → Fréchet tail (heavy)
For ξ = 0  → Gumbel tail (light)
For ξ < 0  → Weibull tail (bounded)

For SPY equity index the right tail is roughly Gumbel/light-Fréchet
(ξ ∈ [0, 0.2]) and the left tail is heavier (ξ ∈ [0.1, 0.4]) — crash
risk.

Reference: Figlewski (2010) "Estimating the Implied Risk-Neutral Density".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class GEVTail:
    """One-sided GEV tail."""
    side: str          # 'left' or 'right'
    xi: float          # shape
    mu: float          # location
    sigma: float       # scale
    anchor_K: float    # the strike where this tail attaches
    anchor_q: float    # the density at the anchor (continuity target)


def _gev_pdf(x: np.ndarray, xi: float, mu: float, sigma: float) -> np.ndarray:
    """GEV PDF; vectorised, handles ξ=0 Gumbel limit."""
    z = (x - mu) / sigma
    if abs(xi) < 1e-8:
        # Gumbel
        return np.exp(-z - np.exp(-z)) / sigma
    t = 1.0 + xi * z
    # Domain check: 1 + ξz > 0
    mask = t > 0
    out = np.zeros_like(x, dtype=float)
    valid = t[mask]
    out[mask] = (valid ** (-1.0 / xi - 1.0)
                  * np.exp(-valid ** (-1.0 / xi))) / sigma
    return out


def _gev_cdf(x: np.ndarray, xi: float, mu: float, sigma: float) -> np.ndarray:
    z = (x - mu) / sigma
    if abs(xi) < 1e-8:
        return np.exp(-np.exp(-z))
    t = np.maximum(1.0 + xi * z, 0.0)
    return np.exp(-(t ** (-1.0 / xi)))


def _fit_tail(side: str, anchor_K: float, anchor_q: float, anchor_cdf: float,
               q_slope: float, K_lo: float, K_hi: float) -> GEVTail:
    """Solve for (ξ, μ, σ) matching three conditions at the anchor."""
    # Initial guess: Gumbel (ξ=0) with σ such that the PDF roughly matches.
    # For Gumbel at z=0: PDF = e^(-1)/σ, so σ ≈ e^(-1)/q_anchor.
    sigma0 = max(np.exp(-1) / max(anchor_q, 1e-8), 1.0)
    mu0 = anchor_K - (np.log(-np.log(max(anchor_cdf, 1e-6))) * sigma0
                       if side == "right" else -sigma0)
    xi0 = 0.15 if side == "left" else 0.05

    def objective(theta):
        xi, mu, sigma = theta
        if sigma <= 0:
            return 1e6
        # PDF match at anchor
        f = _gev_pdf(np.array([anchor_K]), xi, mu, sigma)[0]
        # CDF match at anchor (for right tail we match 1-CDF since we're
        # in the upper tail above the anchor)
        F = _gev_cdf(np.array([anchor_K]), xi, mu, sigma)[0]
        if side == "right":
            F_target = anchor_cdf
        else:
            F_target = anchor_cdf
        # Local-slope match (log derivative).
        h = (K_hi - K_lo) * 0.001
        f_plus = _gev_pdf(np.array([anchor_K + h]), xi, mu, sigma)[0]
        f_minus = _gev_pdf(np.array([anchor_K - h]), xi, mu, sigma)[0]
        if f > 1e-12 and f_plus > 0 and f_minus > 0:
            slope_pred = (np.log(f_plus) - np.log(f_minus)) / (2 * h)
        else:
            slope_pred = 0.0
        return ((f - anchor_q) ** 2 / max(anchor_q ** 2, 1e-12)
                + (F - F_target) ** 2
                + ((slope_pred - q_slope) / max(abs(q_slope), 1.0)) ** 2)

    bounds = [(-0.5, 0.8), (K_lo - 10 * (K_hi - K_lo), K_hi + 10 * (K_hi - K_lo)),
              (max((K_hi - K_lo) * 0.001, 0.1), (K_hi - K_lo) * 10)]
    result = minimize(objective, [xi0, mu0, sigma0],
                      method="Nelder-Mead",
                      options={"maxiter": 1000, "xatol": 1e-6})

    xi, mu, sigma = result.x
    return GEVTail(side=side, xi=xi, mu=mu, sigma=sigma,
                    anchor_K=anchor_K, anchor_q=anchor_q)


@dataclass(frozen=True)
class ExpTail:
    """One-sided exponential-decay tail. q(K) = anchor_q · exp(-|K-anchor|/lam)."""
    side: str
    anchor_K: float
    anchor_q: float
    lam: float           # decay length scale (in strike units)


def _exp_tail_lambda(anchor_q: float, slope_log_q: float, side: str) -> float:
    """Choose λ so that the exponential's log-slope matches the BL log-slope
    at the anchor.

    For q(K) = anchor_q · exp(-(K - anchor)/λ) on the right wing,
        d/dK ln q = -1/λ
    so λ = -1 / slope_log_q  on the right (slope_log_q < 0 expected).
    On the left we use q(K) = anchor_q · exp(-(anchor - K)/λ), so
        d/dK ln q = +1/λ  →  λ = +1/slope_log_q  on the left (slope > 0).
    Default λ when the anchor slope is ~0: 5% of strike (sensible scale).
    """
    if slope_log_q == 0 or not np.isfinite(slope_log_q):
        return max(anchor_q * 0.0 + 5.0, 1.0)  # ~$5 default scale
    inv = abs(1.0 / slope_log_q)
    return float(np.clip(inv, 0.5, 100.0))


def attach_tails(strikes_inner: np.ndarray, q_inner: np.ndarray,
                  cdf_inner: np.ndarray,
                  *, left_quantile: float = 0.05,
                  right_quantile: float = 0.95,
                  extension_factor: float = 3.0,
                  grid_density: int = 200):
    """Attach exponential-decay tails to a BL density and renormalise.

    Returns:
        K_full         strike grid spanning  K_lo - (ext-1)·width ... K_hi + (ext-1)·width
        q_full         stitched, renormalised density (integrates to 1)
        left_tail      ExpTail diagnostic dataclass
        right_tail     ExpTail diagnostic dataclass
    """
    K_lo, K_hi = float(strikes_inner.min()), float(strikes_inner.max())
    width = K_hi - K_lo

    idx_L = int(np.argmin(np.abs(cdf_inner - left_quantile)))
    idx_R = int(np.argmin(np.abs(cdf_inner - right_quantile)))
    anchor_K_L = float(strikes_inner[idx_L])
    anchor_K_R = float(strikes_inner[idx_R])
    anchor_q_L = max(float(q_inner[idx_L]), 1e-12)
    anchor_q_R = max(float(q_inner[idx_R]), 1e-12)

    def _local_slope(idx: int) -> float:
        if idx <= 0 or idx >= len(q_inner) - 1:
            return 0.0
        if q_inner[idx - 1] <= 0 or q_inner[idx + 1] <= 0:
            return 0.0
        return (np.log(q_inner[idx + 1]) - np.log(q_inner[idx - 1])) / (
            strikes_inner[idx + 1] - strikes_inner[idx - 1])

    slope_L = _local_slope(idx_L)
    slope_R = _local_slope(idx_R)

    lam_L = _exp_tail_lambda(anchor_q_L, slope_L, side="left")
    lam_R = _exp_tail_lambda(anchor_q_R, slope_R, side="right")

    left_tail = ExpTail(side="left", anchor_K=anchor_K_L,
                         anchor_q=anchor_q_L, lam=lam_L)
    right_tail = ExpTail(side="right", anchor_K=anchor_K_R,
                          anchor_q=anchor_q_R, lam=lam_R)

    K_min_ext = max(0.01, K_lo - width * (extension_factor - 1.0))
    K_max_ext = K_hi + width * (extension_factor - 1.0)
    K_full = np.linspace(K_min_ext, K_max_ext, grid_density * 3)

    q_full = np.zeros_like(K_full)
    inner_mask = (K_full >= anchor_K_L) & (K_full <= anchor_K_R)
    if inner_mask.any():
        q_full[inner_mask] = np.interp(K_full[inner_mask],
                                         strikes_inner, q_inner)
    left_mask = K_full < anchor_K_L
    if left_mask.any():
        q_full[left_mask] = anchor_q_L * np.exp(
            -(anchor_K_L - K_full[left_mask]) / lam_L)
    right_mask = K_full > anchor_K_R
    if right_mask.any():
        q_full[right_mask] = anchor_q_R * np.exp(
            -(K_full[right_mask] - anchor_K_R) / lam_R)

    total = float(np.trapz(q_full, K_full))
    if total > 0:
        q_full = q_full / total

    return K_full, q_full, left_tail, right_tail
