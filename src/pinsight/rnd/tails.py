"""Tail extrapolation for the risk-neutral density.

Production path (post-fix 2026-06-03):
  1. Try Figlewski (2010) GEV tails: fit (ξ, μ, σ) at each anchor using
     PDF match + CDF match + local-slope match (three conditions, three
     parameters).
  2. Fall back to exponential-decay tails if the GEV fit either fails to
     converge OR produces a tail whose PDF at the anchor diverges from
     the BL anchor by more than 20 %.

Method (right tail):
  - Anchor: K_R = strike at the 95th percentile of the BL CDF.
  - Targets at anchor:
      f_target = q(K_R)                       (PDF match)
      F_target = CDF(K_R) = 0.95              (CDF match)
      slope_target = d log q / dK at K_R       (slope match)
  - Solve via Nelder-Mead in 3D.

Method (left tail) — reflection convention:
  - We fit a GEV in the "reflected" strike coordinate K' = 2·K_L − K so
    that left-tail mass below K_L in original coordinates corresponds
    to right-tail mass above K_L in reflected coordinates.
  - Targets:
      f_target = q(K_L)                       (PDF invariant under K→2K_L−K)
      F_target = 1 − CDF(K_L) ≈ 0.95          (mass to the right of K_L in K′)
      slope_target = −d log q / dK at K_L     (slope flips sign under reflection)

GEV functional form:
    f(x; ξ, μ, σ) = (1/σ) · (1 + ξ(x-μ)/σ)^(-1/ξ - 1)
                          · exp(-(1 + ξ(x-μ)/σ)^(-1/ξ))
    F(x; ξ, μ, σ) = exp(-(1 + ξ(x-μ)/σ)^(-1/ξ))
For ξ→0 (Gumbel limit) both become exponentials of -exp(-z).

Reference: Figlewski (2010) "Estimating the Implied Risk-Neutral Density."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
from scipy.optimize import minimize


# ── Dataclasses ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GEVTail:
    """One-sided GEV tail (Figlewski 2010)."""
    side: str
    xi: float
    mu: float
    sigma: float
    anchor_K: float
    anchor_q: float
    fit_loss: float
    converged: bool


@dataclass(frozen=True)
class ExpTail:
    """One-sided exponential-decay fallback tail."""
    side: str
    anchor_K: float
    anchor_q: float
    lam: float
    reason: str = ""


Tail = Union[GEVTail, ExpTail]


# ── GEV PDF/CDF (vectorised) ────────────────────────────────────────────

def _gev_pdf(x: np.ndarray, xi: float, mu: float, sigma: float) -> np.ndarray:
    """GEV PDF. Handles ξ→0 Gumbel limit."""
    z = (x - mu) / sigma
    if abs(xi) < 1e-8:
        return np.exp(-z - np.exp(-z)) / sigma
    t = 1.0 + xi * z
    out = np.zeros_like(x, dtype=float)
    valid = t > 0
    if not np.any(valid):
        return out
    tv = t[valid]
    with np.errstate(over="ignore", invalid="ignore"):
        out[valid] = (tv ** (-1.0 / xi - 1.0)
                       * np.exp(-tv ** (-1.0 / xi))) / sigma
    out = np.where(np.isfinite(out), out, 0.0)
    return out


def _gev_cdf(x: np.ndarray, xi: float, mu: float, sigma: float) -> np.ndarray:
    z = (x - mu) / sigma
    if abs(xi) < 1e-8:
        return np.exp(-np.exp(-z))
    t = np.maximum(1.0 + xi * z, 0.0)
    with np.errstate(over="ignore", invalid="ignore"):
        out = np.exp(-(t ** (-1.0 / xi)))
    return np.where(np.isfinite(out), out, np.where(t > 0, 1.0, 0.0))


# ── Anchor measurements ─────────────────────────────────────────────────

def _local_log_slope(strikes: np.ndarray, q: np.ndarray, idx: int) -> float:
    """d log q / dK at index idx, averaged over a small neighbourhood."""
    if idx <= 1 or idx >= len(q) - 2:
        return 0.0
    i_lo, i_hi = max(0, idx - 2), min(len(q) - 1, idx + 2)
    qs = q[i_lo:i_hi + 1]
    ks = strikes[i_lo:i_hi + 1]
    mask = qs > 0
    if mask.sum() < 2:
        return 0.0
    log_q = np.log(qs[mask])
    return float(np.polyfit(ks[mask], log_q, 1)[0])


# ── GEV fitting (per side) ──────────────────────────────────────────────

_GEV_LOSS_REJECT = 0.5
_PDF_ERR_REJECT = 0.20


def _fit_gev_one_side(side: str, anchor_K: float, anchor_q: float,
                       anchor_cdf: float, slope_log_q: float,
                       K_lo: float, K_hi: float) -> Optional[GEVTail]:
    """Solve 3-target GEV calibration. None if numerically poor."""
    width = max(K_hi - K_lo, 1.0)

    if side == "right":
        target_pdf = anchor_q
        target_cdf = anchor_cdf
        target_slope = slope_log_q
    elif side == "left":
        target_pdf = anchor_q
        target_cdf = 1.0 - anchor_cdf
        target_slope = -slope_log_q
    else:
        raise ValueError(side)

    sigma0 = max(width * 0.10, 1.0)
    if 1e-6 < target_cdf < 1 - 1e-6:
        mu0 = anchor_K - sigma0 * (-np.log(-np.log(target_cdf)))
    else:
        mu0 = anchor_K
    xi0 = 0.1 if side == "left" else 0.05

    def objective(theta):
        xi, mu, sigma = theta
        if sigma <= 1e-4 or abs(xi) > 0.8:
            return 1e6
        f = float(_gev_pdf(np.array([anchor_K]), xi, mu, sigma)[0])
        if f <= 0 or not np.isfinite(f):
            return 1e6
        F = float(_gev_cdf(np.array([anchor_K]), xi, mu, sigma)[0])
        if not np.isfinite(F):
            return 1e6
        h = max(width * 0.001, 0.05)
        f_p = float(_gev_pdf(np.array([anchor_K + h]), xi, mu, sigma)[0])
        f_m = float(_gev_pdf(np.array([anchor_K - h]), xi, mu, sigma)[0])
        if f_p <= 0 or f_m <= 0:
            return 1e6
        slope_pred = (np.log(f_p) - np.log(f_m)) / (2 * h)
        e_pdf = ((f - target_pdf) / max(abs(target_pdf), 1e-12)) ** 2
        cdf_scale = max(target_cdf * (1 - target_cdf), 1e-3)
        e_cdf = ((F - target_cdf) ** 2) / cdf_scale
        slope_scale = max(abs(target_slope), 1e-3)
        e_slope = ((slope_pred - target_slope) / slope_scale) ** 2
        return e_pdf + 0.5 * e_cdf + 0.5 * e_slope

    result = minimize(objective, [xi0, mu0, sigma0],
                       method="Nelder-Mead",
                       options={"maxiter": 2000, "xatol": 1e-6,
                                "fatol": 1e-8, "adaptive": True})
    xi, mu, sigma = result.x
    if not result.success or result.fun > _GEV_LOSS_REJECT:
        return None
    f_final = float(_gev_pdf(np.array([anchor_K]), xi, mu, sigma)[0])
    if f_final <= 0 or abs(f_final - target_pdf) / max(target_pdf, 1e-12) > _PDF_ERR_REJECT:
        return None
    return GEVTail(side=side, xi=float(xi), mu=float(mu), sigma=float(sigma),
                    anchor_K=anchor_K, anchor_q=anchor_q,
                    fit_loss=float(result.fun), converged=True)


# ── Exponential-decay fallback ──────────────────────────────────────────

def _exp_tail_lambda(slope_log_q: float) -> float:
    if not np.isfinite(slope_log_q) or slope_log_q == 0:
        return 5.0
    return float(np.clip(abs(1.0 / slope_log_q), 0.5, 100.0))


# ── Tail PDF evaluator ──────────────────────────────────────────────────

def _tail_pdf_at(K: np.ndarray, tail: Tail) -> np.ndarray:
    """Evaluate the tail PDF at strikes K (must be on the correct side)."""
    if isinstance(tail, GEVTail):
        if tail.side == "right":
            return _gev_pdf(K, tail.xi, tail.mu, tail.sigma)
        # Left tail: reflect K through anchor before evaluating GEV
        K_refl = 2 * tail.anchor_K - K
        return _gev_pdf(K_refl, tail.xi, tail.mu, tail.sigma)
    # ExpTail
    if tail.side == "right":
        return tail.anchor_q * np.exp(-(K - tail.anchor_K) / tail.lam)
    return tail.anchor_q * np.exp(-(tail.anchor_K - K) / tail.lam)


# ── Public entry ────────────────────────────────────────────────────────

def attach_tails(strikes_inner: np.ndarray, q_inner: np.ndarray,
                  cdf_inner: np.ndarray,
                  *, left_quantile: float = 0.05,
                  right_quantile: float = 0.95,
                  extension_factor: float = 3.0,
                  grid_density: int = 200
                  ) -> Tuple[np.ndarray, np.ndarray, Tail, Tail]:
    """Attach GEV (with exp fallback) tails to a BL density and renormalise."""
    K_lo, K_hi = float(strikes_inner.min()), float(strikes_inner.max())
    width = K_hi - K_lo

    idx_L = int(np.argmin(np.abs(cdf_inner - left_quantile)))
    idx_R = int(np.argmin(np.abs(cdf_inner - right_quantile)))
    anchor_K_L = float(strikes_inner[idx_L])
    anchor_K_R = float(strikes_inner[idx_R])
    anchor_q_L = max(float(q_inner[idx_L]), 1e-12)
    anchor_q_R = max(float(q_inner[idx_R]), 1e-12)
    anchor_cdf_L = float(cdf_inner[idx_L])
    anchor_cdf_R = float(cdf_inner[idx_R])
    slope_L = _local_log_slope(strikes_inner, q_inner, idx_L)
    slope_R = _local_log_slope(strikes_inner, q_inner, idx_R)

    left_gev = _fit_gev_one_side(
        "left", anchor_K_L, anchor_q_L, anchor_cdf_L, slope_L, K_lo, K_hi)
    if left_gev is not None:
        left_tail: Tail = left_gev
    else:
        left_tail = ExpTail(side="left", anchor_K=anchor_K_L,
                             anchor_q=anchor_q_L,
                             lam=_exp_tail_lambda(slope_L),
                             reason="gev_fit_failed")

    right_gev = _fit_gev_one_side(
        "right", anchor_K_R, anchor_q_R, anchor_cdf_R, slope_R, K_lo, K_hi)
    if right_gev is not None:
        right_tail: Tail = right_gev
    else:
        right_tail = ExpTail(side="right", anchor_K=anchor_K_R,
                              anchor_q=anchor_q_R,
                              lam=_exp_tail_lambda(slope_R),
                              reason="gev_fit_failed")

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
        q_full[left_mask] = _tail_pdf_at(K_full[left_mask], left_tail)
    right_mask = K_full > anchor_K_R
    if right_mask.any():
        q_full[right_mask] = _tail_pdf_at(K_full[right_mask], right_tail)

    trapezoid = getattr(np, "trapezoid", np.trapz)
    total = float(trapezoid(q_full, K_full))
    if total > 0:
        q_full = q_full / total

    return K_full, q_full, left_tail, right_tail
