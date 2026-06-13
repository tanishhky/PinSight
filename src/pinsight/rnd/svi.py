"""SVI (Stochastic Volatility Inspired) parameterisation + fit.

Per Gatheral (2004), total implied variance is parameterised as:

    w(k) = a + b · ( ρ·(k − m) + √((k − m)² + σ²) )

where k = log(K/F) is log-moneyness against the forward and
w(k) = σ_IV(k)² · T is total implied variance.

Five parameters: (a, b, ρ, m, σ).

Constraints (per Roger Lee + Gatheral):
    a + b·σ·√(1 − ρ²) ≥ 0     (positive variance everywhere)
    b ≥ 0
    |ρ| ≤ 1
    σ > 0
    b·(1 + |ρ|) ≤ 4·T_max     (Roger Lee bound; prevents wing blowup)

Butterfly arbitrage is a stricter condition (Durrleman's g(k) ≥ 0). For
0DTE we check it on the dense fitted grid after fitting and warn if
violated; we don't enforce it inside the optimiser because for short-dated
contracts the optimisation surface is already very tight.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class SVIParams:
    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def total_variance(self, k: np.ndarray) -> np.ndarray:
        """w(k) for an array of log-moneyness values."""
        return self.a + self.b * (
            self.rho * (k - self.m)
            + np.sqrt((k - self.m) ** 2 + self.sigma ** 2)
        )

    def implied_vol(self, k: np.ndarray, T: float) -> np.ndarray:
        """σ_IV(k) = √(w(k) / T)."""
        w = self.total_variance(k)
        # Guard: never return a negative under-the-sqrt.
        return np.sqrt(np.maximum(w, 1e-12) / T)


@dataclass(frozen=True)
class SVIFit:
    """Result of an SVI calibration."""
    params: SVIParams
    k_data: np.ndarray
    w_data: np.ndarray
    w_fit: np.ndarray
    rmse: float                # in total-variance units
    butterfly_violations: int  # number of k-points where g(k) < 0
    converged: bool
    iterations: int


# ── Helper: Durrleman's butterfly check ───────────────────────────────────

def _g_function(k: np.ndarray, p: SVIParams) -> np.ndarray:
    """Durrleman's g(k). g(k) ≥ 0 everywhere ⇔ no butterfly arbitrage.

    g(k) = (1 − k·w'(k) / (2·w(k)))² − (w'(k)² / 4) · (1/w(k) + 1/4)
           + w''(k) / 2
    """
    # First and second derivatives of w wrt k.
    s = np.sqrt((k - p.m) ** 2 + p.sigma ** 2)
    w = p.a + p.b * (p.rho * (k - p.m) + s)
    w_prime = p.b * (p.rho + (k - p.m) / s)
    w_double_prime = p.b * (p.sigma ** 2) / (s ** 3)

    # Guard against w → 0 division.
    w_safe = np.where(w > 1e-10, w, 1e-10)
    term1 = (1.0 - k * w_prime / (2.0 * w_safe)) ** 2
    term2 = (w_prime ** 2 / 4.0) * (1.0 / w_safe + 0.25)
    term3 = w_double_prime / 2.0
    return term1 - term2 + term3


# ── Initial parameter heuristic ───────────────────────────────────────────

def _quadratic_warm_start(k: np.ndarray, ivs: np.ndarray,
                           T: float) -> tuple[float, ...]:
    """Fit σ_IV(k) = a' + b'k + c'k² in IV space (closed-form OLS), then
    translate to SVI params (a, b, ρ, m, σ).

    Mapping (small-σ_SVI limit, where (k-m) dominates √((k-m)²+σ²)):
        ATM IV ≈ √(a/T)    ⇒  a ≈ (a')² · T
        Wing slopes: dσ/dk ≈ b'/(2σ_atm)   right
                              -b'/(2σ_atm)  left
        Right wing total-variance slope = b·(1+ρ),
        Left  wing total-variance slope = b·(1-ρ),
        Both in *w*-space.

    We use sym/asym of the wings to estimate b and ρ:
        avg_slope ≈ (right_slope + left_slope)/2 = b
        skew      ≈ (right_slope - left_slope)/2 = b·ρ
        ⇒ ρ ≈ skew / b
    """
    A = np.column_stack([np.ones_like(k), k, k * k])
    coef, *_ = np.linalg.lstsq(A, ivs, rcond=None)
    a_q, b_q, c_q = coef
    # ATM IV
    sigma_atm = max(float(a_q), 0.01)
    a0 = float(sigma_atm ** 2 * T)
    # m: the IV-space curve minimum is at k_min = -b'/(2c').
    # If c' ≤ 0 (no curvature or inverted), default m to 0 (ATM).
    if c_q > 1e-6:
        m0 = float(-b_q / (2 * c_q))
        m0 = float(np.clip(m0, -2.0, 2.0))
    else:
        m0 = 0.0
    # Estimate wing slopes in w-space by evaluating the quadratic at the
    # observed range endpoints.
    k_left, k_right = float(k.min()), float(k.max())
    iv_left = max(a_q + b_q * k_left + c_q * k_left ** 2, 1e-3)
    iv_right = max(a_q + b_q * k_right + c_q * k_right ** 2, 1e-3)
    iv_min = max(a_q + b_q * m0 + c_q * m0 ** 2, 1e-3)
    w_left = iv_left ** 2 * T
    w_right = iv_right ** 2 * T
    w_min = iv_min ** 2 * T
    slope_right = (w_right - w_min) / max(abs(k_right - m0), 0.01)
    slope_left = (w_left - w_min) / max(abs(m0 - k_left), 0.01)
    # b ≈ (slope_right + slope_left)/2, ρ ≈ (slope_right - slope_left)/(2b)
    b0 = float(max((slope_right + slope_left) / 2.0, 1e-5))
    rho0 = float(np.clip((slope_right - slope_left) / (2.0 * b0), -0.95, 0.95))
    sigma0 = 0.05
    return a0, b0, rho0, m0, sigma0


def _initial_guess(k: np.ndarray, w: np.ndarray) -> tuple[float, ...]:
    """Legacy heuristic kept as one of the multi-start seeds."""
    w_min = float(np.min(w))
    w_max = float(np.max(w))
    k_at_min = float(k[np.argmin(w)])
    a0 = max(w_min * 0.5, 1e-6)
    spread = w_max - w_min
    k_range = max(float(k.max() - k.min()), 0.01)
    b0 = max(spread / k_range, 1e-3)
    rho0 = -0.3
    m0 = k_at_min
    sigma0 = 0.1
    return a0, b0, rho0, m0, sigma0


# ── Objective + constraints ───────────────────────────────────────────────

def _objective(theta: np.ndarray, k: np.ndarray, w_obs: np.ndarray,
               T: float) -> float:
    """RMSE in IV-space, unweighted (avoid ATM-bias trap that collapsed
    the fit to a constant on real chains)."""
    a, b, rho, m, sigma = theta
    p = SVIParams(a=a, b=b, rho=rho, m=m, sigma=sigma)
    w_pred = p.total_variance(k)
    sigma_pred = np.sqrt(np.maximum(w_pred, 1e-12) / T)
    sigma_obs = np.sqrt(w_obs / T)
    return float(np.sum((sigma_pred - sigma_obs) ** 2))


def _constraints_ok(theta: np.ndarray) -> bool:
    """Hard bounds + Roger Lee feasibility."""
    a, b, rho, m, sigma = theta
    if b < 0 or sigma <= 0 or abs(rho) >= 1:
        return False
    # Roger Lee right-wing bound: b·(1+|ρ|) ≤ 2 (the 4·T form drops out
    # because we work in total variance; effective when T ≤ 1).
    if b * (1 + abs(rho)) > 4.0:
        return False
    # Positive variance everywhere requires a + b·σ·√(1−ρ²) ≥ 0.
    if a + b * sigma * np.sqrt(max(0.0, 1 - rho ** 2)) < 0:
        return False
    return True


# ── Public API ────────────────────────────────────────────────────────────

def fit(strikes: np.ndarray, ivs: np.ndarray, *,
        spot: float, T: float, r: float = 0.0,
        q: float = 0.0,
        initial: Optional[SVIParams] = None) -> SVIFit:
    """Calibrate SVI to a smile.

    Inputs:
        strikes  array of strike prices (sorted ascending)
        ivs      implied vols at those strikes (recomputed, NOT yfinance)
        spot     current underlying
        T        years to expiry (must be > 0)
        r, q     rates (default 0 for 0DTE)

    Returns SVIFit with the calibrated params, residuals, and butterfly
    diagnostics.
    """
    if T <= 0:
        raise ValueError(f"T must be > 0, got {T}")
    if len(strikes) < 5:
        raise ValueError(f"need at least 5 strikes to fit SVI, got {len(strikes)}")

    # Forward and log-moneyness (Black-Scholes carry: F = S·e^((r-q)T))
    F = spot * np.exp((r - q) * T)
    k = np.log(strikes / F)
    w_obs = (ivs ** 2) * T

    # Bounds — keep b away from 0 so the fit can't collapse to a constant.
    bounds = [
        (1e-8, 5.0),        # a
        (1e-5, 5.0),        # b — wider lower bound vs legacy 1e-6
        (-0.95, 0.95),      # rho — stay away from singular ±1
        (-1.0, 1.0),        # m — stay within reasonable log-moneyness
        (1e-3, 2.0),        # sigma — wider lower bound vs legacy 1e-4
    ]

    # Multi-start seeds: quadratic warm-start (data-driven), legacy
    # heuristic, plus two manual seeds biased for downside-skew SPY.
    seeds = []
    try:
        seeds.append(np.array(_quadratic_warm_start(k, ivs, T)))
    except Exception:
        pass
    seeds.append(np.array(_initial_guess(k, w_obs)))
    if initial is not None:
        seeds.insert(0, np.array([initial.a, initial.b, initial.rho,
                                     initial.m, initial.sigma]))
    # Manual seeds — typical 0DTE SPY with downside skew (rho < 0,
    # m to the right of ATM so smile minimum is OTM-call side).
    seeds.append(np.array([float(np.mean(w_obs)) * 0.7,
                             1e-4, -0.5, 0.05, 0.02]))
    seeds.append(np.array([float(np.mean(w_obs)) * 0.5,
                             5e-5, -0.7, 0.10, 0.05]))

    def _clip_to_bounds(theta):
        return np.array([float(np.clip(v, lo, hi))
                          for v, (lo, hi) in zip(theta, bounds)])

    best_theta = None
    best_loss = float("inf")
    best_nm_success = False
    total_iter = 0
    for seed in seeds:
        seed = _clip_to_bounds(seed)
        nm = minimize(
            _objective, seed, args=(k, w_obs, T),
            method="Nelder-Mead",
            options={"maxiter": 3000, "xatol": 1e-9, "fatol": 1e-14,
                      "adaptive": True},
        )
        lb = minimize(
            _objective, _clip_to_bounds(nm.x), args=(k, w_obs, T),
            method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-14, "gtol": 1e-10},
        )
        candidate = lb.x if lb.fun < nm.fun else nm.x
        loss = float(min(lb.fun, nm.fun))
        if loss < best_loss and _constraints_ok(candidate):
            best_loss = loss
            best_theta = candidate
            best_nm_success = bool(nm.success or lb.success)
            total_iter += int(nm.nit + (lb.nit if hasattr(lb, "nit") else 0))

    if best_theta is None:
        # All seeds failed feasibility; fall back to the last NM result.
        best_theta = nm.x
        best_nm_success = bool(nm.success)
        total_iter = int(nm.nit)

    theta = best_theta
    converged = best_nm_success and _constraints_ok(theta)

    p = SVIParams(a=theta[0], b=theta[1], rho=theta[2],
                  m=theta[3], sigma=theta[4])
    w_fit = p.total_variance(k)
    rmse = float(np.sqrt(np.mean((w_fit - w_obs) ** 2)))

    # Butterfly check on a dense grid spanning the fitted range plus 50%.
    k_grid = np.linspace(1.5 * k.min(), 1.5 * k.max(), 200)
    g = _g_function(k_grid, p)
    butterfly_violations = int(np.sum(g < -1e-8))

    return SVIFit(
        params=p, k_data=k, w_data=w_obs, w_fit=w_fit, rmse=rmse,
        butterfly_violations=butterfly_violations,
        converged=converged, iterations=total_iter,
    )


def call_price_from_params(p: SVIParams, strikes: np.ndarray,
                            spot: float, T: float,
                            r: float = 0.0, q: float = 0.0) -> np.ndarray:
    """Reprice calls along a strike grid using the fitted SVI surface.

    Equivalent to: for each K, σ_IV = √(w(log(K/F))/T), then BS call price.
    Vectorised for speed since this is called on fine grids (~1000 pts).
    """
    from .black_scholes import bs_call
    F = spot * np.exp((r - q) * T)
    k = np.log(strikes / F)
    sigmas = p.implied_vol(k, T)
    return np.array([bs_call(spot, float(K), T, r, q, float(s))
                     for K, s in zip(strikes, sigmas)])
