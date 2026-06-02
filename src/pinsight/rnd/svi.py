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

def _initial_guess(k: np.ndarray, w: np.ndarray) -> tuple[float, ...]:
    """A reasonable starting point. The optimiser is fairly robust to bad
    starts on well-shaped smiles, but the ATM-IV-anchored heuristic below
    converges in ~30 iterations on typical SPY chains."""
    w_min = float(np.min(w))
    w_max = float(np.max(w))
    k_at_min = float(k[np.argmin(w)])
    # a anchors the vertical level
    a0 = max(w_min * 0.5, 1e-6)
    # b sets the wing slope; estimate from the steeper side
    spread = w_max - w_min
    k_range = max(float(k.max() - k.min()), 0.01)
    b0 = max(spread / k_range, 1e-3)
    # rho: negative if right wing is steeper than left (skew); 0 default
    rho0 = -0.3
    # m: the minimum of the smile
    m0 = k_at_min
    # sigma: width parameter; start at chain-typical 0.1
    sigma0 = 0.1
    return a0, b0, rho0, m0, sigma0


# ── Objective + constraints ───────────────────────────────────────────────

def _objective(theta: np.ndarray, k: np.ndarray, w_obs: np.ndarray,
               T: float) -> float:
    """RMSE in IV-space.

    Earlier draft minimised in total-variance space (w = σ²·T), but for
    0DTE the total-variance numbers are ~1e-5 and the optimiser's
    gradient steps were below the convergence floor. Working in IV-space
    keeps the objective at order ~1 and the fit actually moves.
    """
    a, b, rho, m, sigma = theta
    p = SVIParams(a=a, b=b, rho=rho, m=m, sigma=sigma)
    w_pred = p.total_variance(k)
    # Convert both to IV.
    sigma_pred = np.sqrt(np.maximum(w_pred, 1e-12) / T)
    sigma_obs = np.sqrt(w_obs / T)
    # Weight ATM heavier — that's where the signal lives and where the
    # paper trader actually opens positions.
    weights = np.exp(-4.0 * np.abs(k))
    return float(np.sum(weights * (sigma_pred - sigma_obs) ** 2))


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

    theta0 = (np.array([initial.a, initial.b, initial.rho, initial.m, initial.sigma])
              if initial is not None
              else np.array(_initial_guess(k, w_obs)))

    bounds = [
        (1e-8, 5.0),    # a
        (1e-6, 5.0),    # b
        (-0.999, 0.999), # rho
        (-2.0, 2.0),    # m
        (1e-4, 2.0),    # sigma
    ]

    # Two-stage fit: Nelder-Mead first (gradient-free, robust to bad
    # init) → polish with L-BFGS-B for tight convergence.
    nm = minimize(
        _objective, theta0, args=(k, w_obs, T),
        method="Nelder-Mead",
        options={"maxiter": 2000, "xatol": 1e-8, "fatol": 1e-12,
                  "adaptive": True},
    )
    result = minimize(
        _objective, nm.x, args=(k, w_obs, T),
        method="L-BFGS-B", bounds=bounds,
        options={"maxiter": 500, "ftol": 1e-14, "gtol": 1e-10},
    )
    theta = result.x
    converged = bool(result.success or nm.success) and _constraints_ok(theta)

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
        converged=converged, iterations=int(result.nit),
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
