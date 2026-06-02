"""Smile fit — parametric IV surface from a cleaned chain.

For 0DTE option chains (thin strikes, narrow moneyness window) a full SVI
parameterisation is overkill and numerically fragile. We use a quadratic
in log-moneyness:

    σ_IV(k) = a + b·k + c·k²        where k = log(K / F)

Three parameters, fits via linear regression (closed-form, no optimisation).
Captures level (a ≈ ATM IV), skew (b), and curvature (c).

Limitations vs SVI:
  * Does not enforce no-butterfly arbitrage at every k. Post-fit we
    verify monotonicity of d²C/dK² on the dense grid and emit a warning.
  * Extrapolates to ±∞ as a parabola — that is why density.py applies
    Figlewski GEV tails beyond the traded strike range, not the smile
    surface itself.

When we move beyond 0DTE / broader-strike universes, replace this module
with a proper SVI fit (see svi.py stub).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .black_scholes import bs_call


@dataclass(frozen=True)
class SmileParams:
    """σ_IV(k) = a + b·k + c·k². k in log-moneyness."""
    a: float           # ATM IV (at k=0)
    b: float           # skew slope
    c: float           # curvature

    def implied_vol(self, k: np.ndarray) -> np.ndarray:
        sigma = self.a + self.b * k + self.c * k ** 2
        # Floor to a tiny positive so downstream doesn't blow up on
        # extrapolated negatives.
        return np.maximum(sigma, 1e-4)

    def total_variance(self, k: np.ndarray, T: float) -> np.ndarray:
        return self.implied_vol(k) ** 2 * T


@dataclass(frozen=True)
class SmileFit:
    params: SmileParams
    k_data: np.ndarray
    iv_data: np.ndarray
    iv_fit: np.ndarray
    rmse_iv: float
    r_squared: float
    n_strikes: int


def fit(strikes: np.ndarray, ivs: np.ndarray, *,
        spot: float, T: float, r: float = 0.0,
        q: float = 0.0) -> SmileFit:
    """Closed-form quadratic fit in log-moneyness.

    Equally weights all surviving strikes. If the smile is asymmetric
    around ATM (typical: downside skew on SPY), the fit captures it via
    a non-zero b coefficient.
    """
    if T <= 0:
        raise ValueError(f"T must be > 0, got {T}")
    if len(strikes) < 3:
        raise ValueError(f"need at least 3 strikes for quadratic smile, got {len(strikes)}")

    F = spot * np.exp((r - q) * T)
    k = np.log(strikes / F)

    # Design matrix [1, k, k²]; solve normal equations.
    X = np.column_stack([np.ones_like(k), k, k ** 2])
    coeffs, *_ = np.linalg.lstsq(X, ivs, rcond=None)
    a, b, c = coeffs

    p = SmileParams(a=float(a), b=float(b), c=float(c))
    iv_fit = p.implied_vol(k)
    residuals = iv_fit - ivs
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((ivs - ivs.mean()) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")

    return SmileFit(
        params=p, k_data=k, iv_data=ivs, iv_fit=iv_fit,
        rmse_iv=rmse, r_squared=r2, n_strikes=len(strikes),
    )


def call_prices(p: SmileParams, strikes: np.ndarray, *,
                spot: float, T: float,
                r: float = 0.0, q: float = 0.0) -> np.ndarray:
    """Reprice calls along a strike grid using the smile.

    The Breeden-Litzenberger module differentiates this surface twice to
    recover the risk-neutral density.
    """
    F = spot * np.exp((r - q) * T)
    k = np.log(strikes / F)
    sigmas = p.implied_vol(k)
    return np.array([bs_call(spot, float(K), T, r, q, float(s))
                     for K, s in zip(strikes, sigmas)])
