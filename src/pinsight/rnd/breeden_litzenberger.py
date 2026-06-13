"""Breeden-Litzenberger: risk-neutral density from option prices.

The identity (Breeden & Litzenberger 1978):

    q(K) = e^(r·T) · ∂²C/∂K²

where C(K) is the call price as a function of strike at fixed expiry T,
and q(K) is the risk-neutral density of the underlying at T evaluated at S=K.

Inputs come from `smile.call_prices` on a fine strike grid; we just take
central differences and renormalise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BLDensity:
    """Discretised risk-neutral density on a strike grid."""
    strikes: np.ndarray         # grid points (sorted)
    density: np.ndarray         # q(K) at each grid point, non-negative
    cdf: np.ndarray             # cumulative distribution (0 → 1)
    raw_negative_mass: float    # mass that was clipped (diagnostic)
    raw_total_mass: float       # ∫q before normalisation


def density_from_calls(strikes: np.ndarray, calls: np.ndarray,
                       *, r: float, T: float) -> BLDensity:
    """Compute q(K) via second-difference of C(K) on a uniform grid.

    Requires `strikes` uniformly spaced; the caller (orchestrator) builds
    that grid. We return non-negative q (clip + renormalise) and the
    diagnostic mass that was thrown away — large clipping is the signal
    that the smile is butterfly-arbitrageable on that interval.
    """
    if len(strikes) < 5:
        raise ValueError(f"need >= 5 grid points, got {len(strikes)}")
    # Verify uniform spacing.
    dK = np.diff(strikes)
    if not np.allclose(dK, dK[0], rtol=1e-6):
        raise ValueError("strikes must be uniformly spaced")
    h = float(dK[0])

    # Second central difference. Endpoints get one-sided differences.
    d2C = np.zeros_like(calls)
    d2C[1:-1] = (calls[2:] - 2 * calls[1:-1] + calls[:-2]) / (h * h)
    d2C[0] = (calls[2] - 2 * calls[1] + calls[0]) / (h * h)
    d2C[-1] = (calls[-1] - 2 * calls[-2] + calls[-3]) / (h * h)

    trapezoid = getattr(np, "trapezoid", np.trapz)
    q_raw = np.exp(r * T) * d2C
    raw_total = float(trapezoid(q_raw, strikes))
    raw_neg = float(trapezoid(np.maximum(-q_raw, 0.0), strikes))

    # Clip negatives (Carr-Madan style) and renormalise.
    q = np.maximum(q_raw, 0.0)
    total = float(trapezoid(q, strikes))
    if total > 0:
        q = q / total
    cdf = np.concatenate([[0.0], np.cumsum((q[:-1] + q[1:]) / 2.0 * h)])
    if cdf[-1] > 0:
        cdf = cdf / cdf[-1]

    return BLDensity(strikes=strikes, density=q, cdf=cdf,
                      raw_negative_mass=raw_neg,
                      raw_total_mass=raw_total)


def moments(d: BLDensity) -> dict[str, float]:
    """First four moments of the density (for diagnostic / comparison)."""
    trapezoid = getattr(np, "trapezoid", np.trapz)
    K = d.strikes
    q = d.density
    mu = float(trapezoid(K * q, K))
    var = float(trapezoid((K - mu) ** 2 * q, K))
    std = float(np.sqrt(max(var, 1e-12)))
    skew = float(trapezoid(((K - mu) / std) ** 3 * q, K))
    kurt = float(trapezoid(((K - mu) / std) ** 4 * q, K)) - 3.0
    return {"mean": mu, "std": std, "skew": skew, "kurtosis_excess": kurt}
