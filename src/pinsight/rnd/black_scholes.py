"""Black-Scholes pricing + implied-volatility root finder.

The two primitives every downstream module depends on. Implemented from
scratch (no QuantLib) so the math is auditable and the failure modes are
explicit.

Notation throughout:
    S    spot price of the underlying
    K    strike
    T    time to expiry in YEARS (0DTE => T ≈ hours/8760)
    r    risk-free rate (continuous compounding); 0DTE ⇒ r·T ≪ 1, so
         passing r=0.0 is fine in practice
    q    dividend yield (continuous compounding); SPY ≈ 0.013 long-run,
         but for 0DTE q·T is negligible — pass 0.0 unless you care
    sigma  implied volatility (annualised)
"""

from __future__ import annotations

import math
from typing import Optional


_SQRT_2PI = math.sqrt(2.0 * math.pi)


def _phi(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / _SQRT_2PI


def _Phi(x: float) -> float:
    """Standard normal CDF via erf — numerically stable across the whole
    real line."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_d1(S: float, K: float, T: float, r: float, q: float,
          sigma: float) -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return float("nan")
    return (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))


def bs_d2(d1: float, sigma: float, T: float) -> float:
    return d1 - sigma * math.sqrt(T)


def bs_call(S: float, K: float, T: float, r: float, q: float,
            sigma: float) -> float:
    """Black-Scholes-Merton call price.

    At T=0 the price is the intrinsic value max(S-K, 0). At sigma≤0 the
    price is the discounted forward intrinsic.
    """
    if T <= 0:
        return max(S - K, 0.0)
    if sigma <= 0:
        # Deterministic forward = S*e^((r-q)T); discount to today.
        fwd = S * math.exp((r - q) * T)
        return math.exp(-r * T) * max(fwd - K, 0.0)
    d1 = bs_d1(S, K, T, r, q, sigma)
    d2 = bs_d2(d1, sigma, T)
    return S * math.exp(-q * T) * _Phi(d1) - K * math.exp(-r * T) * _Phi(d2)


def bs_put(S: float, K: float, T: float, r: float, q: float,
           sigma: float) -> float:
    if T <= 0:
        return max(K - S, 0.0)
    if sigma <= 0:
        fwd = S * math.exp((r - q) * T)
        return math.exp(-r * T) * max(K - fwd, 0.0)
    d1 = bs_d1(S, K, T, r, q, sigma)
    d2 = bs_d2(d1, sigma, T)
    return K * math.exp(-r * T) * _Phi(-d2) - S * math.exp(-q * T) * _Phi(-d1)


def bs_price(S: float, K: float, T: float, r: float, q: float,
             sigma: float, *, kind: str) -> float:
    if kind == "call":
        return bs_call(S, K, T, r, q, sigma)
    if kind == "put":
        return bs_put(S, K, T, r, q, sigma)
    raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")


def bs_vega(S: float, K: float, T: float, r: float, q: float,
            sigma: float) -> float:
    """Vega = ∂Price / ∂σ. Used as the derivative in IV Newton steps when
    we want to fall back from Brent."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1 = bs_d1(S, K, T, r, q, sigma)
    return S * math.exp(-q * T) * _phi(d1) * math.sqrt(T)


# ── Implied volatility ─────────────────────────────────────────────────

# Reasonable IV bounds for equity index options. 0DTE SPY can briefly spike
# above 100 % annualised vol during gamma squeezes; cap at 500 % to stay
# numerically sane.
IV_LO = 1e-4
IV_HI = 5.0
IV_TOL = 1e-6
IV_MAX_ITER = 100


def implied_vol(price_observed: float, S: float, K: float, T: float,
                r: float, q: float, *, kind: str) -> Optional[float]:
    """Solve for σ such that BS(σ) = price_observed via Brent's method.

    Returns None if no root in [IV_LO, IV_HI] (price violates arbitrage
    bounds, or sits inside the bid-ask cross-of-zero where IV is undefined).

    Why Brent and not vega-Newton: Brent has guaranteed convergence inside
    a bracketing interval; vega goes to zero for deep ITM/OTM contracts
    and Newton can diverge there. Brent is ~3× slower per call, which is
    fine for our scale (~250 contracts per chain).
    """
    if T <= 0 or S <= 0 or K <= 0 or price_observed < 0:
        return None

    # Arbitrage bounds for the contract.
    if kind == "call":
        intrinsic = max(S * math.exp(-q * T) - K * math.exp(-r * T), 0.0)
        upper = S * math.exp(-q * T)
    else:
        intrinsic = max(K * math.exp(-r * T) - S * math.exp(-q * T), 0.0)
        upper = K * math.exp(-r * T)
    if price_observed < intrinsic - 1e-10 or price_observed > upper + 1e-10:
        return None

    f = lambda sig: bs_price(S, K, T, r, q, sig, kind=kind) - price_observed

    f_lo = f(IV_LO)
    f_hi = f(IV_HI)
    if f_lo * f_hi > 0:
        # No sign change in the bracket — no root. Usually means the
        # observed price equals intrinsic to numerical precision (sigma
        # collapses to 0) or exceeds the BS upper bound.
        return None

    # Brent's method (van Wijngaarden-Dekker-Brent).
    a, b = IV_LO, IV_HI
    fa, fb = f_lo, f_hi
    if abs(fa) < abs(fb):
        a, b = b, a
        fa, fb = fb, fa
    c, fc = a, fa
    mflag = True
    s = b
    fs = fb

    for _ in range(IV_MAX_ITER):
        if abs(fb) < IV_TOL:
            return b
        if fa != fc and fb != fc:
            # Inverse quadratic interpolation
            s = (a * fb * fc / ((fa - fb) * (fa - fc))
                 + b * fa * fc / ((fb - fa) * (fb - fc))
                 + c * fa * fb / ((fc - fa) * (fc - fb)))
        else:
            # Secant
            s = b - fb * (b - a) / (fb - fa)
        cond1 = not ((3 * a + b) / 4 < s < b or b < s < (3 * a + b) / 4)
        cond2 = mflag and abs(s - b) >= abs(b - c) / 2
        cond3 = (not mflag) and abs(s - b) >= abs(c - (a if mflag else b)) / 2
        cond4 = mflag and abs(b - c) < IV_TOL
        cond5 = (not mflag) and abs(c - (a if mflag else b)) < IV_TOL
        if cond1 or cond2 or cond3 or cond4 or cond5:
            s = (a + b) / 2
            mflag = True
        else:
            mflag = False
        fs = f(s)
        c, fc = b, fb
        if fa * fs < 0:
            b, fb = s, fs
        else:
            a, fa = s, fs
        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa
    return b if abs(fb) < IV_TOL * 100 else None
