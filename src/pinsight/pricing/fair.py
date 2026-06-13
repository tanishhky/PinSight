"""Numerically integrate option payoffs over the fitted RND."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..rnd.density import RNDFit


@dataclass(frozen=True)
class ContractSpec:
    """Minimal contract identity for pricing."""
    kind: str          # 'call' | 'put'
    strike: float
    market_premium: float   # observed mid (or ask, depending on use case)
    # Optional metadata (passed through so the paper trader can identify the
    # contract later without re-keying).
    ticker: Optional[str] = None
    expiry: Optional[str] = None


@dataclass(frozen=True)
class FairValue:
    spec: ContractSpec
    fair_premium: float       # PV of E_q[payoff]
    expected_payoff: float    # E_q[payoff] before PV
    prob_itm: float
    edge_ratio: float         # market / fair; <1 ⇒ underpriced (BUY signal)
    expected_pnl: float       # = expected_payoff (PV-adjusted) - market
    stdev_payoff: float       # spread of payoff under q


def price_contract(rnd: RNDFit, spec: ContractSpec) -> FairValue:
    """Integrate the payoff against the RND grid.

    Trapezoidal rule on the RND's strike grid. Numerical precision is
    dominated by the grid spacing (~$0.50 typical), which is fine for
    USD-cent-level fair premia.
    """
    K_grid = rnd.strikes
    q = rnd.density

    if spec.kind == "call":
        payoff = np.maximum(K_grid - spec.strike, 0.0)
        prob_itm = float(rnd.prob_above(spec.strike))
    elif spec.kind == "put":
        payoff = np.maximum(spec.strike - K_grid, 0.0)
        prob_itm = float(rnd.prob_below(spec.strike))
    else:
        raise ValueError(f"kind must be 'call' or 'put', got {spec.kind!r}")

    trapezoid = getattr(np, "trapezoid", np.trapz)
    expected_payoff = float(trapezoid(payoff * q, K_grid))
    second_moment = float(trapezoid(payoff ** 2 * q, K_grid))
    var = max(second_moment - expected_payoff ** 2, 0.0)
    std = float(np.sqrt(var))

    fair = math.exp(-rnd.r * rnd.T) * expected_payoff
    edge = (spec.market_premium / fair) if fair > 1e-8 else float("inf")
    expected_pnl = fair - spec.market_premium

    return FairValue(
        spec=spec,
        fair_premium=fair,
        expected_payoff=expected_payoff,
        prob_itm=prob_itm,
        edge_ratio=edge,
        expected_pnl=expected_pnl,
        stdev_payoff=std,
    )
