"""Chain quality filter + IV recomputation.

The yfinance chain has noisy implied vols (sometimes wildly wrong for OTM
strikes with no real liquidity). We recompute IV ourselves from the mid
price using Brent root-finding, then drop strikes that fail one or more
sanity checks.

Output of `filter_chain` feeds the SVI fit directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from . import black_scholes as bs


@dataclass(frozen=True)
class QualityConfig:
    """Tunable per-symbol filter thresholds."""
    # Tick-aware spread filter: allow up to max_relative_spread for HIGH-PRICED
    # contracts, but always permit `tick_floor_spread` absolute regardless.
    # Effective limit per row: max(tick_floor_spread, max_relative_spread × mid).
    # This lets OTM 0DTE contracts with $0.05/$0.10 quotes pass even though
    # their relative spread is technically 50 %+.
    max_relative_spread: float = 0.30
    tick_floor_spread: float = 0.10     # always allow $0.10 spread
    max_absolute_spread: float = 1.00   # hard upper cap (safety)
    min_mid_price: float = 0.03         # below this, fees + noise dominate
    min_volume: int = 0                 # 0 = no volume filter
    iv_lo: float = 0.02                 # 2 % annualised lower bound
    iv_hi: float = 3.00                 # 300 % upper bound (gamma squeeze ok)
    moneyness_window: float = 0.10      # |K/S - 1| <= 0.10 (±10 %)


@dataclass(frozen=True)
class CleanChain:
    """Result of filter_chain: ready for SVI."""
    spot: float
    T: float                      # years to expiry
    r: float
    q: float
    snapshot_ts: str              # ISO 8601 UTC, the as_of_ts boundary
    strikes: np.ndarray           # call strikes (sorted)
    mids: np.ndarray              # call mids at each strike
    ivs: np.ndarray               # recomputed IVs at each strike
    # Diagnostics:
    n_input: int
    n_after_bid_filter: int
    n_after_spread_filter: int
    n_after_iv_filter: int
    n_after_moneyness_filter: int


def _years_to_expiry(expiry_iso: str, as_of_ts: str) -> float:
    """Annualised time to expiry — assumes T = wall-clock hours / 8760.

    For 0DTE we use calendar time, not trading time, because vol pricing
    is what the market uses. SPY 0DTE typically has T ∈ [0, 8/8760] hours.
    """
    t_now = datetime.fromisoformat(as_of_ts.replace("Z", "+00:00"))
    # yfinance expiry strings are "YYYY-MM-DD" — treat as 16:00 ET (US
    # equity options expire at 4 pm ET) = 20:00 UTC during EDT, 21:00
    # during EST. Approximate as 20:00 UTC; we live with 1-hour DST drift.
    if "T" in expiry_iso:
        t_exp = datetime.fromisoformat(expiry_iso.replace("Z", "+00:00"))
    else:
        t_exp = datetime.fromisoformat(expiry_iso + "T20:00:00+00:00")
    seconds = (t_exp - t_now).total_seconds()
    if seconds <= 0:
        return 0.0
    return seconds / (365.25 * 24 * 3600)


def filter_chain(df: pd.DataFrame, *,
                 spot: float,
                 as_of_ts: str,
                 expiry_iso: str,
                 r: float = 0.0,
                 q: float = 0.0,
                 cfg: Optional[QualityConfig] = None) -> CleanChain:
    """Filter + IV-recompute a raw chain dataframe.

    Expected columns in df: strike, contract_type, bid, ask, mid (optional;
    computed if missing), volume.

    Calls and puts are kept separately, but the SVI fit operates on the
    CALL price surface only — puts are converted to call-equivalents via
    put-call parity:  C = P + S*e^(-qT) - K*e^(-rT).
    This avoids a separate fit and uses put data to extend the strike grid
    on the downside (where put volume is typically higher than call volume).
    """
    if cfg is None:
        cfg = QualityConfig()

    T = _years_to_expiry(expiry_iso, as_of_ts)
    if T <= 0:
        return CleanChain(spot=spot, T=0.0, r=r, q=q, snapshot_ts=as_of_ts,
                          strikes=np.array([]), mids=np.array([]),
                          ivs=np.array([]),
                          n_input=len(df), n_after_bid_filter=0,
                          n_after_spread_filter=0, n_after_iv_filter=0,
                          n_after_moneyness_filter=0)

    n_input = len(df)
    df = df.copy()
    if "mid" not in df.columns:
        df["mid"] = (df["bid"] + df["ask"]) / 2.0

    # ── Bid > 0, mid > min, ask > bid ──
    df = df[(df["bid"] > 0) & (df["mid"] >= cfg.min_mid_price)
            & (df["ask"] > df["bid"])]
    if cfg.min_volume > 0 and "volume" in df.columns:
        df = df[df["volume"] >= cfg.min_volume]
    n_after_bid = len(df)

    # ── Spread filter (tick-aware) ──
    df["abs_spread"] = df["ask"] - df["bid"]
    df["rel_spread"] = df["abs_spread"] / df["mid"]
    effective_limit = np.maximum(cfg.tick_floor_spread,
                                  cfg.max_relative_spread * df["mid"])
    df = df[(df["abs_spread"] <= cfg.max_absolute_spread)
            & (df["abs_spread"] <= effective_limit)]
    n_after_spread = len(df)

    # ── Convert puts to call-equivalent prices via put-call parity ──
    # C_synthetic = P + S - K  (for r=q=0; full formula with discounting otherwise)
    #
    # CRITICAL filter: use only OUT-OF-THE-MONEY contracts on each side.
    # ITM contracts are dominated by intrinsic value, so their mid is mostly
    # intrinsic ± bid-ask noise. Recovering IV from that noise yields
    # garbage smiles (we observed IV=0.8 at deep-ITM puts on real data).
    # Standard practice in RND extraction (Figlewski 2010, Jackwerth 2004).
    import math
    rows = []
    for _, row in df.iterrows():
        strike = float(row["strike"])
        mid = float(row["mid"])
        ctype = row["contract_type"]
        if ctype == "call":
            if strike < spot:  # ITM call — skip; rely on OTM put at this strike
                continue
            call_mid = mid
        else:  # put
            if strike > spot:  # ITM put — skip
                continue
            call_mid = mid + spot * math.exp(-q * T) - strike * math.exp(-r * T)
            if call_mid <= 0:
                # Parity would yield negative call price → arbitrage band
                # violation in the put quote. Skip.
                continue
        rows.append({"strike": strike, "call_mid": call_mid,
                     "volume": row.get("volume", 0),
                     "contract_type": ctype})
    if not rows:
        return CleanChain(spot=spot, T=T, r=r, q=q, snapshot_ts=as_of_ts,
                          strikes=np.array([]), mids=np.array([]),
                          ivs=np.array([]),
                          n_input=n_input, n_after_bid_filter=n_after_bid,
                          n_after_spread_filter=n_after_spread,
                          n_after_iv_filter=0,
                          n_after_moneyness_filter=0)
    syn = pd.DataFrame(rows)
    # When a strike has both call and put quotes, keep the call (closer to
    # the actively-traded side OTM-of-spot would matter; we just dedupe).
    syn = syn.sort_values(["strike", "contract_type"]).drop_duplicates(
        subset="strike", keep="first")

    # ── Recompute IV per strike via Brent ──
    ivs: list[float] = []
    for _, row in syn.iterrows():
        iv = bs.implied_vol(price_observed=float(row["call_mid"]),
                             S=spot, K=float(row["strike"]),
                             T=T, r=r, q=q, kind="call")
        ivs.append(iv if iv is not None else float("nan"))
    syn["iv"] = ivs

    syn = syn.dropna(subset=["iv"])
    syn = syn[(syn["iv"] >= cfg.iv_lo) & (syn["iv"] <= cfg.iv_hi)]
    n_after_iv = len(syn)

    # ── Moneyness window ──
    syn["moneyness"] = syn["strike"] / spot - 1.0
    syn = syn[syn["moneyness"].abs() <= cfg.moneyness_window]
    n_after_moneyness = len(syn)

    syn = syn.sort_values("strike").reset_index(drop=True)

    return CleanChain(
        spot=spot, T=T, r=r, q=q, snapshot_ts=as_of_ts,
        strikes=syn["strike"].to_numpy(),
        mids=syn["call_mid"].to_numpy(),
        ivs=syn["iv"].to_numpy(),
        n_input=n_input,
        n_after_bid_filter=n_after_bid,
        n_after_spread_filter=n_after_spread,
        n_after_iv_filter=n_after_iv,
        n_after_moneyness_filter=n_after_moneyness,
    )
