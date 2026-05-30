"""Normalize Polygon's snapshot payload into a flat DataFrame.

Polygon's snapshot returns a nested structure per contract:
    {
        "details": { "contract_type", "strike_price", "expiration_date", "ticker" },
        "day": { "open", "high", "low", "close", "volume", "vwap", "last_updated" },
        "last_quote": { "bid", "ask", "midpoint", "last_updated" },
        "last_trade": { "price", "size", "exchange", "sip_timestamp" },
        "greeks": { "delta", "gamma", "theta", "vega" },
        "implied_volatility": float,
        "open_interest": int,
        "underlying_asset": { "ticker", "last_updated", "price" },
        "break_even_price": float,
    }

We flatten to one row per contract with stable column names.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .. import obs


_COLUMNS = [
    "ticker", "underlying", "contract_type", "strike", "expiry",
    "bid", "ask", "mid", "last_price", "volume", "open_interest", "vwap",
    "iv", "delta", "gamma", "theta", "vega",
    "underlying_price", "break_even", "quote_ts", "trade_ts",
]


def _nested(d: dict[str, Any] | None, *path: str, default: Any = None) -> Any:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
    return cur if cur is not None else default


def normalize_polygon_chain(payload: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert Polygon's snapshot list into a normalized DataFrame."""
    rows: list[dict[str, Any]] = []
    for c in payload:
        details = c.get("details") or {}
        day = c.get("day") or {}
        last_quote = c.get("last_quote") or {}
        last_trade = c.get("last_trade") or {}
        greeks = c.get("greeks") or {}
        underlying = c.get("underlying_asset") or {}

        rows.append({
            "ticker": details.get("ticker"),
            "underlying": underlying.get("ticker"),
            "contract_type": details.get("contract_type"),
            "strike": details.get("strike_price"),
            "expiry": details.get("expiration_date"),
            "bid": last_quote.get("bid"),
            "ask": last_quote.get("ask"),
            "mid": last_quote.get("midpoint"),
            "last_price": last_trade.get("price"),
            "volume": day.get("volume"),
            "open_interest": c.get("open_interest"),
            "vwap": day.get("vwap"),
            "iv": c.get("implied_volatility"),
            "delta": greeks.get("delta"),
            "gamma": greeks.get("gamma"),
            "theta": greeks.get("theta"),
            "vega": greeks.get("vega"),
            "underlying_price": underlying.get("price"),
            "break_even": c.get("break_even_price"),
            "quote_ts": _nested(c, "last_quote", "last_updated"),
            "trade_ts": _nested(c, "last_trade", "sip_timestamp"),
        })

    df = pd.DataFrame(rows, columns=_COLUMNS)

    # Quality stats for the audit log.
    n = len(df)
    obs.event(channel="persist", kind="chain.normalize", level="INFO",
              contracts=n,
              calls=int((df["contract_type"] == "call").sum()),
              puts=int((df["contract_type"] == "put").sum()),
              with_bid=int(df["bid"].notna().sum()),
              with_iv=int(df["iv"].notna().sum()),
              with_oi=int(df["open_interest"].notna().sum()),
              )
    return df
