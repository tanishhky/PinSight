"""Yahoo Finance adapter (via yfinance).

Free, no API key. Used as the primary chain source while we don't have a
paid Polygon plan. Scrappy: quotes are 15-min delayed, the library can
break when Yahoo changes their JSON schema, and there's no SLA.

Every fetch is logged through `obs` with timing and row counts.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Optional

import pandas as pd
import yfinance as yf

from .. import obs


def _f(v) -> Optional[float]:
    """Coerce to float, returning None for NaN/missing."""
    if v is None or pd.isna(v):
        return None
    f = float(v)
    return f if f != 0 else None


def _i(v) -> int:
    """Coerce to int, treating NaN/missing as 0."""
    if v is None or pd.isna(v):
        return 0
    return int(v)


_COLUMNS = [
    "ticker", "underlying", "contract_type", "strike", "expiry",
    "bid", "ask", "mid", "last_price", "volume", "open_interest",
    "iv", "in_the_money",
    "underlying_price", "quote_ts",
]


def _normalize(df_calls: pd.DataFrame, df_puts: pd.DataFrame,
               underlying: str, underlying_price: float,
               expiry: date) -> pd.DataFrame:
    """Convert yfinance's two-frame output into PinSight's flat schema."""
    rows: list[dict] = []
    for df, side in [(df_calls, "call"), (df_puts, "put")]:
        if df is None or df.empty:
            continue
        for _, r in df.iterrows():
            bid = _f(r.get("bid"))
            ask = _f(r.get("ask"))
            mid = ((bid + ask) / 2) if (bid is not None and ask is not None) else None
            rows.append({
                "ticker": r.get("contractSymbol"),
                "underlying": underlying.upper(),
                "contract_type": side,
                "strike": _f(r.get("strike")) or 0.0,
                "expiry": expiry.isoformat(),
                "bid": bid,
                "ask": ask,
                "mid": mid,
                "last_price": _f(r.get("lastPrice")),
                "volume": _i(r.get("volume")),
                "open_interest": _i(r.get("openInterest")),
                "iv": _f(r.get("impliedVolatility")),
                "in_the_money": bool(r.get("inTheMoney", False)),
                "underlying_price": underlying_price,
                "quote_ts": r.get("lastTradeDate"),
            })
    return pd.DataFrame(rows, columns=_COLUMNS)


def _resolve_underlying_price(ticker: yf.Ticker, calls: pd.DataFrame,
                              puts: pd.DataFrame) -> tuple[float, str]:
    """Best-effort underlying spot price with explicit fallback chain.

    Returns (price, source) so we log which path won.
    """
    # 1. fast_info (in-memory cache; fastest but flaky on weekends)
    try:
        fi = ticker.fast_info
        for key in ("last_price", "lastPrice", "regular_market_price"):
            v = fi.get(key) if hasattr(fi, "get") else getattr(fi, key, None)
            if v is not None and not pd.isna(v) and float(v) > 0:
                return float(v), f"fast_info.{key}"
    except Exception:
        pass

    # 2. recent 1-day history close (very reliable, weekend-safe)
    try:
        hist = ticker.history(period="5d", interval="1d", auto_adjust=False)
        if not hist.empty:
            return float(hist["Close"].iloc[-1]), "history.close"
    except Exception:
        pass

    # 3. put-call parity inference at the strike with smallest |C - P|
    try:
        joined = calls.merge(puts, on="strike", suffixes=("_c", "_p"))
        joined = joined.dropna(subset=["lastPrice_c", "lastPrice_p"])
        if not joined.empty:
            joined["diff"] = (joined["lastPrice_c"] - joined["lastPrice_p"]).abs()
            row = joined.iloc[joined["diff"].argmin()]
            implied = float(row["strike"] + row["lastPrice_c"] - row["lastPrice_p"])
            if implied > 0:
                return implied, "put_call_parity"
    except Exception:
        pass

    return 0.0, "unresolved"


def fetch_chain(underlying: str, expiry: Optional[date] = None) -> pd.DataFrame:
    """Pull one option-chain expiry from Yahoo Finance.

    If `expiry` is None, picks the nearest expiry (closest to today).
    Returns a normalized DataFrame with PinSight's standard schema.
    """
    with obs.timed("api", "yahoo.chain", underlying=underlying,
                   expiry=str(expiry) if expiry else "nearest") as t:
        ticker = yf.Ticker(underlying.upper())

        available = ticker.options
        if not available:
            obs.event(channel="api", kind="yahoo.no_expiries", level="WARNING",
                      underlying=underlying)
            return pd.DataFrame(columns=_COLUMNS)

        if expiry is None:
            chosen = available[0]
        else:
            target = expiry.isoformat()
            if target in available:
                chosen = target
            else:
                obs.event(channel="api", kind="yahoo.expiry_unavailable",
                          level="WARNING", requested=target,
                          available_count=len(available),
                          first_three=available[:3])
                return pd.DataFrame(columns=_COLUMNS)

        chain = ticker.option_chain(chosen)
        info_price, price_source = _resolve_underlying_price(
            ticker, chain.calls, chain.puts)
        obs.event(channel="api", kind="yahoo.underlying_price", level="INFO",
                  underlying=underlying, price=info_price, source=price_source)

        df = _normalize(chain.calls, chain.puts, underlying=underlying,
                        underlying_price=info_price,
                        expiry=date.fromisoformat(chosen))

        obs.bump("api_calls")
        t.add(contracts=len(df), expiry_chosen=chosen,
              underlying_price=float(info_price),
              with_bid=int(df["bid"].notna().sum()),
              with_iv=int(df["iv"].notna().sum()))

    obs.event(channel="api", kind="yahoo.chain.summary", level="INFO",
              underlying=underlying, expiry=chosen, contracts=len(df),
              calls=int((df["contract_type"] == "call").sum()),
              puts=int((df["contract_type"] == "put").sum()),
              total_volume=int(df["volume"].sum()),
              total_oi=int(df["open_interest"].sum()))
    return df


def list_expiries(underlying: str) -> list[str]:
    """Return available expiry dates as ISO strings."""
    with obs.timed("api", "yahoo.expiries", underlying=underlying) as t:
        expiries = list(yf.Ticker(underlying.upper()).options)
        t.add(count=len(expiries))
    obs.bump("api_calls")
    return expiries
