"""Evaluate flow-detector flags against expiry outcomes.

Given a previously-stored chain snapshot:
  1. Re-identify the informed-flow candidates that were flagged.
  2. Determine the underlying's closing price on the expiry day.
  3. Score each candidate: did it end ITM in the direction implied by the flag?
  4. Report hit rate, P&L on a unit-bet basis, log every detail.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from . import obs


def _candidates(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same filter as inspect_chain (Chesney-Crameri-Mancini proxy)."""
    df = df.copy()
    df["v_over_oi"] = df["volume"] / df["open_interest"].replace(0, np.nan)
    return df[(df["v_over_oi"] >= 1.0) & (df["volume"] >= 1000)].copy()


def _final_close_price(underlying: str, expiry: date) -> Optional[float]:
    """Pull the underlying's official close price for the expiry day."""
    with obs.timed("api", "yahoo.expiry_close",
                   underlying=underlying, expiry=str(expiry)) as t:
        # +1 day to make sure the close-of-expiry-day bar is included.
        # +2 days to ensure the expiry-day bar is included (yfinance end
        # is exclusive). Bug 10 fix: previous code used `int and date`
        # which short-circuits to the date because the int is truthy —
        # an unintentional no-op left over from an earlier refactor.
        end_date = date.fromordinal(expiry.toordinal() + 2)
        hist = yf.Ticker(underlying.upper()).history(
            start=expiry.isoformat(),
            end=end_date.isoformat(),
            interval="1d",
            auto_adjust=False,
        )
        if hist.empty:
            t.add(found=False)
            return None
        # Match by date (history index is tz-aware Timestamp)
        for idx, row in hist.iterrows():
            if pd.Timestamp(idx).date() == expiry:
                t.add(found=True, close=float(row["Close"]))
                return float(row["Close"])
        # Fallback: take the first row anyway (expiry may not be a trading day)
        t.add(found=False, fallback_close=float(hist["Close"].iloc[0]))
        return float(hist["Close"].iloc[0])


def eval_flags(data_dir: Path, underlying: str, expiry: date,
               final_price: Optional[float] = None) -> dict:
    """Score the flagged contracts for an expiry against actual outcome."""
    path = data_dir / "chains" / underlying.upper() / f"{expiry.isoformat()}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No chain at {path}")

    df = pd.read_parquet(path)

    # If multiple snapshots, use the FIRST one captured — that's when we'd
    # have decided to act on the flag. Compare via parsed datetime to
    # survive mixed Z/+00:00 timestamp formats. (Bug 11 fix, 2026-06-05.)
    if "_snapshot_ts" in df.columns and not df.empty:
        ts_parsed = pd.to_datetime(df["_snapshot_ts"], utc=True, errors="coerce")
        if ts_parsed.notna().any():
            first_idx = ts_parsed.idxmin()
            first_snapshot = df.loc[first_idx, "_snapshot_ts"]
            df = df[df["_snapshot_ts"] == first_snapshot]

    cands = _candidates(df)
    if cands.empty:
        obs.event(channel="fit", kind="eval.no_candidates", level="WARNING",
                  underlying=underlying, expiry=str(expiry))
        return {"candidates": 0}

    # Determine final close price
    if final_price is None:
        final_price = _final_close_price(underlying, expiry)
    if final_price is None:
        obs.event(channel="fit", kind="eval.no_close_price", level="ERROR",
                  underlying=underlying, expiry=str(expiry))
        return {"candidates": len(cands), "final_price": None}

    cands["final_price"] = final_price
    cands["itm_at_expiry"] = np.where(
        cands["contract_type"] == "call",
        cands["final_price"] > cands["strike"],
        cands["final_price"] < cands["strike"],
    )

    # Intrinsic value at expiry (per share; option is *100 shares)
    cands["intrinsic"] = np.where(
        cands["contract_type"] == "call",
        np.maximum(cands["final_price"] - cands["strike"], 0),
        np.maximum(cands["strike"] - cands["final_price"], 0),
    )

    # If the flag implied buying the option at mid, P&L = intrinsic - mid
    cands["mid_used"] = cands["mid"].fillna(cands["last_price"])
    cands["unit_pnl"] = cands["intrinsic"] - cands["mid_used"]

    # Per-contract P&L (one contract = 100 shares of payoff)
    cands["contract_pnl"] = cands["unit_pnl"] * 100

    # Aggregate
    n = len(cands)
    n_itm = int(cands["itm_at_expiry"].sum())
    hit_rate = n_itm / n if n else None
    total_pnl = float(cands["contract_pnl"].sum())
    avg_pnl_per_flag = float(cands["contract_pnl"].mean())
    flag_winners = cands[cands["unit_pnl"] > 0]

    summary = {
        "underlying": underlying,
        "expiry": expiry.isoformat(),
        "final_close": round(final_price, 2),
        "candidates": n,
        "ended_itm": n_itm,
        "hit_rate": round(hit_rate, 3) if hit_rate is not None else None,
        "total_unit_pnl": round(total_pnl, 2),
        "avg_pnl_per_flag": round(avg_pnl_per_flag, 2),
        "winners": int((cands["unit_pnl"] > 0).sum()),
        "losers": int((cands["unit_pnl"] <= 0).sum()),
    }

    obs.event(channel="fit", kind="eval.summary", level="INFO", **summary)

    # Rich-printed details
    from rich.console import Console
    from rich.table import Table
    con = Console()

    con.print(f"\n[bold]Flag evaluation · {underlying} expiry {expiry}[/bold]")
    con.print(f"Final close: [cyan]${summary['final_close']}[/cyan]   "
              f"Candidates: {summary['candidates']}   "
              f"Ended ITM: [green]{summary['ended_itm']}[/green]   "
              f"Hit rate: [magenta]{summary['hit_rate']:.1%}[/magenta]" if summary['hit_rate'] is not None else "")
    con.print(f"Total P&L per contract: [yellow]${summary['total_unit_pnl']}[/yellow]   "
              f"Avg per flag: [yellow]${summary['avg_pnl_per_flag']}[/yellow]   "
              f"Winners: {summary['winners']} · Losers: {summary['losers']}")

    t = Table(title="Flag-by-flag outcome", show_lines=False, header_style="bold")
    for col in ["type", "strike", "vol/OI", "mid", "intrinsic", "P&L/share", "ITM?"]:
        t.add_column(col)
    cands_sorted = cands.sort_values("contract_pnl", ascending=False)
    for _, r in cands_sorted.iterrows():
        itm_mark = "[green]✓[/green]" if r["itm_at_expiry"] else "[red]✗[/red]"
        pnl_color = "green" if r["unit_pnl"] > 0 else "red"
        t.add_row(
            r["contract_type"],
            f"{r['strike']:.0f}",
            f"{r['v_over_oi']:.1f}",
            f"{r['mid_used']:.2f}" if pd.notna(r['mid_used']) else "—",
            f"{r['intrinsic']:.2f}",
            f"[{pnl_color}]{r['unit_pnl']:+.2f}[/{pnl_color}]",
            itm_mark,
        )
    con.print(t)

    return summary
