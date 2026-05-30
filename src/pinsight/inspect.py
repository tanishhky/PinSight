"""Quick descriptive stats on a fetched chain — print to terminal + log.

Use as: python -m pinsight.cli inspect-chain SPY 2026-06-01
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from . import obs


def inspect_chain(data_dir: Path, underlying: str, expiry: date) -> dict:
    """Load a chain snapshot and emit a structured summary."""
    path = data_dir / "chains" / underlying.upper() / f"{expiry.isoformat()}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No chain at {path}")

    df = pd.read_parquet(path)
    if df.empty:
        return {"contracts": 0}

    underlying_price = float(df["underlying_price"].iloc[0])
    calls = df[df["contract_type"] == "call"].copy()
    puts = df[df["contract_type"] == "put"].copy()

    calls["moneyness"] = calls["strike"] / underlying_price - 1
    puts["moneyness"] = puts["strike"] / underlying_price - 1

    # ATM strike: closest call strike to underlying
    atm_call = calls.iloc[(calls["strike"] - underlying_price).abs().argsort()[:1]]
    atm_put = puts.iloc[(puts["strike"] - underlying_price).abs().argsort()[:1]]

    # 25-delta proxy: roughly 1.5% OTM for short-dated SPY
    otm_call_25d = calls[
        (calls["moneyness"] > 0.013) & (calls["moneyness"] < 0.017)
    ]["iv"].mean()
    otm_put_25d = puts[
        (puts["moneyness"] < -0.013) & (puts["moneyness"] > -0.017)
    ]["iv"].mean()

    skew_25d = (otm_put_25d - otm_call_25d) if pd.notna(otm_put_25d) and pd.notna(otm_call_25d) else None

    # Volume / OI concentration
    top5_vol_calls = calls.nlargest(5, "volume")[["strike", "volume", "open_interest", "iv"]]
    top5_vol_puts = puts.nlargest(5, "volume")[["strike", "volume", "open_interest", "iv"]]

    # P/C ratio
    total_call_vol = int(calls["volume"].sum())
    total_put_vol = int(puts["volume"].sum())
    pcr_volume = total_put_vol / total_call_vol if total_call_vol else None

    total_call_oi = int(calls["open_interest"].sum())
    total_put_oi = int(puts["open_interest"].sum())
    pcr_oi = total_put_oi / total_call_oi if total_call_oi else None

    # Volume / OI ratio per contract (Chesney 2015 informed-flow proxy)
    df["v_over_oi"] = df["volume"] / df["open_interest"].replace(0, np.nan)
    informed_candidates = df[
        (df["v_over_oi"] >= 1.0) & (df["volume"] >= 1000)
    ].sort_values("v_over_oi", ascending=False)[
        ["contract_type", "strike", "volume", "open_interest", "v_over_oi", "iv"]
    ].head(10)

    summary = {
        "underlying": underlying,
        "expiry": expiry.isoformat(),
        "dte": (expiry - date.today()).days,
        "underlying_price": round(underlying_price, 2),
        "contracts": len(df),
        "calls": len(calls),
        "puts": len(puts),
        "atm_call_iv": round(float(atm_call["iv"].iloc[0]), 4) if not atm_call.empty and pd.notna(atm_call["iv"].iloc[0]) else None,
        "atm_put_iv": round(float(atm_put["iv"].iloc[0]), 4) if not atm_put.empty and pd.notna(atm_put["iv"].iloc[0]) else None,
        "skew_25d": round(float(skew_25d), 4) if skew_25d is not None else None,
        "total_call_vol": total_call_vol,
        "total_put_vol": total_put_vol,
        "pcr_volume": round(pcr_volume, 3) if pcr_volume else None,
        "pcr_oi": round(pcr_oi, 3) if pcr_oi else None,
        "informed_flow_candidates": len(informed_candidates),
    }

    obs.event(channel="fit", kind="inspect.chain", level="INFO", **summary)

    # Console-friendly print using rich (already imported by obs)
    from rich.console import Console
    from rich.table import Table
    con = Console()

    con.print(f"\n[bold]{underlying} chain · expiry {expiry} · DTE {summary['dte']}[/bold]")
    con.print(f"Underlying: [cyan]${summary['underlying_price']}[/cyan]   "
              f"Contracts: {summary['contracts']} ({summary['calls']}C / {summary['puts']}P)")
    con.print(f"ATM IV: call=[yellow]{summary['atm_call_iv']}[/yellow]  "
              f"put=[yellow]{summary['atm_put_iv']}[/yellow]   "
              f"25Δ skew (P-C): [magenta]{summary['skew_25d']}[/magenta]")
    con.print(f"Volume: {summary['total_call_vol']:,} calls · {summary['total_put_vol']:,} puts   "
              f"P/C: [cyan]{summary['pcr_volume']}[/cyan]   "
              f"OI P/C: [cyan]{summary['pcr_oi']}[/cyan]")

    t = Table(title="Top 5 calls by volume", show_lines=False, header_style="bold")
    for col in ["strike", "volume", "open_interest", "iv"]:
        t.add_column(col)
    for _, r in top5_vol_calls.iterrows():
        t.add_row(f"{r['strike']:.0f}", f"{int(r['volume']):,}",
                  f"{int(r['open_interest']):,}",
                  f"{r['iv']:.3f}" if pd.notna(r['iv']) else "—")
    con.print(t)

    t = Table(title="Top 5 puts by volume", show_lines=False, header_style="bold")
    for col in ["strike", "volume", "open_interest", "iv"]:
        t.add_column(col)
    for _, r in top5_vol_puts.iterrows():
        t.add_row(f"{r['strike']:.0f}", f"{int(r['volume']):,}",
                  f"{int(r['open_interest']):,}",
                  f"{r['iv']:.3f}" if pd.notna(r['iv']) else "—")
    con.print(t)

    if not informed_candidates.empty:
        t = Table(title="Informed-flow candidates (volume/OI ≥ 1, vol ≥ 1000)",
                  show_lines=False, header_style="bold red")
        for col in ["type", "strike", "volume", "OI", "vol/OI", "iv"]:
            t.add_column(col)
        for _, r in informed_candidates.iterrows():
            t.add_row(
                r["contract_type"],
                f"{r['strike']:.0f}",
                f"{int(r['volume']):,}",
                f"{int(r['open_interest']):,}",
                f"{r['v_over_oi']:.2f}",
                f"{r['iv']:.3f}" if pd.notna(r['iv']) else "—",
            )
        con.print(t)
    else:
        con.print("[dim]No informed-flow candidates flagged.[/dim]")

    return summary
