"""Intrinsic-value floor guard (2026-06-22).

You cannot buy an option below its intrinsic value — that is an instant
arbitrage that does not exist in a real market. A deep-ITM strike quoted
below intrinsic is stale/crossed/garbage data, not edge. The edge_buyer
must REFUSE such a quote; otherwise it "buys free money" and books phantom
profit at settlement (the source of the implausible +250k paper return).

This test builds a chain with exactly one poisoned strike: a deep-ITM put
quoted far below its intrinsic value. Pre-fix, the agent bought it (huge
apparent edge). Post-fix, it must skip it and open nothing.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from pinsight import paper


def _chain_with_one_subintrinsic_put(tmp_path: Path) -> pd.DataFrame:
    """Spot 600. All strikes priced fairly (intrinsic + time value) EXCEPT
    one deep-ITM put (strike 640) quoted at ~5.0 when its intrinsic is 40.
    """
    spot = 600.0
    snap = "2026-06-01T14:00:00+00:00"
    rows = []
    for K in range(560, 641):
        for kind in ("call", "put"):
            intrinsic = max(spot - K, 0.0) if kind == "call" else max(K - spot, 0.0)
            tv = max(0.20, 2.0 * max(0.05, 1.0 - abs(K - spot) / 20))
            mid = intrinsic + tv
            # Poison the one deep-ITM put: quote it far BELOW intrinsic.
            if kind == "put" and K == 640:
                mid = 5.0  # intrinsic is 40 — this is impossible/garbage data
            bid = max(0.01, mid - 0.05)
            ask = mid + 0.05
            rows.append({
                "ticker": f"SPY{K}{kind[0].upper()}",
                "underlying": "SPY",
                "contract_type": kind,
                "strike": float(K),
                "expiry": "2026-06-01",
                "bid": bid, "ask": ask, "mid": mid,
                "volume": 1000, "open_interest": 5000,
                "iv": 0.15, "in_the_money": False,
                "underlying_price": float(spot),
                "quote_ts": snap, "_snapshot_ts": snap,
            })
    return pd.DataFrame(rows)


def _poisoned_put_bought(tmp_path: Path) -> bool:
    t_path = tmp_path / "paper_trades.parquet"
    if not t_path.exists():
        return False
    t = pd.read_parquet(t_path)
    return not t[(t["kind"] == "put") & (t["strike"] == 640.0)].empty


def test_subintrinsic_buy_happens_without_guard(tmp_path: Path, monkeypatch):
    """Sanity / non-vacuous control: with the intrinsic guard DISABLED
    (monkeypatched to 0), the agent DOES buy the poisoned below-intrinsic
    put — proving the fixture generates the buy and the bug was real."""
    monkeypatch.setattr(paper, "_intrinsic", lambda kind, strike, spot: 0.0)
    chain = _chain_with_one_subintrinsic_put(tmp_path)
    paper.tick(tmp_path, chain,
               as_of_ts="2026-06-01T15:00:00+00:00", expiry_iso="2026-06-01")
    assert _poisoned_put_bought(tmp_path), (
        "control failed: poisoned put not bought even with guard disabled — "
        "fixture does not exercise the guard")


def test_subintrinsic_quote_is_refused(tmp_path: Path):
    """With the guard active, the same below-intrinsic put must be refused,
    and NO opened position may have paid below its intrinsic value."""
    chain = _chain_with_one_subintrinsic_put(tmp_path)
    paper.tick(tmp_path, chain,
               as_of_ts="2026-06-01T15:00:00+00:00", expiry_iso="2026-06-01")
    assert not _poisoned_put_bought(tmp_path), (
        "agent bought a put quoted below intrinsic — floor guard failed")
    t_path = tmp_path / "paper_trades.parquet"
    if t_path.exists():
        for _, p in pd.read_parquet(t_path).iterrows():
            spot = float(p["spot_at_entry"]); K = float(p["strike"])
            intr = max(spot - K, 0.0) if p["kind"] == "call" else max(K - spot, 0.0)
            assert float(p["entry_fill_price"]) >= intr - 1e-6, (
                f"opened below intrinsic: fill={p['entry_fill_price']} intr={intr}")
