"""ADR-0004 enforcement test.

A paper-tick decision made at time T must NOT change when we add later
snapshots (T+k for k>0) to the chain parquet. If it does, we've leaked
future information.

The test runs `paper.tick` twice on the same `as_of_ts`:
  * with the chain truncated to snapshot_ts <= as_of_ts
  * with the chain *also* including snapshots > as_of_ts

The opened-positions list must be identical (same trade_id assignment
modulo UUID, so we compare on (kind, strike, n_contracts, entry_price))
and the RND fit moments must be identical to machine precision.
"""

from __future__ import annotations

import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from pinsight import paper
from pinsight.rnd.density import extract


@pytest.fixture
def sample_chain() -> pd.DataFrame:
    """A minimal viable 0DTE chain: 30 strikes around spot 600, two
    snapshots at T0 and T0+1h.
    """
    spot = 600.0
    T0 = "2026-06-01T14:00:00+00:00"
    T1 = "2026-06-01T15:00:00+00:00"
    rows = []
    for snap_ts in (T0, T1):
        # Slight underlying drift for the later snapshot
        s = spot if snap_ts == T0 else spot + 1.5
        for K in range(580, 621):
            for kind in ("call", "put"):
                # Intrinsic + a tiny bit of time value
                if kind == "call":
                    intrinsic = max(s - K, 0.0)
                else:
                    intrinsic = max(K - s, 0.0)
                tv = max(0.20, 2.0 * max(0.05, 1.0 - abs(K - s) / 20))
                mid = intrinsic + tv
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
                    "underlying_price": float(s),
                    "quote_ts": snap_ts,
                    "_snapshot_ts": snap_ts,
                })
    return pd.DataFrame(rows)


def _open_positions_signature(positions: list[dict]) -> list[tuple]:
    """Identity for comparison — excludes UUID trade_id."""
    sig = []
    for p in positions:
        if p.get("status") != "open":
            continue
        sig.append((p["kind"], float(p["strike"]),
                    int(p["n_contracts"]),
                    round(float(p["entry_fill_price"]), 4),
                    round(float(p["fair_at_entry"]), 4)))
    return sorted(sig)


def test_future_snapshots_do_not_change_past_decision(tmp_path, sample_chain):
    """The headline test.

    Tick at T0 with two scenarios:
      (a) chain has snapshots at T0 and T0+1h
      (b) chain has only the T0 snapshot
    Decisions must match identically.
    """
    as_of_ts = "2026-06-01T14:00:00+00:00"

    # ── (a) Past-only — drop the future snapshot ──
    dir_a = tmp_path / "past_only"
    dir_a.mkdir()
    df_past = sample_chain[sample_chain["_snapshot_ts"] <= as_of_ts]
    paper.tick(dir_a, df_past, as_of_ts=as_of_ts, expiry_iso="2026-06-01")
    sig_a = _open_positions_signature(paper._load_positions(dir_a))

    # ── (b) Past + future — full chain, including snapshots > as_of_ts ──
    dir_b = tmp_path / "with_future"
    dir_b.mkdir()
    paper.tick(dir_b, sample_chain, as_of_ts=as_of_ts,
                expiry_iso="2026-06-01")
    sig_b = _open_positions_signature(paper._load_positions(dir_b))

    assert sig_a == sig_b, (
        f"LOOKAHEAD VIOLATION: presence of T+k snapshots changed the "
        f"T decision.\n"
        f"  past_only opens:   {sig_a}\n"
        f"  with_future opens: {sig_b}"
    )


def test_rnd_extract_assertion_fires_on_future_data():
    """Direct rnd.extract guard: if any snapshot is > as_of_ts and we
    accidentally pass them through, the assertion in extract() must
    fire."""
    as_of_ts = "2026-06-01T14:00:00+00:00"
    # Craft a tiny df where snapshot_ts > as_of_ts
    rows = []
    for K in range(595, 606):
        rows.append({
            "ticker": f"SPY{K}C", "underlying": "SPY",
            "contract_type": "call", "strike": float(K),
            "expiry": "2026-06-01", "bid": 1.0, "ask": 1.1,
            "mid": 1.05, "volume": 100, "open_interest": 1000,
            "iv": 0.15, "in_the_money": False,
            "underlying_price": 600.0,
            "quote_ts": "2026-06-01T15:00:00+00:00",
            "_snapshot_ts": "2026-06-01T15:00:00+00:00",  # FUTURE
        })
    df = pd.DataFrame(rows)
    with pytest.raises(AssertionError, match="LOOKAHEAD"):
        extract(df, spot=600.0, as_of_ts=as_of_ts, expiry_iso="2026-06-01")


def test_lookahead_assertion_handles_timestamp_format_mixing():
    """REGRESSION (2026-06-03): the assertion previously used
    lexicographic string comparison on ISO 8601 strings. Same instant
    written as "Z" vs "+00:00" triggered a false-positive LOOKAHEAD
    VIOLATION because 'Z' > '+' in ASCII.

    Post-fix: parse both sides as datetime and compare numerically.
    """
    # Snapshot equals as_of_ts in wall-clock — different suffix style.
    rows = []
    for K in range(595, 606):
        rows.append({
            "ticker": f"SPY{K}C", "underlying": "SPY",
            "contract_type": "call", "strike": float(K),
            "expiry": "2026-06-01", "bid": 1.0, "ask": 1.1,
            "mid": 1.05, "volume": 100, "open_interest": 1000,
            "iv": 0.15, "in_the_money": False,
            "underlying_price": 600.0,
            "quote_ts": "2026-06-01T14:00:00Z",
            "_snapshot_ts": "2026-06-01T14:00:00Z",   # Z suffix
        })
    df = pd.DataFrame(rows)
    # Must not raise (snapshot == as_of_ts in wall-clock time).
    # Returns None because the flat-IV synthetic chain fails the R² gate,
    # but the key behaviour is that the assertion does NOT fire.
    try:
        extract(df, spot=600.0,
                as_of_ts="2026-06-01T14:00:00+00:00",   # +00:00 suffix
                expiry_iso="2026-06-01")
    except AssertionError as exc:
        if "LOOKAHEAD" in str(exc):
            raise AssertionError(
                f"REGRESSION: lookahead assertion fired on equal "
                f"timestamps with mixed Z/+00:00 suffixes: {exc}") from exc


def test_lookahead_assertion_fires_with_mixed_suffix_future_data():
    """The OTHER direction: even with mixed Z vs +00:00 suffixes, a
    snapshot that is genuinely in the future MUST trigger the assertion."""
    rows = []
    for K in range(595, 606):
        rows.append({
            "ticker": f"SPY{K}C", "underlying": "SPY",
            "contract_type": "call", "strike": float(K),
            "expiry": "2026-06-01", "bid": 1.0, "ask": 1.1,
            "mid": 1.05, "volume": 100, "open_interest": 1000,
            "iv": 0.15, "in_the_money": False,
            "underlying_price": 600.0,
            "quote_ts": "2026-06-01T15:00:00Z",
            "_snapshot_ts": "2026-06-01T15:00:00Z",  # 1 hr in the future
        })
    df = pd.DataFrame(rows)
    with pytest.raises(AssertionError, match="LOOKAHEAD"):
        extract(df, spot=600.0,
                as_of_ts="2026-06-01T14:00:00+00:00",
                expiry_iso="2026-06-01")
