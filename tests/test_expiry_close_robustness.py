"""Regression tests for the 2026-06-09 stuck-positions incident.

Two stacked bugs prevented 32 SPY 0DTE positions from closing at expiry:

  Bug 1 (cli.py): `--market-hours-only` silenced the daemon at 20:00 UTC =
                  4 PM ET, which is also when 0DTE settles. The daemon
                  slept through its own settlement instant.

  Bug 2 (paper.py): paper.tick used the CURRENT chain's `expiry_iso` as
                    the close gate, not each position's stored `expiry`.
                    A T-1 position opened against yesterday's expiry was
                    never recognised as expired once today's chain showed
                    up — and even if it had been, paper.tick would have
                    settled it at today's spot, not yesterday's close.

These tests pin both fixes so a future refactor can't silently undo them.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from pinsight import paper, cli


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def _chain_row(K: float, kind: str, spot: float, expiry: str, snap_ts: str) -> dict:
    """One synthetic row of a 0DTE chain — minimal columns required by
    paper.tick and rnd.extract."""
    intrinsic = max(spot - K, 0.0) if kind == "call" else max(K - spot, 0.0)
    tv = max(0.20, 2.0 * max(0.05, 1.0 - abs(K - spot) / 20))
    mid = intrinsic + tv
    return {
        "ticker": f"SPY{int(K)}{kind[0].upper()}",
        "underlying": "SPY",
        "contract_type": kind,
        "strike": float(K),
        "expiry": expiry,
        "bid": max(0.01, mid - 0.05),
        "ask": mid + 0.05,
        "mid": mid,
        "volume": 1000,
        "open_interest": 5000,
        "iv": 0.15,
        "in_the_money": False,
        "underlying_price": float(spot),
        "quote_ts": snap_ts,
        "_snapshot_ts": snap_ts,
    }


def _write_chain(data_dir: Path, expiry: str, spot: float, snap_ts: str) -> pd.DataFrame:
    """Persist a chain parquet at the canonical path and return the DF."""
    rows = [_chain_row(K, kind, spot, expiry, snap_ts)
            for K in range(580, 621)
            for kind in ("call", "put")]
    df = pd.DataFrame(rows)
    out = data_dir / "chains" / "SPY" / f"{expiry}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, compression="snappy", index=False)
    return df


# ----------------------------------------------------------------------
# Bug 2 — per-position expiry close + correct settlement spot
# ----------------------------------------------------------------------

def test_paper_tick_closes_prior_expiry_at_its_own_settlement_spot(tmp_path):
    """Open a position against expiry T-1 at spot=600. Today (T), spot
    has drifted to 615 and a fresh T chain is loaded. paper.tick must:

      (a) close the T-1 position (status -> closed_expiry),
      (b) settle it at T-1's stored spot (600), NOT today's spot (615).

    Pre-fix: the position would either stay open (gate used current
    expiry) or close at the wrong spot (today's spot, not T-1's).
    """
    # ── T-1: open one call at spot=600 ──
    t_minus_1_open_ts = "2026-06-08T14:00:00+00:00"
    chain_t_minus_1 = _write_chain(tmp_path, "2026-06-08", spot=600.0,
                                     snap_ts=t_minus_1_open_ts)
    paper.tick(tmp_path, chain_t_minus_1,
                as_of_ts=t_minus_1_open_ts, expiry_iso="2026-06-08")

    # Confirm at least one position opened.
    positions = paper._load_positions(tmp_path)
    open_before = [p for p in positions if p.get("status") == "open"]
    assert len(open_before) >= 1, (
        "Test prereq failed: no positions opened at T-1.")
    open_expiries = {p["expiry"] for p in open_before}
    assert open_expiries == {"2026-06-08"}, (
        f"Test prereq failed: expected only 2026-06-08 opens, got {open_expiries}.")

    # ── T-1 close: write the settlement snapshot so paper.tick can find
    #    yesterday's spot when it goes to close yesterday's positions today.
    t_minus_1_close_ts = "2026-06-08T20:00:00+00:00"
    settle_spot_t_minus_1 = 600.0
    settle_rows = [_chain_row(K, kind, settle_spot_t_minus_1,
                                "2026-06-08", t_minus_1_close_ts)
                    for K in range(580, 621)
                    for kind in ("call", "put")]
    full_t_minus_1 = pd.concat([chain_t_minus_1, pd.DataFrame(settle_rows)],
                                 ignore_index=True)
    full_t_minus_1.to_parquet(
        tmp_path / "chains" / "SPY" / "2026-06-08.parquet",
        compression="snappy", index=False)

    # ── T (today): fresh chain at expiry T, today's spot is 615 ──
    t_today_ts = "2026-06-09T15:00:00+00:00"
    chain_t = _write_chain(tmp_path, "2026-06-09", spot=615.0,
                            snap_ts=t_today_ts)

    # Tick now — paper.tick must close the T-1 open positions even
    # though the driving chain is for T.
    paper.tick(tmp_path, chain_t,
                as_of_ts=t_today_ts, expiry_iso="2026-06-09")

    positions = paper._load_positions(tmp_path)
    still_open_t_minus_1 = [p for p in positions
                             if p.get("expiry") == "2026-06-08"
                             and p.get("status") == "open"]
    assert not still_open_t_minus_1, (
        "Bug 2 regression: prior-expiry positions remained open after "
        "today's tick. paper.tick must use per-position stored expiry, "
        "not the current chain's expiry, as the close gate.")

    closed_t_minus_1 = [p for p in positions
                         if p.get("expiry") == "2026-06-08"
                         and str(p.get("status", "")).startswith("closed")]
    assert closed_t_minus_1, "No T-1 positions closed."
    for p in closed_t_minus_1:
        assert float(p["spot_at_exit"]) == pytest.approx(settle_spot_t_minus_1), (
            f"Bug 2 regression: T-1 position settled at "
            f"{p['spot_at_exit']}, not yesterday's settlement spot "
            f"{settle_spot_t_minus_1}. paper.tick must use the stored "
            f"chain for the position's own expiry, not today's chain.")


def test_settlement_spot_helper_prefers_post_expiry_snapshot(tmp_path):
    """_settlement_spot must pick the earliest at-or-after-expiry snapshot
    when one exists, and fall back to the latest pre-expiry snapshot
    only if no post-expiry snapshot was captured."""
    # Write a chain with a pre-expiry snap at 599 and a post-expiry
    # settlement snap at 601.
    expiry = "2026-06-08"
    pre_ts = "2026-06-08T19:55:00+00:00"
    post_ts = "2026-06-08T20:00:00+00:00"
    rows = []
    for snap_ts, s in [(pre_ts, 599.0), (post_ts, 601.0)]:
        for K in range(595, 606):
            rows.append(_chain_row(K, "call", s, expiry, snap_ts))
    df = pd.DataFrame(rows)
    out = tmp_path / "chains" / "SPY" / f"{expiry}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, compression="snappy", index=False)

    from pinsight.paper import _settlement_spot, _parse_iso
    expiry_dt = _parse_iso(expiry + "T20:00:00+00:00")
    spot = _settlement_spot(tmp_path, "SPY", expiry, expiry_dt)
    assert spot == pytest.approx(601.0), (
        "Settlement spot must prefer the at-or-after-expiry snapshot.")


def test_settlement_spot_helper_returns_none_if_chain_missing(tmp_path):
    from pinsight.paper import _settlement_spot, _parse_iso
    expiry_dt = _parse_iso("2026-06-08T20:00:00+00:00")
    assert _settlement_spot(tmp_path, "SPY", "2026-06-08", expiry_dt) is None


# ----------------------------------------------------------------------
# Bug 1 — market-hours gate must yield to expired open positions
# ----------------------------------------------------------------------

def test_has_expired_open_positions_true_when_open_position_is_past_expiry(tmp_path):
    """The cli helper must detect a stale open position whose expiry has
    already passed — that's the case the daemon must NOT sleep through."""
    trades = pd.DataFrame([
        {"status": "open",          "expiry": "2026-06-08"},
        {"status": "open",          "expiry": "2026-06-09"},
        {"status": "closed_expiry", "expiry": "2026-06-08"},
    ])
    trades.to_parquet(tmp_path / "paper_trades.parquet",
                       compression="snappy", index=False)
    # 2026-06-10 00:00 UTC — both opens are past their 20:00 UTC expiry.
    now_dt = datetime(2026, 6, 10, 0, 0, tzinfo=timezone.utc)
    assert cli._has_expired_open_positions(tmp_path, now_dt) is True


def test_has_expired_open_positions_false_when_only_future_expiries_open(tmp_path):
    trades = pd.DataFrame([
        {"status": "open",          "expiry": "2026-06-10"},
        {"status": "closed_expiry", "expiry": "2026-06-08"},
    ])
    trades.to_parquet(tmp_path / "paper_trades.parquet",
                       compression="snappy", index=False)
    # 2026-06-09 12:00 UTC — open expiry is tomorrow at 20:00 UTC.
    now_dt = datetime(2026, 6, 9, 12, 0, tzinfo=timezone.utc)
    assert cli._has_expired_open_positions(tmp_path, now_dt) is False


def test_has_expired_open_positions_false_when_no_trades_file(tmp_path):
    """No file → no stuck positions → fine to sleep."""
    now_dt = datetime(2026, 6, 9, 22, 0, tzinfo=timezone.utc)
    assert cli._has_expired_open_positions(tmp_path, now_dt) is False


def test_has_expired_open_positions_false_when_only_closed(tmp_path):
    trades = pd.DataFrame([
        {"status": "closed_expiry", "expiry": "2026-06-08"},
        {"status": "closed_stop",   "expiry": "2026-06-09"},
    ])
    trades.to_parquet(tmp_path / "paper_trades.parquet",
                       compression="snappy", index=False)
    now_dt = datetime(2026, 6, 10, 0, 0, tzinfo=timezone.utc)
    assert cli._has_expired_open_positions(tmp_path, now_dt) is False


def test_latest_stored_chain_returns_most_recent(tmp_path):
    """Fallback driver for settle-only ticks: when yahoo can't return a
    fresh chain (weekend, after-hours), the daemon must still be able to
    drive paper.tick. _latest_stored_chain provides the chain frame."""
    import time
    _write_chain(tmp_path, "2026-06-08", spot=600.0,
                  snap_ts="2026-06-08T14:00:00+00:00")
    time.sleep(0.05)  # ensure mtime ordering
    _write_chain(tmp_path, "2026-06-09", spot=615.0,
                  snap_ts="2026-06-09T14:00:00+00:00")

    result = cli._latest_stored_chain(tmp_path, "SPY")
    assert result is not None
    df, expiry_iso = result
    assert expiry_iso == "2026-06-09"
    assert not df.empty


def test_latest_stored_chain_returns_none_when_no_chains(tmp_path):
    assert cli._latest_stored_chain(tmp_path, "SPY") is None
