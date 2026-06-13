"""PinSight CLI — entry point for fetches, fits, and signal runs.

Usage:
    python -m pinsight.cli fetch-chain SPY            # today's 0DTE chain
    python -m pinsight.cli fetch-chain SPY --expiry 2026-06-06
    python -m pinsight.cli ping                       # health check
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from . import config as cfg
from . import obs
from .data.chain_normalizer import normalize_polygon_chain
from .data.persistence import write_chain
from .data.polygon import PolygonClient, PolygonError
from .data import yahoo
from .inspect import inspect_chain
from .evaluator import eval_flags


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def cmd_ping(_: argparse.Namespace, c: cfg.Config) -> int:
    """Cheap probe: hit a tiny Polygon endpoint to verify the key works."""
    if not c.polygon_api_key:
        obs.event(channel="run", kind="ping.no_key", level="ERROR")
        return 2
    client = PolygonClient(c.polygon_api_key)
    try:
        # Reference query for SPY contracts; doesn't pull market data.
        contracts = client.reference_contracts("SPY", limit=1)
        obs.event(channel="run", kind="ping.ok", level="INFO",
                  sample_ticker=contracts[0].get("ticker") if contracts else None)
        return 0
    except PolygonError as exc:
        obs.event(channel="run", kind="ping.fail", level="ERROR", err=str(exc))
        return 1


def cmd_fetch_chain(args: argparse.Namespace, c: cfg.Config) -> int:
    underlying = args.symbol.upper()
    expiry = _parse_date(args.expiry) if args.expiry else None
    snapshot_ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    provider = args.provider

    obs.event(channel="run", kind="fetch.begin", level="INFO",
              underlying=underlying, expiry=str(expiry) if expiry else "nearest",
              provider=provider)

    if provider == "polygon":
        if not c.polygon_api_key:
            obs.event(channel="run", kind="fetch.no_key", level="ERROR")
            return 2
        client = PolygonClient(c.polygon_api_key)
        try:
            payload = client.snapshot_option_chain(
                underlying, expiry=expiry or date.today())
        except PolygonError as exc:
            obs.event(channel="run", kind="fetch.fail", level="ERROR", err=str(exc))
            return 1
        df = normalize_polygon_chain(payload) if payload else None
    elif provider == "yahoo":
        df = yahoo.fetch_chain(underlying, expiry=expiry)
    else:
        obs.event(channel="run", kind="fetch.unknown_provider",
                  level="ERROR", provider=provider)
        return 2

    if df is None or df.empty:
        obs.event(channel="run", kind="fetch.empty", level="WARNING",
                  underlying=underlying, provider=provider,
                  hint="No contracts returned; expiry may be unavailable")
        return 0

    chosen_expiry = expiry or date.fromisoformat(str(df["expiry"].iloc[0]))
    path = write_chain(df, c.data_dir, underlying=underlying,
                       expiry=chosen_expiry, snapshot_ts=snapshot_ts)
    obs.event(channel="run", kind="fetch.complete", level="INFO",
              path=str(path), contracts=len(df),
              dte=(chosen_expiry - date.today()).days)
    return 0


def cmd_paper_tick(args: argparse.Namespace, c: cfg.Config) -> int:
    """One-shot: fetch the chain, run paper.tick, log result."""
    from . import paper
    underlying = args.symbol.upper()
    expiry = _parse_date(args.expiry) if args.expiry else None
    snapshot_ts = datetime.now(timezone.utc).isoformat(timespec="seconds")

    df = yahoo.fetch_chain(underlying, expiry=expiry)
    if df is None or df.empty:
        obs.event(channel="run", kind="paper_tick.fetch_empty", level="WARNING")
        return 0
    chosen_expiry = expiry or date.fromisoformat(str(df["expiry"].iloc[0]))
    write_chain(df, c.data_dir, underlying=underlying,
                 expiry=chosen_expiry, snapshot_ts=snapshot_ts)

    # Re-read from disk so we get _snapshot_ts column and consistent shape.
    chain_path = c.data_dir / "chains" / underlying / f"{chosen_expiry.isoformat()}.parquet"
    chain_df = pd.read_parquet(chain_path)
    result = paper.tick(c.data_dir, chain_df,
                         as_of_ts=snapshot_ts,
                         expiry_iso=chosen_expiry.isoformat())
    obs.event(channel="run", kind="paper_tick.done", level="INFO", **result)
    return 0


def _has_expired_open_positions(data_dir: Path, now_dt: datetime) -> bool:
    """Return True if any open position's stored expiry is at-or-past `now_dt`.

    Used by the market-hours gate (Bug 1 fix 2026-06-10): we must NOT sleep
    through the close instant of an open expiry, even when the configured
    window says we are off-hours. Reads the trades parquet directly to keep
    the check cheap — no chain fetch required.
    """
    trades_path = data_dir / "paper_trades.parquet"
    if not trades_path.exists():
        return False
    try:
        df = pd.read_parquet(trades_path, columns=["status", "expiry"])
    except Exception:
        return False
    open_df = df[df["status"].astype(str) == "open"]
    if open_df.empty:
        return False
    expiry_dt = pd.to_datetime(
        open_df["expiry"].astype(str) + "T20:00:00+00:00",
        utc=True, errors="coerce",
    )
    return bool((expiry_dt <= pd.Timestamp(now_dt)).any())


def _latest_stored_chain(data_dir: Path, underlying: str):
    """Return (chain_df, expiry_iso) for the most recently modified chain
    parquet on disk under `data/chains/<UNDERLYING>/`, or None if no chain
    has been stored yet. Used as a fallback driver when yahoo can't return
    a fresh chain but stuck expired positions still need settling.
    """
    chains_dir = data_dir / "chains" / underlying.upper()
    if not chains_dir.exists():
        return None
    parquets = sorted(
        chains_dir.glob("*.parquet"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not parquets:
        return None
    latest = parquets[0]
    try:
        df = pd.read_parquet(latest)
    except Exception:
        return None
    return df, latest.stem


def cmd_poll(args: argparse.Namespace, c: cfg.Config) -> int:
    """Long-running daemon: fetch chain + paper-tick on a fixed interval."""
    import time
    from . import paper

    underlying = args.symbol.upper()
    interval = max(30, int(args.interval))
    market_hours_only = bool(args.market_hours_only)

    obs.event(channel="run", kind="poll.start", level="INFO",
              underlying=underlying, interval_s=interval,
              market_hours_only=market_hours_only)

    iteration = 0
    while True:
        loop_start = time.time()
        iteration += 1
        as_of_ts = datetime.now(timezone.utc).isoformat(timespec="seconds")

        # Optional market-hours gate (rough — UTC; 13:30-20:00 UTC ≈ 09:30-16:00 ET
        # during US summer, 14:30-21:00 in winter; we use the broader summer
        # window since most of the year is EDT and false-positives off-hours
        # are harmless — they just no-op on stale chains).
        #
        # Bug 1 fix (2026-06-10): the original gate silenced the daemon at
        # 20:00 UTC = 4 PM ET, which is also the 0DTE settlement instant.
        # If a position was open at the bell, the daemon never woke to close
        # it. Now we check for stale expired open positions before sleeping;
        # if any exist, one tick is forced so paper.tick can settle them.
        now_dt = datetime.now(timezone.utc)
        if market_hours_only:
            weekday = now_dt.weekday()  # 0=Mon, 6=Sun
            hour_utc = now_dt.hour + now_dt.minute / 60.0
            outside_window = (
                weekday >= 5 or hour_utc < 13.5 or hour_utc > 20.25
            )
            if outside_window:
                if _has_expired_open_positions(c.data_dir, now_dt):
                    obs.event(channel="run", kind="poll.expiry_sweep_forced",
                              level="INFO", weekday=weekday, hour_utc=hour_utc)
                else:
                    obs.event(channel="run", kind="poll.outside_market_hours",
                              level="DEBUG", weekday=weekday, hour_utc=hour_utc)
                    time.sleep(interval)
                    continue

        # 1. Fetch chain
        try:
            df = yahoo.fetch_chain(underlying)
        except Exception as exc:
            obs.event(channel="error", kind="poll.fetch_fail",
                      level="WARNING", err=str(exc))
            df = None

        if df is not None and not df.empty:
            chosen_expiry = date.fromisoformat(str(df["expiry"].iloc[0]))
            try:
                write_chain(df, c.data_dir, underlying=underlying,
                             expiry=chosen_expiry, snapshot_ts=as_of_ts)
            except Exception as exc:
                obs.event(channel="error", kind="poll.persist_fail",
                          level="WARNING", err=str(exc))

            # 2. Paper tick
            try:
                chain_path = c.data_dir / "chains" / underlying / f"{chosen_expiry.isoformat()}.parquet"
                chain_df = pd.read_parquet(chain_path)
                result = paper.tick(c.data_dir, chain_df,
                                     as_of_ts=as_of_ts,
                                     expiry_iso=chosen_expiry.isoformat())
                obs.event(channel="fit", kind="poll.tick_done",
                          level="INFO", iteration=iteration, **result)
            except Exception as exc:
                obs.event(channel="error", kind="poll.tick_fail",
                          level="WARNING", err=str(exc),
                          exc_type=type(exc).__name__)
        else:
            # No fresh chain. If we still have stuck expired positions, drive
            # a settle-only tick from the most recently stored chain — the
            # per-position expiry-close path (Bug 2 fix) reads each position's
            # own settlement spot from its own stored chain parquet, so any
            # chain is fine here; this only feeds paper.tick a non-empty
            # frame so it reaches the close block.
            if _has_expired_open_positions(c.data_dir, now_dt):
                fallback = _latest_stored_chain(c.data_dir, underlying)
                if fallback is not None:
                    fb_chain_df, fb_expiry_iso = fallback
                    try:
                        result = paper.tick(c.data_dir, fb_chain_df,
                                             as_of_ts=as_of_ts,
                                             expiry_iso=fb_expiry_iso)
                        obs.event(channel="fit", kind="poll.tick_settle_only",
                                  level="INFO", iteration=iteration,
                                  fallback_expiry=fb_expiry_iso, **result)
                    except Exception as exc:
                        obs.event(channel="error", kind="poll.tick_fail",
                                  level="WARNING", err=str(exc),
                                  exc_type=type(exc).__name__)
                else:
                    obs.event(channel="run", kind="poll.no_chain_for_settle",
                              level="WARNING")
            else:
                obs.event(channel="run", kind="poll.no_chain", level="DEBUG")

        # Sleep until next tick.
        elapsed = time.time() - loop_start
        sleep_for = max(0.5, interval - elapsed)
        time.sleep(sleep_for)


def cmd_monday_workflow(args: argparse.Namespace, c: cfg.Config) -> int:
    """Convenience: fetch today's 0DTE chain, persist, inspect.

    With --eval, additionally score flagged contracts against close.
    """
    underlying = args.symbol.upper()
    expiry = _parse_date(args.expiry) if args.expiry else None

    # 1. Fetch
    df = yahoo.fetch_chain(underlying, expiry=expiry)
    if df is None or df.empty:
        obs.event(channel="run", kind="monday.fetch_empty", level="WARNING")
        return 0
    chosen_expiry = expiry or date.fromisoformat(str(df["expiry"].iloc[0]))
    snapshot_ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    write_chain(df, c.data_dir, underlying=underlying,
                expiry=chosen_expiry, snapshot_ts=snapshot_ts)

    # 2. Inspect
    inspect_chain(c.data_dir, underlying, chosen_expiry)

    # 3. Optional: evaluate
    if args.eval:
        eval_flags(c.data_dir, underlying, chosen_expiry)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pinsight", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("ping", help="Verify Polygon API key works").set_defaults(func=cmd_ping)

    fc = sub.add_parser("fetch-chain", help="Pull an option chain snapshot")
    fc.add_argument("symbol", help="Underlying ticker (e.g. SPY)")
    fc.add_argument("--expiry", help="YYYY-MM-DD; defaults to nearest expiry")
    fc.add_argument("--provider", choices=["yahoo", "polygon"], default="yahoo",
                    help="Data provider (default: yahoo)")
    fc.set_defaults(func=cmd_fetch_chain)

    le = sub.add_parser("list-expiries", help="List available option expiries")
    le.add_argument("symbol", help="Underlying ticker (e.g. SPY)")
    le.set_defaults(func=lambda args, c: (
        obs.event(channel="run", kind="expiries.list", level="INFO",
                  underlying=args.symbol.upper(),
                  expiries=yahoo.list_expiries(args.symbol)) or 0
    ))

    ic = sub.add_parser("inspect-chain", help="Describe a stored chain snapshot")
    ic.add_argument("symbol", help="Underlying ticker")
    ic.add_argument("expiry", help="YYYY-MM-DD")
    ic.set_defaults(func=lambda args, c: (
        inspect_chain(c.data_dir, args.symbol, _parse_date(args.expiry)) and 0
    ))

    ef = sub.add_parser("eval-flags",
                        help="Score flagged contracts against actual expiry outcome")
    ef.add_argument("symbol", help="Underlying ticker")
    ef.add_argument("expiry", help="YYYY-MM-DD (must match a stored chain)")
    ef.add_argument("--final-price", type=float, default=None,
                    help="Override expiry-day close price (default: fetch via Yahoo)")
    ef.set_defaults(func=lambda args, c: (
        eval_flags(c.data_dir, args.symbol, _parse_date(args.expiry),
                   final_price=args.final_price) and 0
    ))

    mw = sub.add_parser("monday-workflow",
                        help="Fetch 0DTE chain + inspect (run morning); add --eval to score at close")
    mw.add_argument("symbol", default="SPY", nargs="?")
    mw.add_argument("--expiry", help="YYYY-MM-DD; defaults to nearest")
    mw.add_argument("--eval", action="store_true",
                    help="Run end-of-day evaluation (use after close, with full chain stored)")
    mw.set_defaults(func=cmd_monday_workflow)

    pt = sub.add_parser("paper-tick",
                        help="One-shot paper trader tick (fetch chain + decide)")
    pt.add_argument("symbol", default="SPY", nargs="?")
    pt.add_argument("--expiry", help="YYYY-MM-DD; defaults to nearest")
    pt.set_defaults(func=cmd_paper_tick)

    poll = sub.add_parser("poll",
                          help="Long-running daemon: fetch chain + paper-tick every N seconds")
    poll.add_argument("symbol", default="SPY", nargs="?")
    poll.add_argument("--interval", type=int, default=90,
                       help="Seconds between ticks (default 90)")
    poll.add_argument("--market-hours-only", action="store_true",
                       help="Only run during US equity market hours (09:30-16:00 ET)")
    poll.set_defaults(func=cmd_poll)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    c = cfg.load()
    obs.configure(c.log_dir, level=c.log_level)
    obs.install_excepthook()
    try:
        code = args.func(args, c)
    finally:
        summary = obs.finish()
    return code


if __name__ == "__main__":
    sys.exit(main())
