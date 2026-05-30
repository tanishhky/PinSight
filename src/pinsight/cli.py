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
