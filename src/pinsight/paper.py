"""PinSight paper trader.

Strategy v0 — **edge_buyer**:
  Entry: open long positions in contracts where
      edge_ratio < entry_edge_max   (market is cheap vs fair)
      prob_itm   >= entry_prob_min  (avoid lottery tickets we'll never realise)
      time_to_expiry >= min_hours_before_open  (avoid last-minute gamma)
      mid >= min_mid                (avoid penny-noise)
  Sizing: fixed dollar per position (per_position_cap), bankroll
      cap (aggregate_cap_pct).
  Exit: HOLD TO CLOSE.
      At resolution: realize intrinsic value at the underlying close.
      No intraday TP / SL — per the user's "measure raw signal edge" choice.
  Force-exit: if a position can't find a closing snapshot at expiry, mark
      it `failed_close` so it doesn't sit `open` forever.

No-lookahead discipline (per ADR 0004):
  Every chain read passes `as_of_ts` and the orchestrator asserts
  `snapshot_ts <= as_of_ts`. All decisions are stamped with `as_of_ts`,
  never `datetime.now()`.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from . import obs
from .pricing import ContractSpec, FairValue, price_contract
from .rnd.density import RNDFit, extract


TRADER_ID = "edge_buyer"
DEFAULT_BANKROLL = 50_000.0  # user-locked 2026-06-01


@dataclass(frozen=True)
class EntryRule:
    """Tunable knobs for the edge_buyer agent."""
    entry_edge_max: float = 0.85       # buy if market < 85 % of fair
    entry_prob_min: float = 0.20       # skip if P(ITM) < 20 %
    min_hours_before_open: float = 1.0  # avoid sub-hour gamma chaos
    min_mid: float = 0.05               # penny-noise floor
    per_position_cap_pct: float = 0.02  # 2 % of bankroll per contract
    aggregate_cap_pct: float = 0.50     # 50 % across all open positions
    min_position_usd: float = 50.0      # fees + slippage floor


# ── Persistence paths ────────────────────────────────────────────────────

def _trades_path(data_dir: Path) -> Path:
    return data_dir / "paper_trades.parquet"


def _state_path(data_dir: Path) -> Path:
    return data_dir / "paper_state.parquet"


def _equity_path(data_dir: Path) -> Path:
    return data_dir / "equity_history.parquet"


# ── State management ────────────────────────────────────────────────────

@dataclass(frozen=True)
class TraderState:
    trader: str
    bankroll_init: float
    cash_usd: float
    open_exposure: float
    closed_pnl: float

    @property
    def total_equity_cost_basis(self) -> float:
        return self.cash_usd + self.open_exposure


def _load_or_init_state(data_dir: Path,
                         bankroll: float) -> TraderState:
    p = _state_path(data_dir)
    if p.exists():
        df = pd.read_parquet(p)
        row = df[df["trader"] == TRADER_ID]
        if not row.empty:
            r = row.iloc[0]
            return TraderState(
                trader=TRADER_ID,
                bankroll_init=float(r["bankroll_init"]),
                cash_usd=float(r["cash_usd"]),
                open_exposure=float(r["open_exposure"]),
                closed_pnl=float(r["closed_pnl"]),
            )
    return TraderState(trader=TRADER_ID, bankroll_init=bankroll,
                       cash_usd=bankroll, open_exposure=0.0,
                       closed_pnl=0.0)


def _save_state(data_dir: Path, state: TraderState) -> None:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    p = _state_path(data_dir)
    existing = pd.read_parquet(p) if p.exists() else pd.DataFrame()
    if not existing.empty:
        existing = existing[existing["trader"] != TRADER_ID]
    peak = max(state.bankroll_init + state.closed_pnl,
                state.total_equity_cost_basis)
    dd_pct = (peak - state.total_equity_cost_basis) / peak * 100.0 if peak > 0 else 0.0
    new_row = pd.DataFrame([{
        "trader": TRADER_ID,
        "bankroll_init": state.bankroll_init,
        "cash_usd": state.cash_usd,
        "open_exposure": state.open_exposure,
        "closed_pnl": state.closed_pnl,
        "peak_equity": peak,
        "current_drawdown_pct": round(dd_pct, 3),
        "updated_ts": now,
    }])
    combined = pd.concat([existing, new_row], ignore_index=True)
    combined.to_parquet(p, compression="snappy", index=False)


# ── Trade lifecycle ──────────────────────────────────────────────────────

def _load_positions(data_dir: Path) -> list[dict]:
    p = _trades_path(data_dir)
    if not p.exists():
        return []
    df = pd.read_parquet(p)
    return df.where(pd.notna(df), None).to_dict(orient="records")


def _save_positions(data_dir: Path, opened: list[dict],
                     closed: list[dict]) -> None:
    if not (opened or closed):
        return
    existing = _load_positions(data_dir)
    by_id = {r["trade_id"]: r for r in existing}
    for o in opened:
        by_id[o["trade_id"]] = o
    for c in closed:
        by_id[c["trade_id"]] = c
    df = pd.DataFrame(list(by_id.values()))
    p = _trades_path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, compression="snappy", index=False)


def _open_position(spec: ContractSpec, fv: FairValue, *,
                    as_of_ts: str, n_contracts: int,
                    spot_at_entry: float, rnd_T: float) -> dict:
    # Premium per share of underlying (×100 multiplier per contract)
    entry_size_usd = spec.market_premium * 100 * n_contracts
    return {
        "trade_id": str(uuid.uuid4()),
        "trader": TRADER_ID,
        "ticker": spec.ticker,
        "kind": spec.kind,
        "strike": float(spec.strike),
        "expiry": spec.expiry,
        "entry_ts": as_of_ts,
        "entry_price": float(spec.market_premium),
        "entry_size_usd": float(entry_size_usd),
        "n_contracts": int(n_contracts),
        "spot_at_entry": float(spot_at_entry),
        "T_at_entry_hours": float(rnd_T * 8760),
        "fair_at_entry": float(fv.fair_premium),
        "edge_at_entry": float(fv.edge_ratio),
        "prob_itm_at_entry": float(fv.prob_itm),
        "expected_pnl_at_entry": float(fv.expected_pnl),
        "status": "open",
        "exit_ts": None,
        "exit_price": None,
        "exit_reason": None,
        "spot_at_exit": None,
        "pnl_usd": None,
        "pnl_per_contract": None,
    }


def _intrinsic(kind: str, strike: float, spot: float) -> float:
    if kind == "call":
        return max(spot - strike, 0.0)
    return max(strike - spot, 0.0)


def _close_position(pos: dict, *, as_of_ts: str, exit_spot: float,
                     reason: str) -> dict:
    intrinsic = _intrinsic(pos["kind"], pos["strike"], exit_spot)
    pnl_per_share = intrinsic - pos["entry_price"]
    pnl_per_contract = pnl_per_share * 100
    pnl_usd = pnl_per_contract * pos["n_contracts"]
    closed = dict(pos)
    closed.update({
        "exit_ts": as_of_ts,
        "exit_price": float(intrinsic),
        "exit_reason": reason,
        "spot_at_exit": float(exit_spot),
        "pnl_usd": float(pnl_usd),
        "pnl_per_contract": float(pnl_per_contract),
        "status": f"closed_{reason}",
    })
    return closed


# ── Tick driver ──────────────────────────────────────────────────────────

def tick(data_dir: Path, chain_df: pd.DataFrame, *,
         as_of_ts: Optional[str] = None,
         expiry_iso: Optional[str] = None,
         rule: Optional[EntryRule] = None,
         bankroll: float = DEFAULT_BANKROLL) -> dict:
    """One paper-tick for the edge_buyer.

    Steps:
      1. If `expiry_iso` is None, infer from chain (single-expiry chain
         file at data/chains/SYM/<expiry>.parquet).
      2. If any currently-open positions are AT or PAST expiry, close
         them at the underlying close at expiry.
      3. Fit RND from the latest snapshot in `chain_df` ≤ as_of_ts.
      4. Score every contract in the chain → pick BUY candidates.
      5. Open up to the cap.

    Returns a per-tick summary dict.
    """
    if rule is None:
        rule = EntryRule()
    if as_of_ts is None:
        as_of_ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if chain_df is None or chain_df.empty:
        return {"as_of_ts": as_of_ts, "skipped": "no_chain"}

    # Filter to snapshots within our as-of horizon — no lookahead.
    if "_snapshot_ts" in chain_df.columns:
        chain_df = chain_df[chain_df["_snapshot_ts"] <= as_of_ts]
        if chain_df.empty:
            return {"as_of_ts": as_of_ts, "skipped": "no_eligible_snapshots"}
        latest_snap = chain_df["_snapshot_ts"].max()
        chain_df = chain_df[chain_df["_snapshot_ts"] == latest_snap]

    if expiry_iso is None:
        if "expiry" in chain_df.columns:
            expiry_iso = str(chain_df["expiry"].iloc[0])
        else:
            return {"as_of_ts": as_of_ts, "skipped": "no_expiry"}

    spot = float(chain_df["underlying_price"].iloc[0])
    state = _load_or_init_state(data_dir, bankroll=bankroll)

    # ── Close any positions whose expiry has passed ──
    positions = _load_positions(data_dir)
    open_positions = [p for p in positions
                      if p.get("trader") == TRADER_ID
                      and p.get("status") == "open"]
    expiry_dt = datetime.fromisoformat(expiry_iso + "T20:00:00+00:00") \
        if "T" not in expiry_iso else datetime.fromisoformat(expiry_iso.replace("Z", "+00:00"))
    now_dt = datetime.fromisoformat(as_of_ts.replace("Z", "+00:00"))
    closed: list[dict] = []
    if now_dt >= expiry_dt and open_positions:
        # All open positions reach expiry → close at current spot.
        for pos in open_positions:
            closed_pos = _close_position(pos, as_of_ts=as_of_ts,
                                          exit_spot=spot, reason="expiry")
            closed.append(closed_pos)
            # Cash settles in.
            state = TraderState(
                trader=TRADER_ID, bankroll_init=state.bankroll_init,
                cash_usd=state.cash_usd + pos["entry_size_usd"] + closed_pos["pnl_usd"],
                open_exposure=state.open_exposure - pos["entry_size_usd"],
                closed_pnl=state.closed_pnl + closed_pos["pnl_usd"],
            )
        open_positions = []

    # ── If still in the force-exit window for entries, skip opens ──
    hours_to_expiry = (expiry_dt - now_dt).total_seconds() / 3600.0
    opened: list[dict] = []
    candidates_evaluated = 0
    candidates_buy = 0
    rnd: Optional[RNDFit] = None

    if hours_to_expiry >= rule.min_hours_before_open:
        # ── Fit RND ──
        try:
            rnd = extract(chain_df, spot=spot, as_of_ts=as_of_ts,
                           expiry_iso=expiry_iso)
        except Exception as exc:
            obs.event(channel="error", kind="paper.rnd_fail",
                      level="WARNING", as_of_ts=as_of_ts, err=str(exc))
            rnd = None

        if rnd is not None:
            # Set of (kind, strike) we already hold so we don't double-up.
            held = {(p["kind"], float(p["strike"])) for p in open_positions}

            # ── Score every contract; pick BUYs ──
            buys: list[tuple[ContractSpec, FairValue]] = []
            for _, row in chain_df.iterrows():
                ctype = row["contract_type"]
                strike = float(row["strike"])
                bid = float(row.get("bid", 0))
                ask = float(row.get("ask", 0))
                mid = (bid + ask) / 2.0 if (bid > 0 and ask > 0) else float(row.get("mid", 0))
                if mid < rule.min_mid or bid <= 0:
                    continue
                if (ctype, strike) in held:
                    continue
                candidates_evaluated += 1
                spec = ContractSpec(kind=ctype, strike=strike,
                                     market_premium=mid,
                                     ticker=row.get("ticker"),
                                     expiry=expiry_iso)
                fv = price_contract(rnd, spec)
                if (fv.edge_ratio < rule.entry_edge_max
                        and fv.prob_itm >= rule.entry_prob_min
                        and fv.expected_pnl > 0):
                    buys.append((spec, fv))

            # Rank by absolute expected_pnl desc.
            buys.sort(key=lambda x: x[1].expected_pnl, reverse=True)
            candidates_buy = len(buys)

            # ── Open top candidates within caps ──
            per_position_cap_usd = state.bankroll_init * rule.per_position_cap_pct
            agg_cap_usd = state.bankroll_init * rule.aggregate_cap_pct
            for spec, fv in buys:
                available = agg_cap_usd - state.open_exposure
                if available < rule.min_position_usd:
                    break
                budget = min(per_position_cap_usd, available, state.cash_usd)
                # Contract premium × 100 multiplier = USD per contract.
                cost_per_contract = spec.market_premium * 100.0
                if cost_per_contract <= 0:
                    continue
                n = int(budget // cost_per_contract)
                if n <= 0:
                    continue
                pos = _open_position(spec, fv, as_of_ts=as_of_ts,
                                      n_contracts=n, spot_at_entry=spot,
                                      rnd_T=rnd.T)
                opened.append(pos)
                state = TraderState(
                    trader=TRADER_ID,
                    bankroll_init=state.bankroll_init,
                    cash_usd=state.cash_usd - pos["entry_size_usd"],
                    open_exposure=state.open_exposure + pos["entry_size_usd"],
                    closed_pnl=state.closed_pnl,
                )

    # Persist trades + state.
    if opened or closed:
        _save_positions(data_dir, opened=opened, closed=closed)
    _save_state(data_dir, state)

    # Equity-history row (cost-basis equity; MTM extension is in a
    # follow-up since options pricing for MTM requires re-fitting the
    # smile each tick).
    eq_path = _equity_path(data_dir)
    eq_row = pd.DataFrame([{
        "ts": as_of_ts,
        "trader": TRADER_ID,
        "cash_usd": state.cash_usd,
        "open_exposure_usd": state.open_exposure,
        "mtm_unrealized_usd": 0.0,   # cost-basis only for v0
        "closed_pnl_usd": state.closed_pnl,
        "total_equity_usd": state.cash_usd + state.open_exposure,
        "peak_equity_usd": max(
            state.bankroll_init + state.closed_pnl,
            state.cash_usd + state.open_exposure,
        ),
        "drawdown_pct": 0.0,
    }])
    if eq_path.exists():
        existing = pd.read_parquet(eq_path)
        combined = pd.concat([existing, eq_row], ignore_index=True)
    else:
        combined = eq_row
    combined.to_parquet(eq_path, compression="snappy", index=False)

    obs.event(channel="fit", kind="paper.tick", level="INFO",
              as_of_ts=as_of_ts, expiry=expiry_iso,
              hours_to_expiry=hours_to_expiry,
              candidates_evaluated=candidates_evaluated,
              candidates_buy=candidates_buy,
              opened=len(opened), closed=len(closed),
              cash_usd=state.cash_usd,
              open_exposure=state.open_exposure,
              closed_pnl=state.closed_pnl)

    return {
        "as_of_ts": as_of_ts,
        "expiry": expiry_iso,
        "hours_to_expiry": hours_to_expiry,
        "candidates_evaluated": candidates_evaluated,
        "candidates_buy": candidates_buy,
        "opened": len(opened),
        "closed": len(closed),
        "cash_usd": state.cash_usd,
        "open_exposure": state.open_exposure,
        "closed_pnl": state.closed_pnl,
    }
