"""PinSight paper trader.

Strategy v0 — **edge_buyer**:
  Entry: open long positions in contracts where
      market_premium + fill_slippage < fair_premium - per_share_costs
      (i.e., post-cost expected_pnl > 0)
      AND  edge_ratio < entry_edge_max  (market < fair before cost)
      AND  prob_itm   >= entry_prob_min  (avoid lottery tickets)
      AND  time_to_expiry >= min_hours_before_open
      AND  mid >= min_mid
  Fill price: mid + slippage_frac_of_half_spread × (ask − mid)
  Sizing: fixed dollar per position (per_position_cap), bankroll
      cap (aggregate_cap_pct).
  Exit: HOLD TO CLOSE.
      At resolution: realize intrinsic value at the underlying close;
      pay exit commission.
  Force-exit: position past expiry without closing snapshot → marked
      `closed_failed_close`.

Transaction costs (institutional-desk profile, locked 2026-06-03):
  commission_per_contract      = $0.35 each leg
  slippage_frac_of_half_spread = 0.25 (close to mid)
  fees_per_contract            = $0.00 (assumed passthrough)

MTM (dual):
  Each tick computes both
    mtm_fair_usd  = (fair_premium_now − entry_fill_price) × 100 × n
    mtm_market_usd = (chain_mid_now      − entry_fill_price) × 100 × n
  Stored in equity_history with drawdown computed from mtm_market.

No-lookahead discipline (per ADR-0004):
  Every chain read passes `as_of_ts` and the orchestrator asserts
  `snapshot_ts <= as_of_ts` via parsed-datetime comparison. All
  decisions stamped with `as_of_ts`, never `datetime.now()`.
"""

from __future__ import annotations

import math
import os
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
DEFAULT_BANKROLL = 50_000.0


# ── ISO 8601 parsing (shared with RND module) ───────────────────────────

def _parse_iso(ts: str) -> datetime:
    s = str(ts).replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


@dataclass(frozen=True)
class EntryRule:
    """Tunable knobs for the edge_buyer agent.

    2026-07-04 redesign, from the audit of the first 126 post-reset trades
    (-$21k realized, hit 28.6%):

      * every sub-$0.30 entry lost (0/27, -$26.1k) — all were trades where
        the RND fair diverged hugely from the market quote (median
        fair/mid 2.5-4.4x, max 235x). "mid < 0.85 x fair" buys whatever
        the model MOST overvalues, which is adverse selection against our
        own model error;
      * near-money entries ($1+ fills, fair/mid ~1.3x) were PROFITABLE
        (+$7.4k at ~41% hit);
      * replaying the same 126 trades with the divergence gate at 2.0x
        keeps 93 of them and flips total P&L to +$8.7k.

    Three formulated mechanisms replace magic-threshold buying:
      1. divergence gate — when model fair and market mid disagree by more
         than `max_model_market_ratio`, model error is the likelier
         explanation than edge: no trade.
      2. quote-quality gate — relative spread above `max_rel_spread`
         means the quote isn't executable at anything near mid and the
         slippage model is meaningless.
      3. conviction sizing — fractional Kelly on the shrunk edge (fair
         blended toward mid in log space with `model_trust_weight`),
         replacing the fixed full-cap budget that put ~$1,000 on 167
         nickel contracts as happily as on an ATM straddle leg.

    All knobs env-overridable via PINSIGHT_* (see from_env).
    """
    entry_edge_max: float = 0.85
    entry_prob_min: float = 0.20
    min_hours_before_open: float = 1.0
    min_mid: float = 0.05
    per_position_cap_pct: float = 0.02
    aggregate_cap_pct: float = 0.50
    min_position_usd: float = 50.0
    # Transaction costs (institutional desk)
    commission_per_contract: float = 0.35
    slippage_frac_of_half_spread: float = 0.25
    fees_per_contract: float = 0.0
    # Model-error and quote-quality gates (2026-07-04)
    max_model_market_ratio: float = 2.0
    max_rel_spread: float = 0.15
    # Conviction sizing: posterior log-fair = (1-w)*log(mid) + w*log(fair)
    # (w = trust in our model vs the market's price), then fractional
    # Kelly at `kelly_fraction` on the blended edge.
    model_trust_weight: float = 0.5
    kelly_fraction: float = 0.25

    @classmethod
    def from_env(cls) -> "EntryRule":
        """EntryRule with PINSIGHT_* environment overrides applied."""
        def f(name: str, default: float) -> float:
            try:
                return float(os.getenv(name, str(default)))
            except (TypeError, ValueError):
                return default
        d = cls()
        return cls(
            entry_edge_max=f("PINSIGHT_ENTRY_EDGE_MAX", d.entry_edge_max),
            entry_prob_min=f("PINSIGHT_ENTRY_PROB_MIN", d.entry_prob_min),
            min_hours_before_open=f("PINSIGHT_MIN_HOURS_BEFORE_OPEN",
                                    d.min_hours_before_open),
            min_mid=f("PINSIGHT_MIN_MID", d.min_mid),
            per_position_cap_pct=f("PINSIGHT_PER_POSITION_CAP_PCT",
                                   d.per_position_cap_pct),
            aggregate_cap_pct=f("PINSIGHT_AGGREGATE_CAP_PCT",
                                d.aggregate_cap_pct),
            min_position_usd=f("PINSIGHT_MIN_POSITION_USD",
                               d.min_position_usd),
            commission_per_contract=f("PINSIGHT_COMMISSION_PER_CONTRACT",
                                      d.commission_per_contract),
            slippage_frac_of_half_spread=f(
                "PINSIGHT_SLIPPAGE_FRAC", d.slippage_frac_of_half_spread),
            fees_per_contract=f("PINSIGHT_FEES_PER_CONTRACT",
                                d.fees_per_contract),
            max_model_market_ratio=f("PINSIGHT_MAX_MODEL_MARKET_RATIO",
                                     d.max_model_market_ratio),
            max_rel_spread=f("PINSIGHT_MAX_REL_SPREAD", d.max_rel_spread),
            model_trust_weight=f("PINSIGHT_MODEL_TRUST_WEIGHT",
                                 d.model_trust_weight),
            kelly_fraction=f("PINSIGHT_KELLY_FRACTION", d.kelly_fraction),
        )


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
    open_exposure: float    # USD cost basis (entry_fill_price × 100 × n) of open positions
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


def _save_state(data_dir: Path, state: TraderState,
                 peak_market_equity: float,
                 current_market_equity: float) -> None:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    p = _state_path(data_dir)
    existing = pd.read_parquet(p) if p.exists() else pd.DataFrame()
    if not existing.empty:
        existing = existing[existing["trader"] != TRADER_ID]
    dd_pct = ((peak_market_equity - current_market_equity)
              / peak_market_equity * 100.0
              if peak_market_equity > 0 else 0.0)
    new_row = pd.DataFrame([{
        "trader": TRADER_ID,
        "bankroll_init": state.bankroll_init,
        "cash_usd": state.cash_usd,
        "open_exposure": state.open_exposure,
        "closed_pnl": state.closed_pnl,
        "peak_equity": peak_market_equity,
        "current_market_equity": current_market_equity,
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


# ── Cost helpers ────────────────────────────────────────────────────────

def _entry_fill_price(mid: float, ask: float, bid: float,
                       slip_frac: float) -> float:
    """Long buy: pay mid + slippage_frac × (ask - mid). slip_frac=0 → mid,
    slip_frac=1 → ask. Bounded above by ask."""
    half_spread = max(ask - mid, 0.0)
    return float(min(ask, mid + slip_frac * half_spread))


def _exit_fill_price_market(market_mid: float, market_bid: float,
                              slip_frac: float) -> float:
    """Long sell at exit: receive mid - slippage_frac × (mid - bid).
    Bounded below by bid."""
    half_spread = max(market_mid - market_bid, 0.0)
    return float(max(market_bid, market_mid - slip_frac * half_spread))


def _open_position(spec: ContractSpec, fv: FairValue, *,
                    as_of_ts: str, n_contracts: int,
                    spot_at_entry: float, rnd_T: float,
                    entry_fill_price: float, ask: float, bid: float,
                    commission_paid: float,
                    fees_paid: float) -> dict:
    """A position record. cost basis = (fill × 100 × n) + commission + fees."""
    notional = entry_fill_price * 100 * n_contracts
    total_cost = notional + commission_paid + fees_paid
    return {
        "trade_id": str(uuid.uuid4()),
        "trader": TRADER_ID,
        "ticker": spec.ticker,
        "kind": spec.kind,
        "strike": float(spec.strike),
        "expiry": spec.expiry,
        "entry_ts": as_of_ts,
        "entry_mid": float(spec.market_premium),
        "entry_ask": float(ask),
        "entry_bid": float(bid),
        "entry_fill_price": float(entry_fill_price),
        "entry_notional_usd": float(notional),
        "entry_commission_usd": float(commission_paid),
        "entry_fees_usd": float(fees_paid),
        "entry_total_cost_usd": float(total_cost),
        "n_contracts": int(n_contracts),
        "spot_at_entry": float(spot_at_entry),
        "T_at_entry_hours": float(rnd_T * 8760),
        "fair_at_entry": float(fv.fair_premium),
        "edge_at_entry": float(fv.edge_ratio),
        "prob_itm_at_entry": float(fv.prob_itm),
        "expected_pnl_at_entry_gross": float(fv.expected_pnl),
        "status": "open",
        "exit_ts": None,
        "exit_price": None,
        "exit_proceeds_usd": None,
        "exit_commission_usd": None,
        "exit_fees_usd": None,
        "exit_reason": None,
        "spot_at_exit": None,
        "pnl_usd": None,
        "pnl_per_contract": None,
    }


def _intrinsic(kind: str, strike: float, spot: float) -> float:
    if kind == "call":
        return max(spot - strike, 0.0)
    return max(strike - spot, 0.0)


def _settlement_spot(data_dir: Path, symbol: str, expiry_iso: str,
                       expiry_dt: datetime) -> Optional[float]:
    """Return the underlying close price for `expiry_iso` from the stored
    chain parquet, picking the snapshot at-or-after the expiry time. Falls
    back to the latest pre-expiry snapshot if no post-expiry snapshot was
    captured. Returns None if the chain file is missing or empty.

    Used by the per-position expiry-close path so positions from a prior
    day's expiry close at THAT day's settlement spot, not today's.
    (Bug 2 fix 2026-06-10.)
    """
    chain_path = data_dir / "chains" / symbol / f"{expiry_iso}.parquet"
    if not chain_path.exists():
        return None
    try:
        df = pd.read_parquet(chain_path)
    except Exception:
        return None
    if df.empty or "_snapshot_ts" not in df.columns:
        return None
    snap_dt = df["_snapshot_ts"].map(_parse_iso)
    post = df[snap_dt >= expiry_dt]
    if not post.empty:
        # Prefer the earliest at-or-after-expiry snapshot (true settlement).
        idx = snap_dt[snap_dt >= expiry_dt].idxmin()
        return float(df.loc[idx, "underlying_price"])
    # Fallback: most recent pre-expiry snapshot.
    pre = df[snap_dt < expiry_dt]
    if pre.empty:
        return None
    idx = snap_dt[snap_dt < expiry_dt].idxmax()
    return float(df.loc[idx, "underlying_price"])


def _close_position_at_expiry(pos: dict, *, as_of_ts: str, exit_spot: float,
                                rule: EntryRule) -> dict:
    """HOLD-TO-CLOSE settlement: payoff = intrinsic per share × 100 × n.
    Exit commission applies (settlement also costs at most institutional
    desks; if pos goes worthless, no exit commission per most broker
    contracts — we model it conservatively as charged regardless)."""
    n = int(pos["n_contracts"])
    intrinsic = _intrinsic(pos["kind"], pos["strike"], exit_spot)
    proceeds = intrinsic * 100 * n
    exit_commission = rule.commission_per_contract * n if intrinsic > 0 else 0.0
    exit_fees = rule.fees_per_contract * n if intrinsic > 0 else 0.0
    net_proceeds = proceeds - exit_commission - exit_fees
    pnl_usd = net_proceeds - pos["entry_total_cost_usd"]
    pnl_per_contract = pnl_usd / n if n else 0.0
    closed = dict(pos)
    closed.update({
        "exit_ts": as_of_ts,
        "exit_price": float(intrinsic),
        "exit_proceeds_usd": float(proceeds),
        "exit_commission_usd": float(exit_commission),
        "exit_fees_usd": float(exit_fees),
        "exit_reason": "expiry",
        "spot_at_exit": float(exit_spot),
        "pnl_usd": float(pnl_usd),
        "pnl_per_contract": float(pnl_per_contract),
        "status": "closed_expiry",
    })
    return closed


# ── MTM helpers ──────────────────────────────────────────────────────────

def _mtm_market_mid(chain_df: pd.DataFrame, kind: str,
                     strike: float) -> Optional[float]:
    """Find the current mid for a (kind, strike) in the latest chain
    snapshot. Returns None if not present."""
    row = chain_df[(chain_df["contract_type"] == kind)
                   & (chain_df["strike"] == strike)]
    if row.empty:
        return None
    r = row.iloc[0]
    bid = float(r.get("bid", 0) or 0)
    ask = float(r.get("ask", 0) or 0)
    if bid <= 0 or ask <= 0:
        m = r.get("mid")
        return float(m) if m is not None and not pd.isna(m) else None
    return float((bid + ask) / 2.0)


def _mtm_for_open_positions(open_positions: list[dict],
                              rnd: Optional[RNDFit],
                              chain_df: pd.DataFrame,
                              expiry_iso: str) -> list[dict]:
    """For each open position, compute both mtm_fair and mtm_market.

    Returns a list of dicts (one per position) with the per-position MTM
    info, for storage in equity_history details.
    """
    rows = []
    for pos in open_positions:
        n = int(pos["n_contracts"])
        fill = float(pos["entry_fill_price"])
        kind = pos["kind"]
        strike = float(pos["strike"])
        cost = float(pos["entry_total_cost_usd"])
        # Fair value via RND
        if rnd is not None:
            fv = price_contract(rnd, ContractSpec(
                kind=kind, strike=strike,
                market_premium=fill,  # market_premium placeholder; we only need fair
                ticker=pos.get("ticker"),
                expiry=pos.get("expiry"),
            ))
            mtm_fair_per_share = fv.fair_premium - fill
            mtm_fair_usd = mtm_fair_per_share * 100 * n
            current_fair_value_usd = fv.fair_premium * 100 * n
        else:
            mtm_fair_per_share = float("nan")
            mtm_fair_usd = float("nan")
            current_fair_value_usd = float("nan")
        # Market mid
        market_mid = _mtm_market_mid(chain_df, kind, strike)
        if market_mid is not None:
            mtm_market_per_share = market_mid - fill
            mtm_market_usd = mtm_market_per_share * 100 * n
            current_market_value_usd = market_mid * 100 * n
        else:
            mtm_market_per_share = float("nan")
            mtm_market_usd = float("nan")
            current_market_value_usd = cost   # no quote → fall back to cost basis
        rows.append({
            "trade_id": pos["trade_id"],
            "kind": kind,
            "strike": strike,
            "n_contracts": n,
            "entry_fill_price": fill,
            "current_fair_price": current_fair_value_usd / (100 * n) if n else float("nan"),
            "current_market_mid": market_mid if market_mid is not None else float("nan"),
            "mtm_fair_per_share": mtm_fair_per_share,
            "mtm_market_per_share": mtm_market_per_share,
            "mtm_fair_usd": mtm_fair_usd,
            "mtm_market_usd": mtm_market_usd,
            "current_fair_value_usd": current_fair_value_usd,
            "current_market_value_usd": current_market_value_usd,
            "entry_cost_usd": cost,
        })
    return rows


# ── Tick driver ──────────────────────────────────────────────────────────

def tick(data_dir: Path, chain_df: pd.DataFrame, *,
         as_of_ts: Optional[str] = None,
         expiry_iso: Optional[str] = None,
         rule: Optional[EntryRule] = None,
         bankroll: float = DEFAULT_BANKROLL) -> dict:
    """One paper-tick for the edge_buyer."""
    if rule is None:
        rule = EntryRule.from_env()
    if as_of_ts is None:
        as_of_ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if chain_df is None or chain_df.empty:
        return {"as_of_ts": as_of_ts, "skipped": "no_chain"}

    # ── No-lookahead filter (parsed-datetime, not string) ──
    as_of_dt = _parse_iso(as_of_ts)
    if "_snapshot_ts" in chain_df.columns:
        snap_dt = chain_df["_snapshot_ts"].map(_parse_iso)
        chain_df = chain_df[snap_dt <= as_of_dt]
        if chain_df.empty:
            return {"as_of_ts": as_of_ts, "skipped": "no_eligible_snapshots"}
        snap_dt = chain_df["_snapshot_ts"].map(_parse_iso)
        latest_idx = snap_dt.idxmax()
        latest_snap = chain_df.loc[latest_idx, "_snapshot_ts"]
        chain_df = chain_df[chain_df["_snapshot_ts"] == latest_snap]

    if expiry_iso is None:
        if "expiry" in chain_df.columns:
            expiry_iso = str(chain_df["expiry"].iloc[0])
        else:
            return {"as_of_ts": as_of_ts, "skipped": "no_expiry"}

    spot = float(chain_df["underlying_price"].iloc[0])
    state = _load_or_init_state(data_dir, bankroll=bankroll)

    # ── Close any positions whose expiry has passed ──
    # Per-position expiry handling (Bug 2 fix 2026-06-10): each position's
    # own `expiry` field is the gate, not the current chain's expiry. A
    # position opened against an earlier expiry must still be closed when
    # `now_dt >= position.expiry`, regardless of which expiry the current
    # chain is for. Settlement spot is sourced from the stored chain
    # parquet for that expiry, not from the current chain.
    positions = _load_positions(data_dir)
    open_positions = [p for p in positions
                      if p.get("trader") == TRADER_ID
                      and p.get("status") == "open"]
    now_dt = _parse_iso(as_of_ts)
    closed: list[dict] = []

    # Group open positions by stored expiry, then settle each expired group.
    by_expiry: dict[str, list[dict]] = {}
    for p in open_positions:
        exp = str(p.get("expiry") or "")
        if exp:
            by_expiry.setdefault(exp, []).append(p)

    underlying_symbol = str(
        chain_df["underlying"].iloc[0] if "underlying" in chain_df.columns else "SPY"
    ).upper()

    survivors: list[dict] = []
    for pos_expiry, pos_list in by_expiry.items():
        group_expiry_dt = (_parse_iso(pos_expiry + "T20:00:00+00:00")
                            if "T" not in pos_expiry
                            else _parse_iso(pos_expiry))
        if now_dt < group_expiry_dt:
            survivors.extend(pos_list)
            continue

        # Past expiry — find a settlement spot from the stored chain for
        # THIS expiry, not the current chain.
        if pos_expiry == expiry_iso:
            settlement_spot = spot
        else:
            settlement_spot = _settlement_spot(
                data_dir, underlying_symbol, pos_expiry, group_expiry_dt)
        if settlement_spot is None:
            obs.event(channel="error", kind="paper.settlement_spot_missing",
                      level="WARNING", as_of_ts=as_of_ts,
                      expiry=pos_expiry, symbol=underlying_symbol,
                      n_unsettled=len(pos_list))
            survivors.extend(pos_list)
            continue

        for pos in pos_list:
            closed_pos = _close_position_at_expiry(
                pos, as_of_ts=as_of_ts, exit_spot=settlement_spot, rule=rule)
            closed.append(closed_pos)
            net_proceeds = (closed_pos["exit_proceeds_usd"]
                             - closed_pos["exit_commission_usd"]
                             - closed_pos["exit_fees_usd"])
            # Cost basis: prefer the new-schema entry_total_cost_usd;
            # fall back to legacy entry_size_usd. Defensive against any
            # pre-2026-06-04 trades that didn't track commissions.
            cost_basis = pos.get("entry_total_cost_usd")
            if cost_basis is None or (isinstance(cost_basis, float)
                                        and pd.isna(cost_basis)):
                cost_basis = pos.get("entry_size_usd") or 0.0
            state = TraderState(
                trader=TRADER_ID, bankroll_init=state.bankroll_init,
                cash_usd=state.cash_usd + net_proceeds,
                open_exposure=state.open_exposure - float(cost_basis),
                closed_pnl=state.closed_pnl + closed_pos["pnl_usd"],
            )
    open_positions = survivors

    # Compute hours-to-expiry off the CURRENT chain's expiry (used by the
    # entry-window guard below). This is the next deadline coming up.
    expiry_dt = (_parse_iso(expiry_iso + "T20:00:00+00:00")
                  if "T" not in expiry_iso
                  else _parse_iso(expiry_iso))

    # ── RND fit (always — used for both entry decisions and MTM) ──
    hours_to_expiry = (expiry_dt - now_dt).total_seconds() / 3600.0
    rnd: Optional[RNDFit] = None
    try:
        rnd = extract(chain_df, spot=spot, as_of_ts=as_of_ts,
                       expiry_iso=expiry_iso)
    except Exception as exc:
        obs.event(channel="error", kind="paper.rnd_fail",
                  level="WARNING", as_of_ts=as_of_ts, err=str(exc))

    # ── Open new positions if time-window permits ──
    opened: list[dict] = []
    candidates_evaluated = 0
    candidates_buy = 0
    if hours_to_expiry >= rule.min_hours_before_open and rnd is not None:
        held = {(p["kind"], float(p["strike"])) for p in open_positions}
        per_share_cost_floor = (rule.commission_per_contract +
                                  rule.fees_per_contract) / 100.0
        buys: list[tuple[ContractSpec, FairValue, float, float, float]] = []
        for _, row in chain_df.iterrows():
            ctype = row["contract_type"]
            strike = float(row["strike"])
            bid = float(row.get("bid", 0) or 0)
            ask = float(row.get("ask", 0) or 0)
            if bid <= 0 or ask <= 0:
                continue
            mid = (bid + ask) / 2.0
            if mid < rule.min_mid:
                continue
            if (ctype, strike) in held:
                continue
            candidates_evaluated += 1
            entry_fill = _entry_fill_price(mid, ask, bid,
                                             rule.slippage_frac_of_half_spread)
            # ── Intrinsic-value floor guard (2026-06-22) ──
            # You cannot buy an option below its intrinsic value — that's an
            # instant arbitrage that does not exist in a real market. An ITM
            # strike quoted below intrinsic is stale/crossed/garbage data, NOT
            # edge. Without this guard the agent "buys free money" on bad ITM
            # quotes and books phantom profit at settlement (the source of the
            # implausible +250k paper return: puts +283k / calls -33k). Also
            # reject any non-finite fill. Legitimate OTM edge trades have
            # intrinsic == 0 and pass trivially, so the real thesis is intact.
            intrinsic = _intrinsic(ctype, strike, spot)
            if not math.isfinite(entry_fill) or entry_fill < intrinsic - 1e-9:
                continue
            # ── Quote-quality gate (2026-07-04) ──
            # A relative spread this wide means "mid" is a fiction: the
            # quote can't be executed near it and the slippage model no
            # longer describes anything. Winning trades in the audit had
            # median rel-spread 0.006-0.008; the 0-for-27 lottery bucket
            # sat at 0.05-0.11.
            rel_spread = (ask - bid) / mid
            if rel_spread > rule.max_rel_spread:
                continue
            spec = ContractSpec(kind=ctype, strike=strike,
                                 market_premium=mid,
                                 ticker=row.get("ticker"),
                                 expiry=expiry_iso)
            fv = price_contract(rnd, spec)
            if not math.isfinite(fv.fair_premium) or fv.fair_premium <= 0:
                continue
            # ── Model-error gate (2026-07-04) ──
            # When our fair and the market's mid disagree by more than
            # max_model_market_ratio, the likelier explanation is that
            # the RND is wrong at that strike (the audit found "fair
            # $15.36" on a $0.106 put), not that the market is offering
            # a free multiple. Replayed on the first 126 trades this
            # single gate turned -$21k into +$8.7k.
            div_ratio = fv.fair_premium / mid
            if div_ratio > rule.max_model_market_ratio:
                continue
            # Pre-cost edge check (a contract must look cheap before
            # we even think about costs).
            if not (fv.edge_ratio < rule.entry_edge_max
                     and fv.prob_itm >= rule.entry_prob_min):
                continue
            # ── Shrunk fair: posterior blend of model and market ──
            # log(fair_used) = (1-w) log(mid) + w log(fair). With w=0.5
            # this treats model and market as equally noisy estimators;
            # the claimed edge is haircut accordingly before EV and
            # sizing ever see it.
            w = rule.model_trust_weight
            fair_used = (mid ** (1.0 - w)) * (fv.fair_premium ** w)
            # Post-cost EV check on the SHRUNK fair, not the raw model.
            roundtrip_cost_per_share = 2 * per_share_cost_floor
            net_edge_per_share = (fair_used - entry_fill
                                    - roundtrip_cost_per_share)
            if net_edge_per_share <= 0:
                continue
            buys.append((spec, fv, entry_fill, bid, ask,
                         fair_used, div_ratio, rel_spread))

        buys.sort(
            key=lambda x: (x[5] - x[2]) * x[1].prob_itm,
            reverse=True)
        candidates_buy = len(buys)

        per_position_cap_usd = state.bankroll_init * rule.per_position_cap_pct
        agg_cap_usd = state.bankroll_init * rule.aggregate_cap_pct
        for spec, fv, fill, bid, ask, fair_used, div_ratio, rel_spread in buys:
            available = agg_cap_usd - state.open_exposure
            if available < rule.min_position_usd:
                break
            cost_per_contract_with_fees = (fill * 100.0
                                            + rule.commission_per_contract
                                            + rule.fees_per_contract)
            if cost_per_contract_with_fees <= 0:
                continue
            # ── Conviction sizing: fractional Kelly on the shrunk edge ──
            # Binary approximation of the expiry payoff: win prob
            # p = model P(ITM); expected payoff conditional on ITM is
            # fair_used / p (since fair_used ≈ p · E[payoff | ITM]);
            # odds per $ of cost b = payoff_if_itm / cost - 1.
            # f* = p - (1-p)/b, applied at kelly_fraction. The old code
            # spent min(cap, cash) on EVERY signal, which is what put
            # $1,000 on 167 five-cent contracts. Now conviction scales
            # the stake and the per-position cap only truncates it.
            p = fv.prob_itm
            cost_per_share = fill + 2 * per_share_cost_floor
            if p <= 0 or cost_per_share <= 0:
                continue
            payoff_if_itm = fair_used / p
            b = payoff_if_itm / cost_per_share - 1.0
            if b <= 0:
                continue
            f_star = p - (1.0 - p) / b
            if f_star <= 0:
                continue
            budget = min(rule.kelly_fraction * f_star * state.bankroll_init,
                         per_position_cap_usd, available, state.cash_usd)
            if budget < rule.min_position_usd:
                continue
            n = int(budget // cost_per_contract_with_fees)
            if n <= 0:
                continue
            commission_paid = rule.commission_per_contract * n
            fees_paid = rule.fees_per_contract * n
            pos = _open_position(spec, fv, as_of_ts=as_of_ts,
                                  n_contracts=n, spot_at_entry=spot,
                                  rnd_T=rnd.T, entry_fill_price=fill,
                                  ask=ask, bid=bid,
                                  commission_paid=commission_paid,
                                  fees_paid=fees_paid)
            # Audit trail for the next post-mortem.
            pos.update({
                "fair_used_at_entry": round(fair_used, 4),
                "model_market_ratio_at_entry": round(div_ratio, 3),
                "rel_spread_at_entry": round(rel_spread, 4),
                "kelly_f_at_entry": round(f_star, 4),
            })
            opened.append(pos)
            total_cost = float(pos["entry_total_cost_usd"])
            state = TraderState(
                trader=TRADER_ID,
                bankroll_init=state.bankroll_init,
                cash_usd=state.cash_usd - total_cost,
                open_exposure=state.open_exposure + total_cost,
                closed_pnl=state.closed_pnl,
            )

    # Persist trades + state.
    if opened or closed:
        _save_positions(data_dir, opened=opened, closed=closed)

    # ── Dual MTM for open positions ──
    open_after = [p for p in _load_positions(data_dir)
                   if p.get("trader") == TRADER_ID
                   and p.get("status") == "open"]
    mtm_rows = _mtm_for_open_positions(open_after, rnd, chain_df, expiry_iso)
    sum_mtm_fair_usd = sum(r["mtm_fair_usd"] for r in mtm_rows
                            if r["mtm_fair_usd"] == r["mtm_fair_usd"])  # filter NaN
    sum_mtm_market_usd = sum(r["mtm_market_usd"] for r in mtm_rows
                              if r["mtm_market_usd"] == r["mtm_market_usd"])
    sum_current_market_value = sum(r["current_market_value_usd"] for r in mtm_rows)
    sum_current_fair_value = sum(r["current_fair_value_usd"] for r in mtm_rows
                                   if r["current_fair_value_usd"]
                                      == r["current_fair_value_usd"])

    # Equity (market basis): cash + sum(current_market_value)
    current_market_equity = state.cash_usd + sum_current_market_value
    current_fair_equity = state.cash_usd + (sum_current_fair_value
                                              if sum_current_fair_value
                                              == sum_current_fair_value
                                              else sum_current_market_value)

    # Update peak (market-basis) — read from existing equity_history
    eq_path = _equity_path(data_dir)
    if eq_path.exists():
        prev_eq = pd.read_parquet(eq_path)
        prev_peak = float(prev_eq["peak_market_equity_usd"].max()
                           if "peak_market_equity_usd" in prev_eq.columns
                           and not prev_eq.empty
                           else state.bankroll_init)
    else:
        prev_peak = state.bankroll_init
    peak_market_equity = max(prev_peak, current_market_equity)
    drawdown_pct = ((peak_market_equity - current_market_equity)
                    / peak_market_equity * 100.0
                    if peak_market_equity > 0 else 0.0)

    _save_state(data_dir, state, peak_market_equity, current_market_equity)

    eq_row = pd.DataFrame([{
        "ts": as_of_ts,
        "trader": TRADER_ID,
        "cash_usd": state.cash_usd,
        "open_exposure_cost_basis_usd": state.open_exposure,
        "open_positions": len(open_after),
        "sum_mtm_fair_usd": float(sum_mtm_fair_usd) if mtm_rows else 0.0,
        "sum_mtm_market_usd": float(sum_mtm_market_usd) if mtm_rows else 0.0,
        "sum_current_fair_value_usd": float(sum_current_fair_value) if mtm_rows else 0.0,
        "sum_current_market_value_usd": float(sum_current_market_value) if mtm_rows else 0.0,
        "closed_pnl_usd": state.closed_pnl,
        "total_equity_market_usd": current_market_equity,
        "total_equity_fair_usd": current_fair_equity,
        "peak_market_equity_usd": peak_market_equity,
        "drawdown_pct": round(drawdown_pct, 3),
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
              closed_pnl=state.closed_pnl,
              sum_mtm_market_usd=float(sum_mtm_market_usd),
              sum_mtm_fair_usd=float(sum_mtm_fair_usd),
              total_equity_market=current_market_equity,
              drawdown_pct=drawdown_pct)

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
        "sum_mtm_market_usd": float(sum_mtm_market_usd),
        "sum_mtm_fair_usd": float(sum_mtm_fair_usd),
        "total_equity_market": current_market_equity,
        "drawdown_pct": drawdown_pct,
    }
