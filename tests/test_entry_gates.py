"""Entry gates + conviction sizing (2026-07-04 redesign).

The pricing seam (paper.price_contract / paper.extract) is monkeypatched so
each strike's model fair value is fully controlled - the tests exercise the
tick's decision logic, not the RND numerics.

What must hold:
  * model-market divergence gate: fair/mid > max_model_market_ratio -> no
    trade, even when the raw "edge" looks enormous (that IS the failure
    mode: the audit's 0-for-27 lottery bucket all had ratio > 2);
  * quote-quality gate: rel spread > max_rel_spread -> no trade;
  * conviction sizing: stakes scale with the shrunk (blended) edge - a
    marginal signal gets meaningfully fewer dollars than a strong one,
    and the audit fields are stamped on the position row.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from pinsight import paper
from pinsight.pricing import ContractSpec, FairValue


SNAP = "2026-06-01T14:00:00+00:00"
AS_OF = "2026-06-01T15:00:00+00:00"


class _FakeRND:
    T = 0.25 / 365.0
    r = 0.0


def _chain(rows: list[dict]) -> pd.DataFrame:
    base = {
        "underlying": "SPY", "expiry": "2026-06-01",
        "volume": 1000, "open_interest": 5000, "iv": 0.15,
        "in_the_money": False, "underlying_price": 600.0,
        "quote_ts": SNAP, "_snapshot_ts": SNAP,
    }
    return pd.DataFrame([{**base, **r} for r in rows])


def _fake_pricer(fairs: dict[float, tuple[float, float]]):
    """fairs: {strike: (fair_premium, prob_itm)}"""
    def _price(rnd, spec: ContractSpec) -> FairValue:
        fair, p = fairs[spec.strike]
        return FairValue(
            spec=spec, fair_premium=fair, expected_payoff=fair,
            prob_itm=p,
            edge_ratio=(spec.market_premium / fair) if fair > 1e-8
                       else float("inf"),
            expected_pnl=fair - spec.market_premium,
            stdev_payoff=1.0,
        )
    return _price


def _run_tick(tmp_path: Path, monkeypatch, chain: pd.DataFrame,
              fairs: dict, rule: paper.EntryRule) -> pd.DataFrame:
    monkeypatch.setattr(paper, "extract",
                        lambda *a, **k: _FakeRND())
    monkeypatch.setattr(paper, "price_contract", _fake_pricer(fairs))
    monkeypatch.setattr(paper, "_mtm_for_open_positions",
                        lambda *a, **k: [])
    paper.tick(tmp_path, chain, rule=rule, as_of_ts=AS_OF,
               expiry_iso="2026-06-01")
    p = tmp_path / "paper_trades.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def test_divergence_gate_refuses_model_error(tmp_path, monkeypatch):
    """A $0.10 quote the model calls 'worth $1.50' (15x) is model error,
    not 15x edge - must not trade even though every legacy check passes."""
    chain = _chain([{"ticker": "T1", "contract_type": "call",
                     "strike": 610.0, "bid": 0.098, "ask": 0.102,
                     "mid": 0.10}])
    fairs = {610.0: (1.50, 0.40)}
    trades = _run_tick(tmp_path, monkeypatch, chain, fairs,
                       paper.EntryRule())
    assert trades.empty

    # Same trade with the gate relaxed IS taken -> the gate is what
    # refused it, nothing else.
    trades2 = _run_tick(tmp_path, monkeypatch, chain, fairs,
                        replace(paper.EntryRule(),
                                max_model_market_ratio=float("inf")))
    assert len(trades2) == 1


def test_rel_spread_gate_refuses_untradeable_quote(tmp_path, monkeypatch):
    """bid 0.05 / ask 0.15: 100% relative spread - 'mid' is a fiction."""
    chain = _chain([{"ticker": "T2", "contract_type": "call",
                     "strike": 610.0, "bid": 0.05, "ask": 0.15,
                     "mid": 0.10}])
    fairs = {610.0: (0.18, 0.40)}   # passes the divergence gate (1.8x)
    trades = _run_tick(tmp_path, monkeypatch, chain, fairs,
                       paper.EntryRule())
    assert trades.empty

    trades2 = _run_tick(tmp_path, monkeypatch, chain, fairs,
                        replace(paper.EntryRule(),
                                max_rel_spread=float("inf")))
    assert len(trades2) == 1


def test_conviction_sizing_scales_with_edge(tmp_path, monkeypatch):
    """A strong signal must get more capital than a marginal one, and the
    audit fields must be stamped."""
    chain = _chain([
        {"ticker": "STRONG", "contract_type": "call", "strike": 601.0,
         "bid": 1.98, "ask": 2.02, "mid": 2.00},
        {"ticker": "WEAK", "contract_type": "call", "strike": 602.0,
         "bid": 1.98, "ask": 2.02, "mid": 2.00},
    ])
    fairs = {
        601.0: (3.20, 0.55),   # big blended edge, decent prob
        602.0: (2.40, 0.45),   # marginal blended edge
    }
    trades = _run_tick(tmp_path, monkeypatch, chain, fairs,
                       paper.EntryRule())
    assert len(trades) == 2
    by_ticker = {t["ticker"]: t for _, t in trades.iterrows()}
    strong = by_ticker.get("STRONG")
    weak = by_ticker.get("WEAK")
    assert strong is not None and weak is not None
    assert strong["entry_total_cost_usd"] > weak["entry_total_cost_usd"]
    for t in (strong, weak):
        assert t["kelly_f_at_entry"] > 0
        assert t["model_market_ratio_at_entry"] <= 2.0
        assert t["fair_used_at_entry"] < fairs[t["strike"]][0]  # shrunk


def test_min_position_floor_skips_dust_stakes(tmp_path, monkeypatch):
    """Kelly conviction below min_position_usd -> no trade instead of a
    dust position."""
    chain = _chain([{"ticker": "T4", "contract_type": "call",
                     "strike": 610.0, "bid": 1.98, "ask": 2.02,
                     "mid": 2.00}])
    # Tiny blended edge: fair barely above cost after shrinkage → Kelly
    # conviction lands below the $50 floor.
    fairs = {610.0: (2.06, 0.30)}
    trades = _run_tick(tmp_path, monkeypatch, chain, fairs,
                       paper.EntryRule())
    assert trades.empty


def test_saturated_prob_itm_cannot_max_kelly(tmp_path, monkeypatch):
    """Live artifact 2026-07-06: the integrated RND returned P(ITM)=1.0
    exactly for a near-money put, sending f* to 1 regardless of edge.
    Sizing must clamp the probability below 1 and stamp kelly_f < 1."""
    chain = _chain([{"ticker": "SAT", "contract_type": "put",
                     "strike": 590.0, "bid": 2.38, "ask": 2.40,
                     "mid": 2.39}])
    fairs = {590.0: (2.82, 1.0)}   # ratio 1.18: passes gates; p saturated
    trades = _run_tick(tmp_path, monkeypatch, chain, fairs,
                       paper.EntryRule())
    assert len(trades) == 1
    t = trades.iloc[0]
    assert t["kelly_f_at_entry"] < 1.0, (
        "saturated P(ITM)=1.0 still produced full-Kelly sizing")
