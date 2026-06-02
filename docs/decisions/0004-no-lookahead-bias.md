# ADR-0004: No Lookahead Bias

**Status:** Accepted (2026-06-01)
**Applies to:** every function that reads historical chain data or makes
a paper-trading decision.

## Decision

Every function that reads historical option-chain data takes an explicit
`as_of_ts` argument and filters its data source so that

    snapshot_ts <= as_of_ts

Decision functions are pure: they consume (state, as_of_ts) and return a
decision. They do **not** call `datetime.now()` internally.

Assertions guard at boundaries — any function that ultimately drives P&L
math asserts `snapshot.ts <= as_of_ts` and crashes loudly if violated.

## Why

In real-time trading this discipline is trivially satisfied (we only have
"now"). But the same code paths get reused for backtests and replays.
Without `as_of_ts` discipline, a backtest accidentally reads forward and
the reported P&L is fiction. The cost of "I forgot to filter" is a paper
that looked profitable in research but loses money live.

This rule is enforced architecturally — not "remembered" — because human
attention is unreliable.

## Mechanics in PinSight

| Function | Discipline |
|---|---|
| `rnd.density.extract(df, *, as_of_ts, ...)` | Filters `df` to `_snapshot_ts ≤ as_of_ts`; asserts the chosen snapshot is not in the future. |
| `paper.tick(data_dir, chain_df, *, as_of_ts, ...)` | Calls `extract(...)` with the same `as_of_ts`; never reads any other snapshot. |
| `paper._close_position(pos, *, as_of_ts, exit_spot, ...)` | Stamps `exit_ts = as_of_ts`. Never `datetime.now()`. |
| `cli.cmd_poll` | Stamps `as_of_ts = datetime.now(utc)` ONCE at the top of each loop iteration; passes it through to `paper.tick`. |

## How we prove it

`tests/test_paper_no_lookahead.py` injects T+k snapshots into the chain
parquet and asserts that a paper-tick made at T returns the same decision
regardless of whether future data is also present on disk. If the
assertion fires, the test fails and CI blocks merge.

## Out of scope

- Cross-session leakage (e.g., reading future data via filesystem mtimes
  outside the chain parquet). We don't currently use mtimes for
  decisions, but if we ever do, that path needs its own assertion.
- Multi-symbol races (no PinSight code looks at multiple symbols
  simultaneously yet).
