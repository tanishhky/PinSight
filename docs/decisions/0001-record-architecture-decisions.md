# 1. Record architecture decisions

Date: 2026-05-30

## Status

Accepted.

## Context

PinSight makes several non-obvious design choices: SVI before differentiation,
max-of-top-2 anomaly aggregation, JSONL logging, archive-from-day-one for
historical data. These will be questioned later. We want a paper trail.

## Decision

Every meaningful design choice gets an ADR in `docs/decisions/NNNN-<slug>.md`.

Format: Context, Decision, Consequences. Date and status header.

## Consequences

- Slight overhead per decision.
- Future contributors (and future-Tanishk) can read *why* something was built
  the way it was without re-deriving.
- Encourages thinking before coding.
