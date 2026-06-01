# PinSight TODO

Tracked work for the 0DTE options engine. See
`~/dev/Sentinel/CONTRACT-SPEC.md` for the contract this engine implements.

---

## Next up — contract slice + lookahead safety net

PinSight has data but no real "engine" yet (no entry/exit logic, no paper
trades). Wiring its side of the contract NOW means that when an entry/exit
strategy lands, it'll be config-driven and audit-tracked from day one.

- [ ] **`manifest/manifest.json`** declaring `engine_id=pinsight`,
      version, `departments=[zero-dte]`, `agents=[flow-detector]`,
      `risk_limits` (TBD — leave the agent disabled by default until a real
      strategy exists), `schemas` (parquet shape for chains + flagged
      contracts), `ui_tabs` (chain / flags / logs / smile), `capabilities`
      (`kill_switch`, `scheduled_only`).
- [ ] **`manifest_runtime.py`**: load manifest, expose to other modules.
- [ ] **`state.json` writer**: at the end of every `morning` / `midday` /
      `close` launchd run, write `data/state.json` with `last_fetch_ts`,
      contracts fetched, kill_switch state.
- [ ] **`allocation-audit.jsonl` writer**: append on enable/disable + on
      every successful scheduled run (so the audit trail proves the
      schedule ran).
- [ ] **kill-switch on the fetcher**: when `kill_switch=true`, the morning /
      midday / close jobs no-op after logging.

## Lookahead safety net

PinSight today is mostly a fetcher — no decisions are made yet. But the
moment a flow-detection strategy lands, it'll need the same discipline as
DriftEdge: every function that reads historical chains takes an explicit
`as_of_ts` and filters `_snapshot_ts <= as_of_ts`.

- [ ] **ADR**: copy / adapt
      `~/dev/DriftEdge/docs/decisions/0004-no-lookahead-bias.md` to
      `docs/decisions/0004-no-lookahead-bias.md` in this repo. The rule is
      identical; the example code is different.
- [ ] **`as_of_ts` plumbing**: when the flow-detection signal exists, every
      function reading chains must accept `as_of_ts`. Assertion at the read
      boundary.
- [ ] **T+k injection test**: same pattern as DriftEdge.

## Engine work (out of scope until contract slice + ADR are in)

These are real-engine tasks that should NOT be started until the contract
slice and the lookahead ADR are in place, because they will get rewritten
otherwise.

- [ ] **flow detector → paper trade**: today FLAGS is just a vol/OI ≥ 1
      ranking. Decide entry/exit rule for those flagged contracts (likely:
      buy the flow, exit on price move or time decay).
- [ ] **IV-smile model**: fit a parametric smile to compare current
      surface against historical.
- [ ] **realized vs implied vol panel**: pull SPY 30-day RV from yfinance,
      overlay on the smile.

## UI work (Sentinel-side — tracked in Sentinel TODO.md)

PinSight's tabs in Sentinel are currently text-only (4 KPI cards + a
metadata table on CHAIN; one table on FLAGS). The build-out (IV smile,
vol-by-strike, contract table, flag-event timeline) lives in
`~/dev/Sentinel/TODO.md` under "PinSight UI build-out" — those are Sentinel
changes, not PinSight changes, per the Five Rules.

PinSight's job here: keep the data clean and the schemas stable so the
Sentinel-side charts have something to read.

## Smaller fixes

- [ ] **Polygon free-tier dead-end**: the experiment with Polygon options is
      noted and abandoned (free tier doesn't include options). Delete the
      half-wired Polygon code in `data/polygon.py` if any remains, or
      document why it's still there.
- [ ] **yfinance rate-limit handling**: the daemon currently swallows
      transient failures. Add exponential backoff with a max of 3 retries
      and log every retry through `obs.event`.
- [ ] **launchd plist times in ET vs system**: confirm the morning/midday/
      close cron times honor DST; macOS launchd runs in local time so this
      should be fine but document it.

## Done (recent)

- [x] Yahoo Finance (yfinance) adapter for SPY options chain
- [x] Three scheduled launchd jobs (morning 09:35 / midday 12:30 / close 16:10)
- [x] Vol/OI flag scoring (≥ 1 flagged, ≥ 1000 vol minimum)
- [x] Move from `~/Documents/` to `~/dev/` (TCC restriction workaround)
