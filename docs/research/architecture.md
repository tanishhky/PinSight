# Architecture & Design

Living document. Updates land as ADRs in `docs/decisions/`.

---

## 1. Three engines, one signal

PinSight runs three engines on the same 0DTE data feed and fuses their output into a single tradable signal.

```
        ┌──────────────────┐
data ──▶│  RND Engine      │──▶ implied distribution Q(S_T)
        ├──────────────────┤
        │  Pricing Engine  │──▶ fair premium vs market premium for a contract
        ├──────────────────┤
        │  Flow Engine     │──▶ anomaly score (informed-flow probability proxy)
        └────────┬─────────┘
                 │
                 ▼
        ┌──────────────────┐
        │  Signal Fusion   │──▶ trade-side recommendation + size suggestion
        └──────────────────┘
```

Each engine is independently testable. Signal fusion is a thin layer that combines them — start with a hand-tuned rule, replace with a calibrated model only after we have labeled outcomes.

---

## 2. RND Engine

**Goal:** Given today's 0DTE option chain at time `t`, return a function `q(S_T)` representing the market-implied risk-neutral density of the underlying at expiry `T`.

**Pipeline:**

1. **Pull chain.** All calls and puts at today's expiry, with bid/ask/mid/IV/volume/OI.
2. **Quality filter.** Drop strikes with zero bid, very wide spreads (configurable), or stale quotes (last-trade > N minutes).
3. **Mid-price → IV.** Recompute IV ourselves via Brent root-finding against Black–Scholes. Don't trust vendor IV blindly. (Pattern reused from VolEdge.)
4. **SVI fit.** Fit a Stochastic Volatility Inspired parameterization to the IV smile. SVI is parameterized as
   `w(k) = a + b · {ρ(k − m) + √((k − m)² + σ²)}`
   where `k = log(K/F)` (log-moneyness) and `w = σ_IV² · τ` (total variance). Five params; closed-form, no-arb conditions enforced.
5. **Reprice off smooth surface.** Compute fine-grid call prices from the SVI surface at strikes spaced ε apart.
6. **Differentiate twice.** `q(K) = e^(rτ) · ∂²C/∂K²`. Use central differences on the fine grid.
7. **Tail extrapolation.** Fit GEV (generalized extreme value) tails to the wings beyond the traded strike range (Figlewski 2010).
8. **Normalize.** Verify ∫q(S) dS ≈ 1 within tolerance. Renormalize.
9. **Cross-check via BKM.** Compute the first four moments of `q` two ways: (a) numerical integration over `q`, (b) BKM model-free integrals over option prices directly. If they diverge by more than a tolerance, log a warning and downgrade confidence in `q`.

**Failure modes & handling:**

- Strike grid too sparse (e.g., after-hours): fall back to Aït-Sahalia–Lo kernel estimator with broader bandwidth and flag as low-confidence.
- Negative q values from numerical differentiation noise: smooth with a small Gaussian kernel; if still negative anywhere, reject the fit and log.
- Arbitrage violations (call price not monotone in strike): repair by isotonic regression before differentiating; log how many strikes were repaired.

---

## 3. Pricing Engine

**Goal:** Given an RND `q` and a contract spec (call/put, strike, expiry = today), return:

- `fair_premium` = `e^(-rτ) · E_q[max(S_T − K, 0)]` (for a call)
- `market_premium` = current mid price
- `edge_ratio` = `market_premium / fair_premium`
- `expected_pnl` = `E_q[payoff] − market_premium`
- `prob_itm` = `Pr_q(S_T > K)` (calls)

A contract is "cheap" if `edge_ratio < 1 − threshold` and "expensive" if `edge_ratio > 1 + threshold`. The threshold accounts for spread + slippage + RND model error.

**Reality check:** the RND is the market's implied distribution. By construction, market-priced contracts have edge ≈ 1. Edge ≠ 1 is therefore one of:

1. We disagree with the market's distribution (we believe a different physical distribution → BKM physical vs risk-neutral spread)
2. Bid/ask noise on the specific contract
3. Our RND fit error

This is why pricing alone doesn't generate trades. It feeds into signal fusion alongside flow anomalies.

---

## 4. Flow Engine

**Goal:** Score each contract (or contract group) on a 0–1 anomaly scale. High score = likely informed positioning.

**Features per contract:**

1. **Δ Open Interest / Volume** (Chesney–Crameri–Mancini 2015). When the day's volume is largely opening new positions (ΔOI ≈ Volume), informed traders are establishing exposure, not closing.
2. **Volume Z-score vs trailing window.** Standard unusual-options-volume detector. Trailing 21-day mean and std on this strike/maturity.
3. **Put/call volume ratio.** Daily aggregate at strike. Conditioned on contract being OTM (Cao–Chen–Griffin 2005).
4. **OTM-call concentration.** Fraction of day's call volume in short-dated OTM strikes vs trailing baseline.
5. **IV-vs-realized spread anomaly.** Z-score of (option's IV − recent realized vol) against trailing window.
6. **VPIN bucket.** Volume-synchronized PIN over a rolling window. High VPIN = high inferred informed-flow share.

**Composite anomaly score:**

`score = max-of-top-2( sigmoid_k(z_feature_k) )`

(Pattern borrowed from Regime-Adaptive Portfolio: at least two features must agree before firing. Avoids one noisy feature triggering false alarms.)

**Labeling for backtest:** A flagged contract is "validated" if, by the close, the underlying moved in the direction implied by the flagged side (e.g., flagged OTM-call burst → underlying closed higher than threshold). Track hit rate.

---

## 5. Signal Fusion (M5)

**Inputs:**

- `q` (RND) + `pricing.edge_ratio` for the candidate contract
- `flow.anomaly_score` for the same contract or its cohort

**Output:**

```
signal = {
    'side': 'long' | 'short' | 'pass',
    'contract': <spec>,
    'conviction': 0–1,
    'rationale': str,
    'timestamp': ...,
}
```

**Initial rule (handwritten, replace later):**

```
if flow.anomaly_score > 0.7 AND pricing.edge_ratio < 1.0:
    side = same side as detected flow
    conviction = flow.anomaly_score * (1 − pricing.edge_ratio)
else:
    side = 'pass'
```

Every signal — fired or not — is appended to `logs/signals.jsonl` with all features. This is the training data for later replacement of the rule with a calibrated model.

---

## 6. Logging discipline

**Every fetch, every fit, every signal logs:**

- ISO-8601 timestamp (UTC)
- Inputs (or hash of inputs if large)
- Output summary
- Wall-clock duration
- Any warnings raised

Log format: JSONL, one event per line, in `logs/<engine>.jsonl`. Rotated daily.

**Why JSONL not text:** machine-grepable. We will need to replay flow events later for backtests.

---

## 7. Storage

**Pattern from ChronoFund:** Parquet with typed schemas, partitioned by date. No CSV. No SQLite for the main data path (too slow for the columnar access patterns we need).

**Schemas (planned):**

- `data/chains/<YYYY-MM-DD>.parquet` — option chain snapshots, one row per (contract, snapshot time)
- `data/underlying/<symbol>/<YYYY-MM-DD>.parquet` — minute-bar OHLCV
- `data/rnd/<YYYY-MM-DD>.parquet` — fitted RNDs (parameters + checkpoints)
- `data/signals.parquet` — all signals ever fired (append-only)

---

## 8. Testing strategy

- **Unit tests** for math primitives (BS pricer, Brent IV solver, SVI fit, BKM integrals) — `tests/test_<module>.py`.
- **Golden-output tests** for the RND engine: feed in a known synthetic chain generated from a Heston model, check that the recovered RND matches the Heston density at expiry within tolerance.
- **Replay tests** for the flow engine: take a known historical event (e.g., a documented takeover with pre-announcement flow), feed in the chain history, verify the detector fires.
- **No mocks of market data feeds** in tests. Use saved sample chains in `tests/fixtures/`.

---

## 9. What we're explicitly NOT doing (yet)

- No live trading execution. Signal output is logged, not routed.
- No machine learning models for signal generation until we have ≥500 labeled signals from the rule-based fusion layer.
- No full SPX feed — we start with SPY because SPX requires a paid Cboe license.
- No multi-asset. SPX/SPY only in M1–M5.
