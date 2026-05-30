# PinSight

**A free, open research platform for 0DTE options: probability distribution inference, fair-premium pricing, and unusual flow detection.**

PinSight studies zero-days-to-expiration (0DTE) options on SPX/SPY with three coupled goals:

1. **Distribution inference.** Extract the risk-neutral probability distribution implied by the 0DTE option chain (Breeden–Litzenberger second-derivative method with SVI smoothing; cross-validated against BKM model-free moments).
2. **Fair-premium pricing.** Given an inferred distribution and a hypothetical contract, compute a fair premium and the expected payoff so a discretionary trader can size positions on edge, not gut.
3. **Unusual-flow detection.** Detect statistically anomalous order flow — large bursts in open interest, volume spikes in short-dated OTM contracts, IV/term-structure dislocations — that historically precede directional moves. Trade alongside the anomaly when conviction is high.

> **Legal framing.** PinSight detects publicly observable anomalies in options markets. It does **not** trade on material non-public information. "Insider detection" here means *statistical inference of likely-informed positioning from observable trade and quote data*, which is the same activity flow-watching firms perform every day.

---

## Status

Pre-alpha. Scaffold + research foundation only. Nothing here trades yet.

## Why 0DTE

0DTE SPX options now make up >50% of SPX option volume on many days. They concentrate three useful properties for research:

- **Pure gamma exposure.** Theta and vega are nearly zero a few hours from expiry, so the option price tracks the underlying nonlinearly through gamma. The risk-neutral distribution at expiry is whatever the strip says it is.
- **Fast feedback.** A signal triggered at 10am has its label by 4pm. No multi-week holding periods, no event drift.
- **Concentrated informed flow.** Recent literature (Adams–Fontaine–Ornthanalai 2024, Božović 2025) documents that 0DTE order flow is dominated by short-horizon directional bets and market-maker hedging. Anomalies stand out.

## Architecture (planned)

```
                ┌──────────────────────────────────────────┐
                │            Data Ingestion Layer          │
                │   Yahoo · Tradier sandbox · CBOE samples │
                └────────────────────┬─────────────────────┘
                                     │
                ┌────────────────────▼─────────────────────┐
                │       Storage (Parquet, typed schemas)   │
                │   Pattern reused from ChronoFund engine  │
                └─────┬──────────────────┬─────────────────┘
                      │                  │
        ┌─────────────▼────────┐  ┌──────▼──────────┐
        │  Distribution Engine │  │   Flow Engine    │
        │  · BL second-derive  │  │  · OI deltas     │
        │  · SVI smoothing     │  │  · Volume/OI ratio│
        │  · BKM moments       │  │  · Skew snapshots │
        │  · Tail extrapolation│  │  · VPIN (toxicity)│
        └─────────────┬────────┘  └──────┬──────────┘
                      │                  │
                ┌─────▼──────────────────▼─────┐
                │    Signal & Pricing Layer    │
                │  · Fair premium given dist.  │
                │  · Anomaly score             │
                │  · Trade-side recommendation │
                └──────────────┬───────────────┘
                               │
                ┌──────────────▼───────────────┐
                │  Output: logs, alerts, CLI   │
                │  (web UI is a later phase)   │
                └──────────────────────────────┘
```

## Research foundation

See `docs/research/papers.md` for the full annotated bibliography. The core threads:

| Theme | Anchor paper |
|---|---|
| RND from option prices | Breeden & Litzenberger (1978) — second-derivative method |
| Model-free moments | Bakshi, Kapadia, Madan (2003) — BKM variance/skew/kurtosis |
| Volatility surface | Gatheral (2004) — SVI parameterization |
| Informed trading (foundational) | Easley & O'Hara (1992) — PIN |
| Informed trading (flow-toxicity) | Easley, López de Prado, O'Hara (2012) — VPIN |
| Options informed flow | Pan & Poteshman (2006); Cao, Chen, Griffin (2005) |
| Detecting abnormal options trades | Chesney, Crameri, Mancini (2015) |
| Pre-takeover informed options | Augustin, Brenner, Subrahmanyam (2019) |
| 0DTE microstructure | Adams, Fontaine, Ornthanalai (2024); Božović (2025); Vasquez et al. (2025) |

## Data sources

All free or freemium. See `docs/research/data_sources.md` for the trade-off matrix.

- **Yahoo Finance** via `yfinance` — primary intraday chain for SPY (free, no key)
- **Tradier sandbox** — chains + Greeks for SPX/SPY (free with brokerage signup)
- **CBOE DataShop free samples** — historical validation set
- **FRED** — risk-free rate (3-month T-bill) for discounting

## Repo layout

```
PinSight/
├── README.md
├── .env.example          # template — copy to .env
├── .gitignore
├── pyproject.toml
├── docs/
│   ├── research/
│   │   ├── papers.md         # annotated bibliography
│   │   ├── architecture.md   # design decisions, math
│   │   └── data_sources.md   # free data evaluation
│   └── decisions/            # ADRs as we go
├── src/
│   └── pinsight/
│       ├── __init__.py
│       ├── config.py
│       ├── data/             # ingestion adapters
│       ├── rnd/              # risk-neutral density engine
│       ├── flow/             # unusual-activity detector
│       └── pricing/          # fair-premium engine
├── tests/
├── logs/                 # gitignored; runtime logs
└── data/                 # gitignored; cached chains
```

## Roadmap

- **M1 — Data ingestion (week 1).** Pull live SPY 0DTE chain from `yfinance`; persist to Parquet; log every fetch with timestamp.
- **M2 — RND engine (week 2).** Implement Breeden–Litzenberger with SVI smoothing; validate against BKM moments and Monte Carlo round-trips.
- **M3 — Fair premium calculator (week 3).** Given a contract and a fitted RND, compute the fair price and an "edge vs market" ratio.
- **M4 — Flow detector (week 4).** Open-interest deltas, volume/OI ratios, IV-vs-realized spread anomaly z-scores. Backtest on historical events.
- **M5 — Signal layer (week 5).** Combine distribution edge + flow anomaly into a single tradable signal; log all signals with timestamps.
- **M6 — Web UI (later).** Optional FastAPI + React dashboard following the VolEdge pattern.

## License

TBD (likely MIT).
