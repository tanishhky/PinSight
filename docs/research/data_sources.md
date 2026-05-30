# Data Sources — Evaluation

Goal: assemble enough free data to build and validate PinSight without ever paying for a feed. Honest about gaps.

---

## Primary: `yfinance` (Yahoo Finance)

- **Cost:** Free, no API key.
- **What it gives us:** SPY (and any equity) option chains via `Ticker.option_chain(expiry)`. Each row has strike, last, bid, ask, volume, openInterest, impliedVolatility.
- **Granularity:** Quotes refresh on-demand (we poll). Not tick-by-tick. Typically 15-minute delayed for the casual user; sometimes near-live during market hours.
- **Limits:** Yahoo rate limits aggressive scrapers. Polite polling (e.g., once per 30s during market hours) generally works. No SLA — it can break.
- **Use:** Primary intraday chain feed for SPY 0DTE. The RND engine consumes this directly.

## Secondary: Tradier sandbox

- **Cost:** Free with brokerage signup. No funded account required for sandbox.
- **What it gives us:** Cleaner option chain data + Greeks for SPX and SPY. Streaming quotes via WebSocket on the live API.
- **Limits:** Sandbox feed is delayed and lacks some endpoints. Live feed requires opening an account (free) but applies for "live data" entitlement.
- **Use:** Cross-validation against Yahoo. Greek values for sanity-checking our own Brent-root IV computation.
- **Setup:** Register at developer.tradier.com, copy API key into `.env`.

## Tertiary: Alpaca

- **Cost:** Free with paper-account signup.
- **What it gives us:** Equity quotes/trades, options chains, historical bars.
- **Limits:** Options data is newer and less broad than equity. Worth testing.
- **Use:** Backup for chain data and for underlying minute bars.

## Underlying minute bars

- `yfinance.download(period='5d', interval='1m')` — last 7 days of 1-min bars, free.
- For older history, Alpaca free tier covers 5+ years of equity minute bars.

## Risk-free rate

- **FRED via `pandas_datareader` or `fredapi`** — free, requires free API key.
- Use 3-month Treasury bill yield (`DGS3MO`) as `r` for discounting. For 0DTE specifically, `τ` is hours, so the impact of `r` is tiny — even a constant 5% assumption is fine. Use FRED only for hygiene.

## Historical option data for backtesting

This is the hardest gap.

- **Cboe DataShop free samples:** Limited (a few days of SPX trades). Useful for code validation, not real backtests.
- **`yfinance` historical option chains:** Yahoo does **not** retain historical chains. We have to build our own archive by polling and persisting. Start now; archive is empty until we run.
- **Optiondata.org and similar:** Sometimes free EOD historical options data, but quality varies.
- **OptionMetrics IvyDB / WRDS:** Gold standard, but paid (academic access via NYU might be available — check after enrolment in coursework that requires it).

**Implication:** the first 4–8 weeks of running PinSight are an investment in building our own historical chain archive. M2 (RND engine) can be developed on whatever days of data we have at that point. Real backtests need 3+ months of archive.

## NOT free, NOT using

- **Polygon.io paid tier** — would solve all the above, but ~$200/mo.
- **CBOE official SPX feed** — $1k+/mo licensing.
- **ORATS, IVolatility** — commercial.

If at some later point we want to spend money for a controlled backtest, Polygon is the most cost-effective single upgrade. Not now.

---

## What this means for the build

- **M1 data ingestion** focuses on Yahoo (SPY) + Tradier sandbox (cross-check). Both wrapped in adapters under `src/pinsight/data/` with a uniform interface so we can swap providers later without touching the RND engine.
- **Archiving starts at M1.** Every chain we pull goes to `data/chains/<date>.parquet`. By M5, we should have weeks of history.
- **No fabricated data.** If a snapshot fails, log it as missing and move on. No interpolation in the data layer.
