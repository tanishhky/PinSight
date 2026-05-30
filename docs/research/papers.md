# Research Foundation — Annotated Bibliography

Curated literature underpinning PinSight. Each entry has a one-paragraph annotation explaining *why it matters here*.

---

## Part 1 — Risk-neutral density extraction

### Breeden, D. T., & Litzenberger, R. H. (1978)
**"Prices of State-Contingent Claims Implicit in Option Prices."** *Journal of Business*, 51(4), 621–651.

The foundational result. The risk-neutral probability density of the underlying at expiry equals the second partial derivative of a European call price with respect to strike, discounted at the risk-free rate. *Why for us:* This is the entire basis of the distribution engine. No pricing model assumed — just no-arbitrage and a sufficient strike grid. The 0DTE chain is dense enough on SPX/SPY for this to work in practice, though smoothing is required.

### Gatheral, J. (2004)
**"A Parsimonious Arbitrage-Free Implied Volatility Parameterization with Application to the Valuation of Volatility Derivatives."** Presentation, Global Derivatives & Risk Management.

Introduces the SVI (Stochastic Volatility Inspired) parameterization of the implied volatility smile. *Why for us:* Raw market IV across strikes is noisy. Before differentiating prices twice (BL), we fit SVI to the IV smile, reprice options off the smooth surface, then differentiate. This is the standard practitioner pipeline.

### Bakshi, G., Kapadia, N., & Madan, D. (2003)
**"Stock Return Characteristics, Skew Laws, and the Differential Pricing of Individual Equity Options."** *Review of Financial Studies*, 16(1), 101–143.

Model-free formulas for the risk-neutral variance, skewness, and kurtosis of returns as integrals over option prices across strikes. *Why for us:* BKM moments serve as a cross-check on the BL-derived RND. If our RND's first four moments diverge from BKM, our smoothing or tail extrapolation is wrong. Tanishk has implemented BKM before in VolEdge, so the math is familiar.

### Aït-Sahalia, Y., & Lo, A. W. (1998)
**"Nonparametric Estimation of State-Price Densities Implicit in Financial Asset Prices."** *Journal of Finance*, 53(2), 499–547.

Kernel-based nonparametric estimation of the state-price density. *Why for us:* An alternative to BL+SVI when the strike grid is sparse — useful in lower-liquidity hours. Worth implementing as a backup estimator.

### Figlewski, S. (2010)
**"Estimating the Implied Risk-Neutral Density for the U.S. Market Portfolio."** In *Volatility and Time Series Econometrics: Essays in Honor of Robert F. Engle*.

A practitioner's guide to actually building an RND on SPX data, covering tail extrapolation (generalized extreme value distribution for the wings) and the role of OTM options. *Why for us:* This is the operational reference for handling the two hard parts of BL — sparse strikes and missing tails. Cite this implementation pattern in `docs/research/architecture.md`.

---

## Part 2 — Informed trading detection (foundational)

### Easley, D., & O'Hara, M. (1992)
**"Time and the Process of Security Price Adjustment."** *Journal of Finance*, 47(2), 577–605.

The original PIN (Probability of Informed Trading) model. Models the order arrival process as a mixture of informed and uninformed traders and backs out the probability of informed flow from observable buy/sell imbalances. *Why for us:* Conceptual anchor for everything in the flow engine. Even though we won't fit PIN directly (it's slow and needs trade-direction signing), the framework — flow toxicity, informed vs noise traders — defines the vocabulary.

### Easley, D., López de Prado, M., & O'Hara, M. (2012)
**"Flow Toxicity and Liquidity in a High-Frequency World."** *Review of Financial Studies*, 25(5), 1457–1493.

VPIN: a high-frequency, volume-clock version of PIN that doesn't require trade direction (uses bulk volume classification instead). *Why for us:* VPIN is implementable on tick-level data we can get from Yahoo/Tradier. It's the workhorse toxicity measure in the flow engine. Adapt it to options volume.

### Easley, D., O'Hara, M., & Srinivas, P. S. (1998)
**"Option Volume and Stock Prices: Evidence on Where Informed Traders Trade."** *Journal of Finance*, 53(2), 431–465.

Empirical: informed traders sometimes prefer options over stock because of leverage. Option volume contains information about future stock prices not yet in the stock price. *Why for us:* Theoretical justification for the whole project. If options volume *only* mirrored stock volume, there'd be no informational edge to extract.

### Pan, J., & Poteshman, A. M. (2006)
**"The Information in Option Volume for Future Stock Prices."** *Review of Financial Studies*, 19(3), 871–908.

Uses put/call volume ratios from initiating trades (not just all volume) and shows they predict next-day stock returns. *Why for us:* Concrete features for the flow engine. Specifically: put-volume / call-volume ratio, conditioned on whether the trades opened or closed positions. Tradier API provides enough to approximate this.

---

## Part 3 — Detecting abnormal/informed options activity

### Cao, C., Chen, Z., & Griffin, J. M. (2005)
**"Informational Content of Option Volume Prior to Takeovers."** *Journal of Business*, 78(3), 1073–1109.

Documents that abnormal call-option volume spikes 1–5 days before takeover announcements, and these spikes are predictive of the deal direction. *Why for us:* The empirical anchor for the "insider detection" use case. The signal pattern they document — short-dated OTM call volume burst — is exactly what the flow engine should flag.

### Augustin, P., Brenner, M., & Subrahmanyam, M. G. (2019)
**"Informed Options Trading prior to Takeover Announcements: Insider Trading?"** *Management Science*, 65(12), 5697–5720.

Analyzes 1,859 U.S. takeovers (1996–2012). Finds 25% have positive abnormal option volume pre-announcement, concentrated in short-dated OTM calls. *Why for us:* Modern, rigorous version of Cao et al. Gives concrete abnormal-volume thresholds and the rough base rate (25% of events are detectable) — important for calibrating the expected hit rate of the flow engine.

### Chesney, M., Crameri, R., & Mancini, L. (2015)
**"Detecting Abnormal Trading Activities in Option Markets."** *Journal of Empirical Finance*, 33, 263–275.

Develops two statistical detectors for abnormal option trades, the first using only ex-ante information: unusually large daily increments in **open interest** that are close to daily trading volume (i.e., the volume is opening new positions, not closing). *Why for us:* This is the most operationally useful paper for us. The specific feature — `Δ open interest / daily volume ≈ 1` — is directly implementable on data we can get free.

### Hu, J. (2014)
**"Does Option Trading Convey Stock Price Information?"** *Journal of Financial Economics*, 111(3), 625–645.

Decomposes option volume into the option-induced order imbalance in the underlying. *Why for us:* Provides a sharper version of the Pan–Poteshman put/call signal by accounting for delta hedging.

---

## Part 4 — 0DTE-specific microstructure

### Adams, G., Fontaine, J.-S., & Ornthanalai, C. (2024)
**"The Market for 0DTE: The Role of Liquidity Providers in Volatility Attenuation."** SSRN 4881008.

Using 2019–2023 intraday data, shows market makers *reduce* SPX realized volatility by 60–90 annualized bp on 0DTE days. End-users initiate flow; MMs absorb and hedge in futures. *Why for us:* Counters the common "0DTE is dangerous" narrative and quantifies who's on each side of the trade. Critical context for interpreting flow.

### Božović, M. (2025)
**"Intraday Jumps and 0DTE Options: Pricing and Hedging Implications."** SSRN 5223127.

Builds a stochastic-vol-with-jumps model fit specifically to 0DTE SPX. Finds intraday jumps cluster at open and close and materially affect 0DTE option prices. *Why for us:* If we extract an RND from a 0DTE chain at 10am, ignoring the close-of-day jump regime will underprice tails. The paper gives a calibration recipe.

### Vasquez, A., Amaya, D., Pearson, N. D., & Garcia-Ares, P. A. (2025)
**"0DTE Index Options and Market Volatility: How Large is Their Impact?"** SSRN 5113405.

Examines gamma-squeeze dynamics from 0DTE hedge rebalancing. *Why for us:* Helps interpret intraday SPX moves: when is a move "real" information vs. just dealer gamma unwind? The flow engine should down-weight signals that coincide with documented gamma-squeeze hours.

### Brogaard, J., Han, J., & Won, P. Y. (2024)
**"How Does Zero-Day-to-Expiry Options Trading Affect the Volatility of Underlying Assets?"**

Cross-sectional/event-study evidence on 0DTE's effect on underlying vol. *Why for us:* Background context for the broader literature debate.

---

## Reading order for new contributors

1. Breeden–Litzenberger 1978 (just the result, not the full proof)
2. Figlewski 2010 (how to actually build an RND in practice)
3. Easley–O'Hara 1992 (vocabulary of informed trading)
4. Chesney–Crameri–Mancini 2015 (the practical detector we're going to implement)
5. Adams–Fontaine–Ornthanalai 2024 (what 0DTE actually looks like)

Everything else as you go deeper.
