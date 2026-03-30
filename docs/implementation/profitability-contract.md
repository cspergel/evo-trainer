# Profitability Contract

This document sits above the phase plans. Every feature, layer, signal source, and strategy must satisfy these constraints before shipping to live. No exceptions.

---

## 1. Baseline-Beating Requirement

Nothing ships to live unless it beats the baseline after costs.

**Baseline by strategy class:**
- Directional long-only: Buy-and-hold SPY (or sector ETF equivalent for sector-specific strategies)
- Sector-relative long-only: Sector ETF or equal-weight sector basket for the traded universe
- Long/short or beta-hedged: Zero-cost, factor-matched benchmark with matched net exposure
- Low-net or defensive systems: Cash plus a beta-matched passive benchmark

**Rule:** A strategy is only compared to the benchmark that matches its exposure profile. Raw return comparisons against SPY are not sufficient for hedged, market-neutral, or low-net systems.

**What "beats" means:**
- Higher out-of-sample Sharpe ratio than the baseline over the same period
- Positive executable alpha (see section 2) after all costs
- Measured across at least 3 independent walk-forward windows
- Must beat baseline in at least 2 of 3 windows (not just on average)

**Applies to:**
- Every strategy skill before promotion from paper to live
- Every new system layer before being enabled in production
- The meta-selector's combined output vs simply holding the baseline

**Signal sources are evaluated differently:**
- A signal source does not need to "beat SPY" on its own
- A signal source must show marginal lift versus the same system without that source
- Required lift metrics: Sharpe improvement, drawdown improvement or no degradation, turnover impact, and hit-rate contribution
- If a source does not produce measurable marginal lift after costs, it stays in observation or is removed

**How to enforce:**
- `BaselineComparator` module computes SPY buy-and-hold returns for any evaluation period
- `BaselineComparator` also supports sector, beta-matched, and low-net benchmark modes
- Every `StrategyPerformance` result is paired with its matched baseline performance
- Fitness evaluation subtracts baseline returns before computing Sharpe
- Dashboard shows strategy returns vs baseline side by side

---

## 2. Executable Alpha Only

We measure alpha after everything that erodes it.

**Costs that must be modeled:**
- **Spread:** Half the bid-ask spread on entry and exit (use historical spread data or conservative estimate: 0.02% for large-cap, 0.10% for mid-cap)
- **Slippage:** Market impact estimate based on order size relative to average daily volume (0.05% for small orders in liquid names, scale up for larger)
- **Commission:** Per-share or per-trade fee (Alpaca: $0 commission, but SEC fee + FINRA TAF still apply)
- **Delay:** Signal-to-execution delay. Congressional signals are 30-45 days stale at filing. 13F signals are 45 days stale. Price at execution, not at signal generation.

**Implementation:**
- `ExecutionCostModel` applied in the composition layer between sizing and constraint check
- Every `SizingResult` has a `cost_estimate` field
- Every `TradeResult` records actual vs estimated costs
- Reject strategies whose expected edge is less than 2x the estimated round-trip cost

---

## 3. Minimum Statistical Bar

No strategy is promoted on headline Sharpe alone. The sample must be large enough to mean something.

**Minimum requirements:**
- At least 3 independent walk-forward windows
- At least 30 completed trades per window for active strategies, or 90 completed trades total for lower-frequency strategies
- At least 2 distinct market regime labels represented across the full evaluation set
- No single window may account for more than 50% of total PnL
- Report Sharpe, max drawdown, turnover, win rate, and bootstrap confidence intervals together

**Required anti-false-positive checks:**
- Deflated Sharpe ratio or equivalent post-selection-adjusted statistic
- Bootstrap or block-bootstrap confidence intervals for Sharpe and alpha
- Promotion is blocked if confidence intervals meaningfully overlap zero alpha after costs
- Promotion is blocked if results are driven by too few names, too few dates, or one event cluster

**Interpretation rule:**
- "Looks good" is not enough
- If the sample is sparse, concentrated, or regime-thin, the strategy stays in research regardless of paper performance

---

## 4. Capacity and Liquidity Constraints

An edge that cannot be entered and exited cleanly is not a live edge.

**Required liquidity filters:**
- Trade only instruments that satisfy minimum average daily dollar volume and maximum spread thresholds for the current phase
- Initial live scope remains US large-cap equities specifically to stay inside these liquidity bounds
- No live trading in names where modeled exit liquidity is worse than the strategy holding horizon can tolerate

**Capacity rules:**
- Position sizing must include ADV participation caps
- Default cap: do not exceed 1% of average daily volume for normal execution without explicit justification
- Position size must also pass an exit-capacity check under stressed volume assumptions
- Capacity-adjusted alpha must remain positive after slippage is recalculated at intended size

**Promotion blocker:**
- If a strategy only works at toy size, it is not eligible for live deployment beyond micro-live
- If realized slippage rises materially with size during paper or micro-live, the strategy is frozen or demoted

---

## 5. Multiple-Testing Discipline

The system will generate many candidates. That creates false winners unless selection pressure is penalized explicitly.

**Rules:**
- Every candidate, mutation, parameter set, and prompt family must be logged in an experiment registry
- Total experiment count is tracked per strategy family and per signal family
- Promotion thresholds become stricter as experiment count rises
- The selected candidate must beat not only the baseline, but also a post-selection penalty tied to search breadth

**Required controls:**
- Keep a `ResearchLedger` with candidate id, parent id, hypothesis, evaluation windows, and outcome
- Track the number of discarded variants before a "winner" appears
- Use lower-confidence-bound Sharpe or another conservative post-selection metric for promotion
- Do not allow silent cherry-picking from incubator or evolution outputs

**Interpretation rule:**
- The more things tested, the less we trust the best backtest
- A candidate discovered late in a wide search needs stronger evidence than a candidate from a narrow, hypothesis-driven search

---

## 6. Narrow Initial Scope

Complexity is the enemy of a first edge. Start narrow, earn the right to expand.

**Constraints until first profitable quarter:**

| Dimension | Constraint |
|-----------|-----------|
| Market | US large-cap equities only (S&P 500 constituents) |
| Holding horizon | Days to weeks (not intraday, not months) |
| Signal family | Congressional + EDGAR 13F + Form 4 clusters + simple regime filter |
| Position style | Sector-relative long-only or low-turnover long/short |
| Active strategies | Maximum 3 simultaneously (plus capital preservation) |
| Active signal sources | Maximum 3 simultaneously |
| Evolution | Freeze after initial seed tuning; unfreeze only with evidence |

**Why these constraints:**
- Large-cap = liquid, low spread, low slippage, reliable data
- Days-to-weeks = matches signal staleness (congressional filings are not intraday signals)
- Few strategies = auditable, debuggable, attributable
- Few sources = provable incremental value per source

**Relaxation criteria:**
- First profitable quarter after costs on paper trading
- Paper/live correlation > 0.8 for 60 days
- Then add ONE dimension at a time (one new signal source OR one new strategy family OR extend to mid-cap)

---

## 7. Simplicity Tax

Each layer must prove it earns its keep.

**The rule:** Before enabling any new component in production, it must demonstrate positive incremental value over the system without that component, measured on out-of-sample data after costs.

**Applies to:**
- Meta-selector vs equal-weight allocation
- Signal scoring vs unweighted signals
- Regime classifier vs "always risk-on"
- Each individual signal source vs the system without it
- Position sizing skill vs fixed 2% allocation
- Orchestrator adjustments vs no orchestrator
- LLM-driven evolution vs the seed strategies alone

**How to test:**
- A/B comparison: run the full system with and without the component
- Measure on the same out-of-sample walk-forward windows
- Component must improve Sharpe by at least 0.1 (not just "positive")
- If it doesn't clear the bar, it stays disabled in live

**Freeze policy:**
- In live mode, run the narrowest configuration that passed validation
- Do not evolve multiple components simultaneously
- When testing a change, freeze everything else
- The orchestrator is advisory-only until it proves causal value (not just correlation)

---

## 8. Champion/Challenger Framework

Every change competes against the current best.

**Champion:** The currently running live configuration (strategies, signals, sizing, parameters).

**Challenger:** Any proposed change (new strategy, evolved skill, new signal source, parameter update).

**Rules:**
- Challenger must beat champion on at least 3 independent walk-forward windows
- Challenger is paper-traded for minimum 20 trading days before live consideration
- If challenger's paper performance diverges from OOS validation by more than 1 standard deviation, it is killed immediately
- Only one challenger runs at a time (no simultaneous experiments in live)
- Track total experiments run; penalize the system's complexity budget proportionally

**Promotion path (unchanged from Phase 6 spec, but with added gates):**
1. Walk-forward validation (OOS Sharpe > baseline + 0.1)
2. Paper trading (20+ days, paper/live correlation check)
3. Micro-live (5-10% capital, human approval every trade)
4. Partial-live (25-50% capital)
5. Full-live

**Kill triggers (any one is sufficient):**
- Paper/live Sharpe deviation > 1 std dev for 10 consecutive days
- Realized costs exceed estimated costs by 50%+
- Max drawdown exceeds OOS max drawdown by 50%+
- Strategy edge (after costs) drops below 0 for 30 days

---

## 9. LLM Role Boundaries

LLMs are researchers and assistants, not traders.

**LLMs MAY:**
- Generate structured hypotheses ("if RSI > 70 and insider cluster detected, consider reducing position")
- Classify signals into categories (regime read, conviction, event-driven)
- Summarize trade rationale for human review
- Suggest skill improvements during evolution (FIX/DERIVED)
- Analyze failure modes and suggest which component to investigate
- Generate new seed strategy candidates for the incubator

**LLMs MAY NOT:**
- Make final trade/no-trade decisions without a symbolic rule firing
- Override risk constraints or position limits
- Determine position size (sizing is formula-based: Kelly, vol target, etc.)
- Bypass the promotion pipeline
- Self-approve their own evolved strategies without OOS validation

**Decision path must be:**
1. Signals arrive (structured data, not LLM output)
2. Regime classifier labels the environment (rule-based or validated LLM classification)
3. Meta-selector routes to strategies (weighted by proven scorecards)
4. Strategy generates trade intent (structured: ticker, direction, conditions met)
5. Sizing computes quantity (formula-based)
6. Constraints approve/veto (immutable, non-bypassable)
7. LLM logs rationale summary (after the fact, for human review)

**Benchmarking requirement:**
- Every LLM-driven component must be benchmarked against a non-LLM baseline
- LLM regime classifier vs simple VIX/momentum regime classifier
- LLM evolution vs random perturbation of seed strategies
- LLM failure analysis vs simple heuristic (which we already have)
- If the LLM version doesn't beat the non-LLM version by at least 0.1 Sharpe, use the simpler version

---

## 10. Paper/Live Deviation Tracking

The most dangerous failure mode is "paper looks great, live doesn't."

**Track continuously:**
- Fill price deviation: expected vs actual fill price per trade
- Timing deviation: intended execution time vs actual
- Cost deviation: estimated round-trip cost vs actual
- Return deviation: paper portfolio equity curve vs live portfolio equity curve
- Correlation: rolling 30-day paper/live correlation coefficient

**Alert thresholds:**
- Fill price deviation > 0.1% average → warning
- Cost deviation > 50% of estimate → warning
- 30-day paper/live correlation < 0.8 → auto-demotion to paper-only
- 10-day rolling Sharpe deviation > 1 std dev → kill switch consideration

**This is not optional.** Paper/live deviation is the primary health metric in live mode, above even PnL.

---

## 11. The Practical Path

The system that makes money is not the full 12-phase autonomous evolution engine. It's:

1. **Find one small durable edge** — congressional + insider cluster signals improving timing of sector-relative positions in large-cap equities
2. **Prove it survives costs** — executable alpha positive after spread, slippage, delay
3. **Add automation around research, validation, and risk** — walk-forward, paper trading, constraints
4. **Add adaptive complexity only where it earns its keep** — each layer passes the simplicity tax

Everything else is research infrastructure. Valuable, but not live until proven.

---

## Enforcement

This contract is checked at:
- **Strategy promotion:** Must beat baseline after costs across 3+ windows
- **Signal source activation:** Must show incremental value over system without it
- **New layer enablement:** Must pass simplicity tax A/B test
- **Live deployment:** Must pass champion/challenger framework
- **Ongoing live operation:** Paper/live deviation tracked continuously
- **Promotion statistics review:** Must satisfy minimum sample-size and anti-selection requirements
- **Capacity review:** Must satisfy liquidity and ADV participation limits at intended size

If any check fails, the component is disabled in live. No negotiation, no "let's give it one more week." The system protects capital by default.
