# Evolve-Trader AI — Master Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a self-evolving autonomous trading system that combines OpenSpace's skill evolution engine with AI-Trader's trading environment, augmented by multi-source intelligence signals, regime detection, and walk-forward validation — progressing from research prototype to live-capital deployment through graduated phases.

**Architecture:** Four-layer system (Strategy Skill Library → Signal Intelligence Layer → Regime Classification Engine → Meta-Strategy Selector) with a meta-evolution orchestrator sitting above all layers. Strategies are natural-language SKILL.md files that undergo FIX/DERIVED/CAPTURED evolution cycles. A strategy incubator continuously discovers new approaches via mutation, crossover, academic paper mining, anomaly detection, counter-strategy generation, regime-conditioned search, and meta-strategy discovery. Signal sources are dynamically weighted via rolling scorecards. Immutable risk constraints (5% position, 25% sector, 20% drawdown) are never subject to AI override.

**Tech Stack:**
- **Trading Environment:** AI-Trader (Python, REST API + WebSocket) — HKUDS, MIT licensed
- **Skill Evolution Engine:** OpenSpace (Python, MCP server) — HKUDS, MIT licensed
- **LLM Backbone:** Claude Sonnet / GPT-4o / Qwen (configurable via LiteLLM + OpenRouter)
- **Data Storage:** SQLite (Phase 0-1) → PostgreSQL + TimescaleDB (Phase 2+)
- **Signal Ingestion:** FastAPI microservices per source, typed SignalEvent objects
- **Dashboards:** React + Next.js + Tailwind (extending OpenSpace frontend)
- **Brokerage:** Alpaca (primary), IBKR (future), ccxt (crypto)
- **Notifications:** Slack, Telegram, email, dashboard push
- **CI/Testing:** pytest, GitHub Actions
- **Logging/Observability:** structured JSON logging, centralized log aggregation
- **Containerization:** Docker + Docker Compose (dev), container orchestration (prod)

**Reference Document:** `plan/Evolve-Trader_Project_Plan.docx` — the original vision document. This implementation plan operationalizes every element from that document.

**Individual Phase Plans:** Each phase below links to a detailed day-by-day plan file in this directory.

---

## Phase Overview

| Phase | Name | Dependencies | Key Deliverable |
|-------|------|-------------|-----------------|
| 0 | Exploration & Foundation Setup | None | Running dev environment with both codebases understood and documented |
| 1 | Core Evolution Loop | Phase 0 | Strategy skills evolving via FIX/DERIVED/CAPTURED against Evolve-Trader replay harness |
| 2 | Data Persistence & Signal Foundation | Phase 1 | PostgreSQL, SignalEvent framework, EDGAR + congressional signal ingestion |
| 3 | Meta-Selector & Signal Intelligence | Phase 2 | Regime-to-strategy routing, signal scoring/weighting, conflict resolution |
| 4 | Position Sizing & Risk Evolution | Phase 3 | Evolvable sizing skills, composition interface, portfolio risk constraints |
| 5 | Dashboards (v1) | Phase 2 (skeleton), Phase 4 (full read-only) | Ops monitoring + user-facing trading dashboard with operator visibility and non-destructive controls |
| 6 | Paper Trading & Notifications | Phase 5 | Alpaca paper trading, 3-gate execution, trade notifications, approval workflow |
| 7 | Orchestrator | Phase 6 | Meta-evolution agent with cross-layer coordination and counterfactual replay |
| 8 | Strategy Incubator | Phase 7 | Concurrent tournament with mutation, crossover, regime-conditioned search |
| 9 | Signal Expansion | Phase 3, 6 (Phase 7 optional tuning) | Prediction markets, options, on-chain, investor letters, discovery engine |
| 10 | Crowding, Contrarian & Synthetic Benchmarks | Phase 8, 9 | Crowding detection, contrarian skills, distributional validation, exploratory scenario packs |
| 11 | Live Trading & Hardening | Phase 10 | Alpaca live execution, kill switch, security hardening, promotion pipeline operationalized |
| 12 | Extensions & Polish | Phase 11 | Crypto regimes, IBKR, prediction market trading, adversarial robustness |

### Parallelization Opportunities
The dependency chain is mostly sequential (0→1→2→3→4→5→6→7→8), but several phases can overlap:
- **Phase 5 (Dashboards) can start a basic monitoring view during Phase 2-3.** A lightweight dashboard showing evolution events and signal ingestion status is invaluable during development. Full dashboard completion still gates on Phase 4, but skeleton + portfolio health panel can come earlier.
- **Phase 9 (Signal Expansion) can run in parallel with Phase 7-8 once Phases 3 and 6 are complete.** Core source integrations depend on the SignalEvent framework, scoring lifecycle, and paper-trading feedback loop. Phase 7 is not required to start Phase 9, but it later tunes discovery aggressiveness and promotion thresholds.
- **Phase 10 depends on both Phase 8 AND Phase 9** — this is the true convergence point where everything must work together.
- **Individual signal sources within Phase 9 are independent of each other** and can be built in parallel.

---

## Cross-Cutting Concerns (Apply to Every Phase)

### Immutable Risk Constraints
Implemented in Phase 1 and **never relaxed by any subsequent phase:**
- Maximum 5% of portfolio in any single position
- Maximum 25% in any single sector
- Maximum 20% portfolio drawdown before forced de-risking to Capital Preservation skill
- No evolved skill can remove, relax, or override these constraints
- These are human-controlled circuit breakers — the AI evolves everything else

### Testing Strategy (Every Phase)
- **Unit tests:** Every new module, function, and class
- **Integration tests:** Cross-module interactions at phase boundaries
- **Regression tests:** Ensure new phases don't break prior functionality
- **Financial validation tests:** Sharpe ratio, drawdown, and return calculations verified against known datasets
- **Evolution tests:** Verify FIX/DERIVED/CAPTURED produce expected skill mutations
- **TDD discipline:** Write failing test → implement → verify pass → commit

### Version Control & Commits
- Feature branches per phase
- Frequent small commits (every completed task)
- PR-based merge to main at phase completion with code review
- Semantic commit messages: `feat:`, `fix:`, `test:`, `refactor:`, `docs:`

### LLM API Cost Management
The system makes LLM calls at multiple layers — per-trade strategy execution, post-execution analysis, evolution cycles (FIX/DERIVED/CAPTURED), orchestrator weekly reviews, incubator candidate generation, regime classification, and signal processing. At scale, this is a significant operational cost.
- **Token usage tracking:** Every LLM call logs model used, input/output tokens, cost, and which system component initiated it. Phase 1 uses a lightweight file/SQLite log (before PostgreSQL exists). Phase 2+ stores in PostgreSQL for analysis.
- **Cost dashboard panel:** Real-time and historical cost breakdown by component (strategy execution, evolution, orchestrator, incubator, signal processing). Added to ops dashboard in Phase 5.
- **Budget controls:** Configurable monthly budget cap. Warning at 80% utilization. Hard stop at 100% (system degrades to existing skills only — no evolution, no new incubator generations, no orchestrator adjustments). Ensures runaway costs can't happen unmonitored.
- **Model tiering by component:** Production strategy execution uses the best available model (Claude Sonnet / GPT-4o). Incubator candidates can use cheaper/faster models (Haiku, local models via LiteLLM). Orchestrator uses the best model (decisions are high-leverage). Evolution FIX/DERIVED can use mid-tier. These assignments are configurable and the cost dashboard shows cost-per-component to inform optimization.
- **Cost-per-trade metric:** Total LLM cost divided by number of trades executed. Tracked over time. If rising without corresponding performance improvement, signals inefficiency.

### Deployment Architecture
Development happens locally; production runs in the cloud. Designed for this split from Phase 0.
- **Local development:** Docker Compose orchestrating all services (PostgreSQL, FastAPI signal microservices, dashboard dev server, mock Alpaca endpoint). Single `docker compose up` to spin up the full stack. Hot-reload for Python and React code.
- **Cloud production:** Containerized deployment (Docker) on a cloud provider (evaluate during Phase 0: AWS ECS/Fargate, GCP Cloud Run, or a simple VPS with Docker Compose for initial deployment). Key requirement: reliable uptime during US market hours (9:30 AM - 4:00 PM ET, plus extended hours if enabled).
- **Infrastructure-as-code:** Terraform or Pulumi for cloud resources (database, compute, networking). Committed to repo. Reproducible environments.
- **Service architecture:** Monolith-first for Phase 0-4 (single Python process + PostgreSQL + React frontend). Split into services only when needed: signal ingestion microservices become separate containers when we need independent scaling/failure isolation (Phase 2+). The dashboard is always a separate service (Next.js).
- **Environment parity:** Docker ensures dev and prod run identical dependencies. Environment variables for configuration (API keys, database URLs, Alpaca endpoints). Never different behavior between local and cloud beyond the Alpaca paper/live endpoint switch.
- **Cloud provider decision:** Deferred to Phase 0 based on cost analysis. Initial estimate: a modest VPS ($20-50/month) can run the system through Phase 6. Cloud scaling only needed when incubator (Phase 8) demands more compute.

### Logging & Observability
Every autonomous decision the system makes must be auditable. This is non-negotiable for a trading system.
- **Structured JSON logging:** All application logs in structured JSON format with consistent fields: timestamp, level, component, event_type, and context-specific data. No unstructured print statements.
- **Log levels:** DEBUG (signal-processing details, compact model I/O metadata, validation traces), INFO (trade executions, evolution events, regime changes), WARNING (constraint proximity alerts, API rate limits, signal source degradation), ERROR (failed trades, API failures, data integrity issues), CRITICAL (kill switch triggers, constraint breaches, system failures).
- **Audit trail:** Every trade decision gets a complete audit record: the SignalEvents that were active, the RegimeLabel, which strategy skill was selected and why, the sizing skill applied, all three gate results, and the final outcome. Store structured rationale summaries and feature attribution, not raw prompt/response transcripts or unrestricted chain-of-thought text. This is separate from application logs — it's a dedicated audit table in PostgreSQL.
- **Log aggregation:** In production, logs ship to a centralized service (evaluate: Grafana Loki, CloudWatch, or simple file rotation with logrotate for initial deployment). Searchable by component, time range, and event type.
- **Alerting:** Critical and error logs trigger immediate notifications via the same channels as trade notifications (Slack, Telegram, email). Dashboard shows a system health log feed.

### Data Backup & Recovery
The PostgreSQL database holds irreplaceable data: evolution history, trade logs, fossil record, signal source scorecards, and the entire skill lineage DAG.
- **Automated backups:** Daily full database backup + continuous WAL archiving for point-in-time recovery. Backups stored off-host (cloud storage bucket).
- **Retention policy:** Daily backups retained for 30 days. Weekly snapshots retained for 6 months. Monthly snapshots retained indefinitely (evolution history is the system's institutional memory).
- **Recovery testing:** Quarterly test of backup restoration to verify recoverability. Documented runbook for disaster recovery.
- **Skill library backup:** SKILL.md files are in git (version-controlled by default). The version DAG metadata in PostgreSQL is backed up with the database. Between git and database backups, full skill lineage is recoverable.
- **Export capability:** One-command export of all data to portable format (JSON/CSV) for analysis outside the system or migration to a different database.

### Signal Source Error Handling
Each signal source is an external dependency that can fail independently. The system must degrade gracefully.
- **Per-source health monitoring:** Track API response times, error rates, and last successful fetch per signal source. Dashboard panel shows source health status (green/yellow/red).
- **Fallback behavior per source:** If a source is unavailable, its signals are simply absent — the meta-selector works with whatever signals are available. No source is required for the system to operate. The strategy library and regime classifier function on whatever information is present.
- **Graceful degradation test:** Integration test that kills all signal feeds and verifies the system continues to operate using only the strategy library and Capital Preservation skill. This test runs in every phase from Phase 3 onward (Phase 3 is when the meta-selector exists to redistribute weights; Phase 2 only has individual signal sources without routing logic).
- **API change detection:** Each signal source parser validates the response schema on every fetch. Schema mismatch → source marked as unhealthy → alert fired → human investigates. Prevents silent data corruption.
- **Rate limit management:** Each source has a configured rate limit with exponential backoff on 429 responses. Rate limit headroom tracked in monitoring.
- **Fallback chains:** For critical data (e.g., EDGAR filings), multiple access paths are configured: primary API → bulk download fallback → cached last-known data. At least two access paths for every Tier 1 signal source.

### Monitoring Metrics (Cumulative)
Each phase adds metrics to a growing monitoring infrastructure. By the end, the system tracks:
- Portfolio Health: total return, Sharpe ratio, max drawdown, cash deployment rate
- Strategy Evolution: active skill count, evolution event rate, FIX/DERIVED/CAPTURED distribution, skill turnover, version DAG lineage
- Signal Source Performance: rolling hit rate, executable alpha, disclosure impact trend, source independence score, signal magnitude accuracy
- Discovery Pipeline: candidates in pipeline, observation-stage count, promotion/demotion rate
- Cross-Layer Health: inter-layer correlation matrix, counterfactual deltas, orchestrator adjustment log
- Risk Constraints: constraint proximity per limit, position concentration heatmap, sector exposure
- Incubator: population size, generation number, fitness distribution, graduation rate, diversity metrics
- Paper-to-Live: rolling 60-day Sharpe, rolling 30-day drawdown, paper-to-live correlation
- Trading Activity: unnecessary trade rate (during quiet markets), cash deployment rate (<10% = overtrading, >50% = too conservative)
- LLM Costs: total spend, cost per component, cost per trade, monthly budget utilization
- Signal Source Health: per-source API response time, error rate, last successful fetch, schema validation status
- Initial Roster Retention Rate: % of original famous-name roster still in active tier (if >80% after 6 months, not demoting aggressively enough)

---

## Phase 0: Exploration & Foundation Setup
**Detailed plan:** [`docs/plans/phase-0-exploration.md`](phase-0-exploration.md)

### Goal
Clone, run, and deeply understand both OpenSpace and AI-Trader codebases. Set up the Evolve-Trader repository with dev environment, CI, testing framework, and project structure. Document verified integration points — not assumed from docs, but confirmed from reading actual code.

### Key Activities
- Clone OpenSpace and AI-Trader repos; get both running locally
- Read and document OpenSpace's skill evolution internals: how SKILL.md files are structured, how the FIX/DERIVED/CAPTURED modes work in code, how the post-execution analyzer evaluates outcomes, how the version DAG tracks skill lineage, how quality monitoring gates promotion
- Read and document AI-Trader's trading internals: how the REST API works, how market data is served (real-time paper trading, not historical replay), how the portfolio engine tracks positions and P&L, how multi-model support works via OpenRouter, what crypto support exists (Hyperliquid, not BITWISE10)
- Identify the concrete integration points: what OpenSpace APIs/interfaces will Evolve-Trader call, what AI-Trader APIs/interfaces will it call, where do we need adapters
- Set up the Evolve-Trader repo: monorepo structure, Python environment (pyproject.toml), linting (ruff), formatting (black), type checking (mypy), testing (pytest), CI (GitHub Actions)
- Create the initial project skeleton: directory structure for all planned modules (even if empty)
- Write a technical integration document summarizing findings

### Deliverables
- [ ] Both codebases running locally with all tests passing
- [ ] Integration analysis document (`docs/integration-analysis.md`)
- [ ] Evolve-Trader repo initialized with CI pipeline passing
- [ ] Project skeleton with directory structure for all modules
- [ ] Dockerfile + docker-compose.yml for local development (PostgreSQL, Python app, placeholder services)
- [ ] Data source evaluation document: document public data sources (EDGAR, House Clerk, Senate eFD, Capitol Trades, yfinance) and build-our-own scraper strategy.
- [ ] Cloud provider cost estimate (AWS vs. GCP vs. VPS) for production deployment through Phase 6

### Testing
- OpenSpace test suite passes locally
- AI-Trader test suite passes locally
- Evolve-Trader CI pipeline runs (even if only linting empty files)
- Integration analysis document reviewed against actual code (not just READMEs)

### Decision Gates
Phase 0 may reveal that our assumptions about these codebases are wrong. Explicit decision points:
- **If OpenSpace's SKILL.md format doesn't support the fields we need** (entry/exit logic, regime, performance characteristics): Extend the schema ourselves. OpenSpace is MIT-licensed — we can fork and modify. Document what we changed and why.
- **AI-Trader has no historical replay capability** (it is a paper trading simulator, not a backtest engine): Build our own replay harness using historical market data from yfinance / Hyperliquid. AI-Trader's price fetcher can be used as a data source adapter, but replay orchestration, walk-forward validation, and anti-look-ahead enforcement are entirely Evolve-Trader's responsibility.
- **If OpenSpace's post-execution analyzer is tightly coupled to binary success/failure**: Write an adapter layer that maps our financial metrics (Sharpe, drawdown, etc.) into OpenSpace's evaluation interface, or replace the analyzer entirely while keeping the evolution engine.
- **If the codebases have diverged significantly from their papers/READMEs**: Document the actual state, assess how much of our architecture needs to change, and update this master plan before proceeding to Phase 1. Do not build on incorrect assumptions.
- **If either codebase has unmaintained dependencies or broken CI**: Fix what we can, fork if necessary. Both are MIT-licensed. Budget up to 1 extra week for environment stabilization.

### Risk
- OpenSpace or AI-Trader may have undocumented dependencies or broken tests → budget extra time for environment setup
- The codebases may have changed since the papers were published → verify current state matches our assumptions

---

## Phase 1: Core Evolution Loop
**Detailed plan:** [`docs/plans/phase-1-core-evolution.md`](phase-1-core-evolution.md)

### Goal
Get a single trading strategy represented as a SKILL.md executing trades against Evolve-Trader's historical replay harness, with the OpenSpace evolution engine running FIX/DERIVED/CAPTURED cycles. Validate that the core thesis works: strategies evolve meaningfully based on trading outcomes, not just binary success/failure.

### Key Activities

#### StrategySkill Schema
Define the SKILL.md template for trading strategies:
- Entry logic (conditions that trigger a buy/sell)
- Exit logic (stop-loss, take-profit, trailing stop, time-based exit)
- Position sizing defaults (before Phase 4 separates this into its own skill family)
- Target regime (which market conditions this strategy is designed for)
- Expected performance characteristics (target Sharpe, acceptable drawdown, expected win rate)
- Risk parameters (max position size within this strategy's scope)

#### Post-Execution Analyzer (Trading-Specific)
Replace OpenSpace's binary success/failure evaluation with a financial metrics engine. Use `quantstats` (~70 built-in metrics + HTML tearsheets) and `empyrical` (lightweight programmatic metrics) rather than building from scratch:
- Sharpe ratio computation per strategy per evaluation period (`quantstats.stats.sharpe()`)
- Maximum drawdown tracking (`quantstats.stats.max_drawdown()`)
- Win rate (% of trades with positive return)
- Average return per trade
- Monte Carlo robustness scoring (`quantstats` built-in)
- **Distributional evaluation:** mean, variance, skewness, kurtosis, tail risk — not just point estimates
- **Failure tracing:** When a strategy underperforms, trace backwards — was the entry logic wrong? The exit logic? The regime assumption? This trace determines whether the evolution engine applies FIX to the strategy itself or flags the regime classifier (in later phases)
- **LLM reasoning attribution:** Log which parts of the SKILL.md the LLM agent referenced in its trade decisions, enabling analysis of which skill components are actually being used vs. ignored

#### Walk-Forward Validation Harness
Reference architectures (both MIT licensed): `DanRedelien/futures-backtesting-engine` (no-lookahead 6-phase bar loop, FastBar numpy 70x speedup, Optuna walk-forward) and `zachisit/july-backtester` (walk-forward overfitting detection, Monte Carlo robustness, SQN/R-Multiple, multiprocessing). Market data via `yfinance`.
- Evolve strategies on period N (training window)
- Validate on period N+1 (out-of-sample window)
- Gate promotion: only strategies with positive risk-adjusted returns on out-of-sample data get promoted
- No-lookahead enforcement: signal at close[t], execute at open[t+1]
- Overfitting detection: flag strategies where in-sample > 0 but out-of-sample < 0, or degradation > 75%
- Configurable window sizes (default: 30-day train, 10-day validate)

#### Capital Preservation ("Do Nothing") Skill
- Explicit skill that holds cash and makes no trades
- Wins strategy selection when confidence is below threshold (default: 0.6)
- Wins when signal source conflicts are unresolved (in later phases)
- The confidence threshold is itself evolvable by the evolution engine
- Critical: without this, the system always deploys some strategy, introducing unnecessary risk

#### Immutable Risk Constraints
Hardcoded outside the evolution engine:
- Maximum 5% of portfolio in any single position
- Maximum 25% in any single sector
- Maximum 20% portfolio drawdown → forced de-risk to Capital Preservation
- No evolved skill can remove, relax, or override these
- **Monitoring metric:** Constraint Proximity — how often the system bumps against limits

#### Stochastic Fitness Evaluation
- Compare return distributions, not single numbers
- A strategy with Sharpe 1.2 ± 0.8 is less fit than one with Sharpe 0.9 ± 0.2
- Use Kolmogorov-Smirnov or similar distributional tests
- Evolution engine selects for consistency, not just raw returns

#### Complexity Penalties
- Skills referencing specific tickers → penalty
- Skills with narrow date ranges → penalty
- Highly parameterized entry conditions → penalty
- Simpler, generalizable reasoning frameworks preferred (regularization for strategies)

#### Version DAG for Skill Lineage
- Track parent → child relationships through FIX/DERIVED/CAPTURED events
- Record the market conditions and performance metrics that triggered each evolution event
- Enable analysis of which evolutionary paths produce strong vs. weak strategies
- Uses/extends OpenSpace's existing version DAG infrastructure

#### Seed Strategy Library
10-15 hand-crafted strategies covering:
- Trend-following (riding momentum in established trends)
- Mean-reversion (buying dips in range-bound markets)
- Momentum (sector rotation based on relative strength)
- Value (fundamental undervaluation signals)
- Earnings-driven (pre/post earnings announcement strategies)
- Defensive/cash (Capital Preservation + low-volatility approaches)
- Each seed strategy as a complete SKILL.md with all schema fields populated

#### Cold-Start Experiments
- 50 trading days of NASDAQ 100 historical simulation
- Measure whether the evolution engine produces viable adaptations
- Track: number of FIX/DERIVED/CAPTURED events, fitness trajectory over generations, skill library growth
- Success criteria: at least one evolved strategy outperforms its seed parent on out-of-sample data

### Deliverables
- [ ] StrategySkill SKILL.md schema defined and documented
- [ ] Post-execution analyzer computing financial metrics + distributional evaluation
- [ ] Walk-forward validation harness with configurable windows
- [ ] Capital Preservation skill implemented
- [ ] Immutable risk constraints enforced at trade-execution level
- [ ] Complexity penalty system integrated into fitness evaluation
- [ ] Version DAG tracking skill lineage
- [ ] 10-15 seed strategies written as SKILL.md files
- [ ] Cold-start experiment results documented
- [ ] Lightweight LLM token usage logger (file/SQLite-based, migrates to PostgreSQL in Phase 2)
- [ ] All components unit tested and integration tested

### Testing
- Post-execution analyzer verified against hand-computed metrics on known trade sequences
- Walk-forward harness verified: strategy evolved on training data cannot see validation data
- Capital Preservation wins selection when confidence < threshold (parameterized test)
- Risk constraints block trades that would violate limits (boundary tests)
- Complexity penalties reduce fitness of overly-specific strategies
- Version DAG correctly records evolution events and lineage
- Cold-start produces at least one FIX and one DERIVED event (proves evolution loop works)

---

## Phase 2: Data Persistence & Signal Foundation
**Detailed plan:** [`docs/plans/phase-2-data-signals.md`](phase-2-data-signals.md)

### Goal
Migrate from SQLite to PostgreSQL, build the signal ingestion framework, and integrate the first three signal sources: SEC EDGAR 13F filings, SEC EDGAR Form 4 insider transactions, and one congressional trading source. Implement per-source signal decay functions. Build the basic regime classifier as an evolvable SKILL.md.

### Key Activities

#### Database Migration
- Design PostgreSQL schema for: trade logs, signal events, evolution events, strategy skills metadata, monitoring metrics, portfolio snapshots
- Data access layer with repository pattern — all database access goes through typed interfaces, making future migrations (e.g., adding TimescaleDB) non-breaking
- Migration tooling (Alembic) for schema versioning
- Migrate existing SQLite data from OpenSpace skill DB and AI-Trader trade logs
- Time-series considerations: trade prices and monitoring metrics will eventually move to TimescaleDB (PostgreSQL extension), so design the schema with that in mind

#### SignalEvent Framework
Define the typed signal interface that ALL signal sources must produce:
```
SignalEvent:
  source: str                    # e.g., "edgar_13f", "capitol_trades", "ark_daily"
  source_entity: str             # e.g., "Warren Buffett", "Nancy Pelosi"
  timestamp: datetime            # when the signal was generated
  trade_date: datetime | None    # when the actual trade occurred (if applicable)
  filing_date: datetime | None   # when the filing was made public
  confidence: float              # 0.0 - 1.0
  decay_profile: DecayProfile    # per-source decay function
  signal_type: SignalType        # enum: REGIME_READ, CONVICTION, EVENT_DRIVEN, THESIS
  payload: dict                  # source-specific structured data
  metadata: dict                 # additional context (committee, sector, etc.)
```

#### Signal Decay Functions
Per-source decay profiles (evolvable starting parameters):
- Buffett 13F: High initial confidence, ~90-day half-life, slow linear decay
- Form 4 insider clusters: High confidence, ~30-day half-life, moderate linear decay
- Congressional disclosures: Medium confidence, ~20-day half-life, moderate exponential decay
- ARK daily trades: Medium confidence, ~10-day half-life, fast exponential decay (Phase 9)
- On-chain whale moves: Medium-high confidence, ~3-day half-life, fast exponential (Phase 9)
- Options unusual activity: High confidence, ~2-day half-life, very fast exponential (Phase 9)
- Fed language / macro news: Variable confidence, ~45-day half-life, slow step-function (Phase 9)
- Prediction markets: Continuous repricing, no traditional decay — always current (Phase 9)

#### SEC EDGAR 13F Parser
Use `pibou-filings` PyPI package ([github.com/Pierre-Bouquet/pibou-filings](https://github.com/Pierre-Bouquet/pibou-filings)) — actively maintained (March 2026), parses 13F-HR and Section 16 (Form 3/4/5) XML into structured CSVs/DataFrames. Has built-in EDGAR rate-limiting and parallel downloads via ThreadPoolExecutor.
- Extract: filer CIK, filing date, report period, holdings (issuer, CUSIP, value, shares, investment discretion, voting authority)
- Add CUSIP-to-ticker mapping layer via OpenFIGI API (free) for actionable signals
- Use `py-sec-edgar` RSS workflow ([github.com/ryansmccoy/py-sec-edgar](https://github.com/ryansmccoy/py-sec-edgar)) for real-time filing alerts
- Automated quarterly polling with rate limiting (EDGAR: 10 requests/second)
- Bulk download support for historical backfill
- Named manager watchlist: Buffett (Berkshire), Dalio (Bridgewater), Ackman (Pershing Square), Soros, Druckenmiller (Duquesne), Burry (Scion), Chase Coleman (Tiger Global), Tepper (Appaloosa), Klarman (Baupost), Howard Marks (Oaktree), Dan Loeb (Third Point), Carl Icahn, Baker Brothers, Abdiel Capital
- Produce SignalEvent objects with signal_type=REGIME_READ for sector tilts and CONVICTION for position changes

#### SEC EDGAR Form 4 Parser
- Parse Form 4 XML filings for insider transactions
- Extract: reporting owner, issuer, transaction date, transaction code (P=purchase, S=sale), shares, price
- **Cluster detection:** Flag when 3+ insiders at different companies within the same sector file purchases within a 2-week window — this is a sector-level signal more robust than any individual trade
- Produce SignalEvent objects with signal_type=CONVICTION
- 2-4 day latency from trade to filing

#### Congressional Trading Source
All congressional trading data is free public record (STOCK Act). Use existing open-source tooling where possible.
- **House data:** Use the `congressional-trading` PyPI package ([github.com/ivanma9/CongressionalTrading](https://github.com/ivanma9/CongressionalTrading)) as primary House scraper. Python 3.12+, FastAPI, scrapes House Clerk ZIP indexes → XML → PDF via pdftotext. Already has rate limiting, circuit breaker, retry logic. Either use as dependency or adapt its scraping pipeline.
- **Senate data:** Build a Senate eFD scraper (efdsearch.senate.gov) — smaller scope (~100 senators), needs Selenium for terms checkbox
- **Pre-normalized backup:** Capitol Trades (capitoltrades.com) scraping for cross-validation — they've already parsed PDFs into clean HTML with 34K+ trades, ~200 politicians
- **Committee enrichment:** ProPublica Congress API (free) for member → committee mapping
- Extract: member name, party, state, trade_date, filing_date, ticker, transaction type, size
- Committee membership context (e.g., Gottheimer on Intelligence buying defense stocks)
- Leadership role tracking (Speaker, Leaders, Whips, Committee Chairs — the Wei & Zhou inflection point)
- Named watchlist: Pelosi, Crenshaw, Wyden, Gottheimer, Greene, Tuberville, Mullin, Davidson, Norcross, Rick Scott
- Produce SignalEvent objects with source metadata including party, state, committees, leadership_role
- Daily polling is sufficient — filings are delayed 30-45 days from trade date

#### Data Source Strategy
All primary data comes from free public sources. No paid APIs required.

**Market prices:**
- `yfinance` (Python package): historical + real-time stock prices, dividends, splits. No API key. Primary source.
- Alpha Vantage free tier (25 req/day): intraday backup
- Hyperliquid public API: crypto prices, no auth

**SEC EDGAR (free, no auth, 10 req/sec):**
- 13F quarterly holdings (XML/SGML)
- Form 4 insider trades (XML)
- Filing stream for real-time notifications

**Future phase packages:**
- `polymarket-apis` (PyPI): Polymarket integration (Phase 9)
- FRED API: macro/economic indicators (Phase 9)
- `ccxt`: unified crypto exchange API covering 100+ exchanges (Phase 12)

**Optional paid APIs (cross-validation only):**
- Unusual Whales: REST API + MCP server, free tier, congress endpoints
- Quiver Quantitative ($30/mo): Python package with `congress_trading()`
- Financial Modeling Prep: free tier (250 calls/day) for fundamentals

#### Basic Regime Classifier
- Initial hand-crafted SKILL.md that consumes SignalEvents and outputs a RegimeLabel
- RegimeLabel structure:
  ```
  RegimeLabel:
    primary_regime: str        # e.g., "risk-off", "risk-on", "transitional"
    sector_bias: str           # e.g., "underweight tech, overweight utilities"
    momentum_state: str        # e.g., "weakening", "strengthening", "transitional"
    confidence: float          # 0.0 - 1.0
    time_horizon: str          # e.g., "short-term (1-4 weeks)", "medium-term (4-12 weeks)"
  ```
- Starts monolithic — later phases allow the evolution engine to decompose it into sub-classifiers
- Subject to the same FIX/DERIVED/CAPTURED evolution as strategy skills

### Deliverables
- [ ] PostgreSQL schema designed, migrated, and documented
- [ ] Data access layer with repository pattern
- [ ] Alembic migration tooling configured
- [ ] SignalEvent type system defined and tested
- [ ] Signal decay function framework with per-source profiles
- [ ] SEC EDGAR 13F ingestion via `pibou-filings` + CUSIP-to-ticker mapping producing SignalEvents
- [ ] SEC EDGAR Form 4 ingestion via `pibou-filings` with cluster detection
- [ ] Congressional trading ingestion producing SignalEvents (House via `congressional-trading` package, Senate via Capitol Trades scraper)
- [ ] Basic regime classifier as evolvable SKILL.md
- [ ] All components unit tested with mock EDGAR data and real historical samples

### Testing
- 13F parser tested against real historical filings (known Berkshire filings with known holdings)
- Form 4 parser tested against known insider transaction filings
- Cluster detection tested: inject 3 synthetic Form 4 buys in same sector within 2 weeks → signal fires
- Congressional source tested against known historical trades (Pelosi NVDA trades as ground truth)
- Decay functions tested: signal confidence decreases correctly over time per profile
- Regime classifier tested: feed known signal combinations → verify expected regime output
- Database migration tested: data integrity verified after SQLite → PostgreSQL migration
- SignalEvent serialization/deserialization round-trip tests

---

## Phase 3: Meta-Selector & Signal Intelligence
**Detailed plan:** [`docs/plans/phase-3-meta-selector.md`](phase-3-meta-selector.md)

### Goal
Build the meta-selector that maps RegimeLabel + SignalEvents to weighted strategy selection. Implement the full signal source scoring/weighting system with rolling scorecards. Build the signal source lifecycle pipeline. Implement conflict resolution and multi-timeframe skill stacking.

### Key Activities

#### Meta-Selector
- Evolvable SKILL.md that acts as a routing function
- Input: current RegimeLabel + active SignalEvents (after decay filtering)
- Output: weighted set of strategy skills with capital allocation percentages
- **Ensemble deployment:** Can activate multiple strategies simultaneously with different allocations — not just pick a single winner
- Subject to FIX/DERIVED/CAPTURED evolution
- When confidence is below threshold or conflicts are unresolved → routes to Capital Preservation

#### Signal Source Scoring & Weighting
Dynamic credibility system replacing static tier assignments:
- **Base Tier Weight:** Starting prior from Tier 1/2/3 classification
  - Tier 1 (congressional leadership, insider clusters, ARK): 3.0x base
  - Tier 2 (active congressional, concentrated value, macro traders): 2.0x base
  - Tier 3 (activist, sector specialist, investor letters): 1.0x base
- **Rolling Hit Rate Multiplier:** Hit rate over last 12 weeks vs. 50% baseline (e.g., 75% = 1.5x multiplier)
- **Regime Alignment Bonus:** +0.5x if source's most recent signal correctly predicted the regime that materialized
- **Cold Streak Penalty:** 0.5x if hit rate falls below 35% over 12 weeks
- **Recency Decay:** More recent signals within lookback window weighted higher
- **Signal Magnitude Accuracy:** When a source makes a large-conviction trade, does size correlate with subsequent return magnitude? Sources whose big bets outperform their small trades are more informative
- **Per-Source Lookback Window Calibration:** Active congressional traders (50 trades/quarter) → 6-week lookback. Buffett (2-3 moves/quarter) → 6-12 month lookback. Minimum 5-trade observation count before rolling scorecard activates — below that, static tier weight applies
- **Disclosure-to-Executable Spread:** Record stock price at disclosure time and at 24/48/72 hours post-disclosure. That gap is the slippage. Evolution engine learns optimal action delay per source (0 days for ARK, ~3 days for Pelosi copycat surge fade, "never" for fully decayed)
- All multiplier formulas and thresholds are evolvable parameters

#### Post-Signal Return Tracking
For each signal source, continuously track:
- Post-signal return at 2, 4, 6, and 12 weeks after disclosed trade
- Alpha vs. S&P 500 or relevant sector index over same period
- Hit rate: % of signals producing positive alpha in lookback window
- Signal magnitude accuracy: correlation between trade size and subsequent return
- Regime alignment score: did the source's trades align with the regime that materialized?

#### Signal Source Lifecycle Pipeline
Staged promotion for all signal sources (named and discovered):
1. **Candidate:** Passes discovery filter thresholds. Tracked but zero weight. Data collection begins.
2. **Observation (1-2 quarters for funds, 6-12 weeks for congressional):** Minimum 5 trade observations. Post-signal returns computed. Still zero weight.
3. **Probation (1 quarter):** Rolling hit rate >50%, positive alpha. Low weight (0.5x Tier 3 base). Signals visible but heavily discounted.
4. **Active (ongoing):** Sustained performance through probation. Survives at least one regime change. Full tier-appropriate weight.
5. **Demotion/Removal:** Rolling hit rate <30% for 2 consecutive lookback periods, or no new trades for 6+ months. Weight zeroed. Remains in database for potential re-promotion.

#### Survivorship Bias Monitoring
- **Initial Roster Retention Rate:** What % of the original famous-name roster remains in the active tier after 6 months? If >80%, the system isn't being aggressive enough at demotion.
- The rolling scorecard and discovery engine (Phase 9) naturally correct for survivorship bias — famous names that don't perform get demoted regardless of reputation

#### Alpha Decay / Popularity Penalty Monitoring
- **Disclosure Impact Trend:** Average stock move in 0-24 hours post-disclosure, tracked per source over time
- If a source's disclosure impact is increasing → market is front-running → residual alpha shrinking
- Evolution engine learns a popularity penalty — widely-tracked sources get downweighted
- Self-correcting: system gravitates toward underfollowed sources where alpha concentrates

#### Signal Source Conflict Resolution
- **Confidence-weighted averaging:** When two sources disagree (e.g., Buffett buys financials, Druckenmiller shorts them)
  - If rolling scorecard weights are similar → signals cancel → default to Capital Preservation
  - If one dramatically outscores the other → higher-weighted source wins
- Evolution engine learns source-pair conflict dynamics over time (e.g., Buffett vs. Druckenmiller conflicts historically resolve in Druckenmiller's favor on 6-month horizon but Buffett's on 2-year)
- **Monitoring metrics:** Conflict Frequency and Conflict Resolution Quality (who was right?)

#### Multi-Timeframe Skill Stacking
Three hierarchical layers with defined interfaces:
- **Strategic (months-years):** Portfolio-level sector tilts, asset class allocation, macro positioning. Sources: Buffett/Dalio 13Fs, investor letters, macro regime classifier. Updated monthly or on major regime change. Outputs constraints like "maximum 20% technology exposure"
- **Tactical (weeks-months):** Individual position entry/exit, sector rotation, earnings plays. Sources: insider transactions, ARK daily trades, evolved momentum/mean-reversion skills. Updated weekly or on catalyst trigger. Must respect strategic layer constraints.
- **Execution (hours-days):** Intraday timing, position sizing optimization, order management. Sources: options unusual activity, real-time news, on-chain whale moves. Updated continuously during market hours. Optimizes within tactical layer decisions.
- Each layer evolves independently but composes through typed interfaces

### Deliverables
- [ ] Meta-selector routing RegimeLabel + signals to weighted strategy set
- [ ] Ensemble deployment: multiple strategies active with capital allocation %
- [ ] Signal source scoring with all weight components (tier, hit rate, regime alignment, cold streak, recency, magnitude accuracy)
- [ ] Per-source lookback window calibration
- [ ] Disclosure-to-executable spread tracking and optimal delay learning
- [ ] Post-signal return tracking at 2/4/6/12 week horizons
- [ ] Signal source lifecycle pipeline (candidate through demotion)
- [ ] Survivorship bias monitoring metric
- [ ] Alpha decay / popularity penalty monitoring
- [ ] Signal conflict resolution with confidence-weighted averaging
- [ ] Multi-timeframe skill stacking with strategic/tactical/execution layers
- [ ] All components unit tested and integration tested with Phase 1-2

### Testing
- Meta-selector tested: given known regime + signals → produces expected strategy weights
- Ensemble deployment: verify capital allocation percentages sum to ≤100%
- Signal scoring: inject known source performance data → verify weight calculations
- Lookback calibration: verify Buffett uses longer window than active congressional traders
- Conflict resolution: inject opposing signals → verify Capital Preservation selected when weights equal
- Multi-timeframe: verify tactical layer respects strategic layer constraints
- Lifecycle pipeline: walk a source through all 5 stages, verify weight changes at each transition
- Disclosure spread: inject known price data at disclosure time → verify spread calculation

---

## Phase 4: Position Sizing & Risk Evolution
**Detailed plan:** [`docs/plans/phase-4-position-sizing.md`](phase-4-position-sizing.md)

### Goal
Separate position sizing into its own evolvable skill family. Build the composition interface where strategy skills output what/when and sizing skills output how much. Implement portfolio-level risk constraint enforcement. Add tax-aware evolution mode.

### Key Activities

#### Evolvable Sizing Skill Family
Position sizing as SKILL.md files that evolve independently from strategy skills:
- **Kelly Criterion variants:** Fractional Kelly sizing based on estimated win probability and payoff ratio. Evolution engine learns optimal Kelly fraction per strategy type.
- **Volatility targeting:** Position sizes scaled inversely to recent realized volatility, ensuring consistent risk contribution per position.
- **Correlation-aware sizing:** Positions in correlated assets jointly sized to prevent concentration risk that appears diversified at security level but is concentrated at factor level.
- **Regime-adjusted sizing:** Regime classifier output directly modulates total portfolio exposure. Risk-off regime might cap gross exposure at 60%; high-conviction bull allows 100%.

#### Composition Interface
- Strategy skills output: ticker, direction, entry conditions, exit conditions, target regime
- Sizing skills output: number of shares/dollars, position size as % of portfolio
- Clean separation: a strategy skill never specifies position size; a sizing skill never specifies what to trade
- The meta-selector pairs each active strategy with an appropriate sizing skill based on regime and portfolio state

#### Portfolio-Level Risk Enforcement
- Real-time portfolio exposure tracking across all active positions
- Sector classification for every holding
- Pre-trade validation: before any trade executes, verify it won't breach immutable constraints
- Post-trade monitoring: if market moves cause a constraint to be breached (e.g., a stock appreciates and exceeds 5%), trigger rebalancing
- Gross exposure limits linked to regime (configurable per regime label)

#### Tax-Aware Evolution Mode
- Configurable flag (off by default)
- When enabled: fitness function penalizes strategies generating short-term capital gains (held <1 year) by applying user's marginal tax rate differential
- Evolution engine naturally drifts toward longer holding periods
- **Monitoring metric:** Tax Drag — difference between pre-tax and after-tax returns, visible even when disabled so user can see cost of rapid trading

#### Paper-Trading Survival Gate
- Evolved skills must demonstrate positive risk-adjusted returns over 20+ trading days before promotion to production library
- Sharpe threshold: 0.5 (configurable)
- Max drawdown threshold: 15% (configurable)
- This gate applies to ALL evolved skills — strategy skills AND sizing skills
- **Before Phase 6 (no Alpaca connection yet):** The gate runs against Evolve-Trader's replay harness on held-out data (walk-forward validation from Phase 1). This is simulated paper trading.
- **After Phase 6 (Alpaca connected):** The gate runs against real-time Alpaca paper trading with live market data. This is real paper trading and is the stronger test.

### Deliverables
- [ ] Kelly, volatility-targeting, correlation-aware, and regime-adjusted sizing skills as SKILL.md
- [ ] Composition interface: strategy outputs what/when, sizing outputs how much
- [ ] Portfolio-level real-time exposure tracking
- [ ] Pre-trade constraint validation
- [ ] Post-trade rebalancing triggers
- [ ] Tax-aware evolution mode with configurable flag
- [ ] Paper-trading survival gate for all skill types
- [ ] All sizing skills unit tested with known portfolios

### Testing
- Kelly sizing: given known win rate and payoff ratio → verify correct fraction
- Volatility targeting: high-volatility asset gets smaller position than low-volatility
- Correlation-aware: two correlated assets sized smaller together than each would be independently
- Regime-adjusted: risk-off regime → reduced gross exposure
- Composition: strategy skill + sizing skill → valid TradeIntent with all fields populated
- Constraint enforcement: trade that would breach 5% limit → blocked
- Tax mode: short-holding strategy penalized more when tax mode enabled
- Survival gate: strategy with Sharpe <0.5 over 20 days → not promoted

---

## Phase 5: Dashboards (v1)
**Detailed plan:** [`docs/plans/phase-5-dashboards.md`](phase-5-dashboards.md)

### Goal
Build both the ops monitoring dashboard and the user-facing trading dashboard. The ops dashboard provides system health visibility. The trading dashboard provides the daily user experience: portfolio view, P&L, trade history, strategy performance, signal explorer, and operator state visibility.

**Early start (during Phase 2-3):** A skeleton dashboard with the Portfolio Health panel and Strategy Evolution panel can be built as soon as PostgreSQL exists (Phase 2). This provides invaluable visibility during development — seeing evolution events and signal ingestion in real-time rather than tailing log files. The full dashboard with all read-only panels completes after Phase 4 when position sizing and risk enforcement exist to power the remaining views. Actionable controls that can approve trades or trigger kill-switch behavior are deferred to Phases 6 and 11.

### Key Activities

#### Ops Monitoring Dashboard
System-health panels built on React + Next.js + Tailwind:
- **Portfolio Health:** Total return, Sharpe ratio, max drawdown, current regime, cash deployment rate. Alerts at drawdown >15% (warning) or >20% (forced de-risk).
- **Strategy Evolution:** Active skills count, evolution event rate, FIX/DERIVED/CAPTURED distribution, skill turnover. Alert on oscillation (same skill FIX'd >3x in 30 days). Version DAG visualization showing skill lineage.
- **Signal Source Performance:** Rolling hit rate per source, executable alpha, disclosure impact trend, source independence score. Alert when any Tier 1 source drops below 40% hit rate for 2 consecutive periods.
- **Discovery Pipeline:** Candidates in pipeline, observation-stage count, promotion/demotion rate. Alert when pipeline empty for >2 quarters.
- **Cross-Layer Health:** Inter-layer correlation matrix, counterfactual deltas, orchestrator adjustment log. Alert when 2+ layers show improving local metrics but declining global performance.
- **Risk Constraints:** Constraint proximity per limit, position concentration heatmap, sector exposure. Alert when any constraint hit >5x in a single week.
- **LLM Costs:** Total spend, cost breakdown by component (strategy execution, evolution, orchestrator, incubator, signal processing), cost per trade, monthly budget utilization. Alert at 80% budget utilization. Tracks model usage distribution.

#### User-Facing Trading Dashboard
The daily interface for the human operator:
- **Portfolio Overview:** Current holdings, total value, daily/weekly/monthly P&L, benchmark comparison (S&P 500)
- **P&L Charts:** Equity curve, drawdown chart, daily returns histogram, rolling Sharpe
- **Trade History:** Searchable/filterable log of all trades with strategy skill, regime, confidence, signal sources that contributed, outcome
- **Strategy Performance:** Per-strategy breakdown showing win rate, Sharpe, total return, number of trades, average holding period. Comparison view across strategies.
- **Signal Source Explorer:** Per-source detail showing rolling hit rate, recent signals, post-signal returns, current weight in meta-selector, lifecycle stage. Drill-down to see individual signals and their outcomes.
- **Active Positions:** Current portfolio with per-position P&L, days held, strategy that opened it, regime when opened, current risk metrics
- **Operator State Surface:**
  - Kill switch status/readiness indicator (Phase 11 wires the action)
  - Pending approvals queue/status (Phase 6 wires the action)
  - System mode display (paper training / paper validation / micro-live / partial-live / full-live)
  - Evolution pause/resume toggle
  - Manual regime override status and audit history

#### Dashboard API Layer
- REST API serving dashboard data from PostgreSQL
- WebSocket connections for real-time updates (trade executions, signal events, alert triggers)
- Authentication (even for local deployment, as a security habit)

### Deliverables
- [ ] Ops monitoring dashboard with all 7 panels (including LLM costs)
- [ ] User-facing trading dashboard with portfolio, P&L, trade history, strategy performance, signal explorer, active positions
- [ ] Operator state surface: kill-switch status, approval queue visibility, mode selector, evolution toggle, manual override status
- [ ] REST API for dashboard data
- [ ] WebSocket for real-time updates
- [ ] Authentication system
- [ ] Responsive design (usable on mobile for quick checks)

### Testing
- Dashboard API endpoints return correct data for known portfolio states
- WebSocket delivers real-time updates within 1 second of event
- Kill switch status UI reflects backend state changes once Phase 11 wiring exists
- Approval queue UI reflects backend state changes once Phase 6 wiring exists
- All dashboard panels render correctly with empty data (cold start)
- All dashboard panels render correctly with large datasets (stress test)

---

## Phase 6: Paper Trading & Notifications
**Detailed plan:** [`docs/plans/phase-6-paper-trading.md`](phase-6-paper-trading.md)

### Goal
Connect to Alpaca paper trading. Implement the TradeIntent object with 3 execution gates. Build the full paper-to-live promotion protocol. Implement the trade notification system with approval workflow across Slack, Telegram, email, and dashboard.

### Key Activities

#### Alpaca Paper Trading Integration
- Connect via `alpaca-py` SDK to paper endpoint (`paper-api.alpaca.markets`)
- Map TradeIntent objects to Alpaca order types (MarketOrderRequest, LimitOrderRequest)
- Real-time position and P&L tracking via Alpaca WebSocket
- Support for fractional shares and extended hours trading
- Up to 3 simultaneous paper accounts (reserve for: primary paper shadow, micro-live validation, incubator candidates)

#### TradeIntent Object & 3 Execution Gates
```
TradeIntent:
  ticker: str
  direction: str              # BUY, SELL, SHORT, COVER
  quantity: float              # shares (can be fractional)
  order_type: str              # MARKET, LIMIT, STOP, TRAILING_STOP
  strategy_skill: str          # which SKILL.md generated this intent
  strategy_lineage: str        # version DAG path (e.g., "Momentum-v1 → DERIVED → Momentum-v3")
  sizing_skill: str            # which sizing SKILL.md determined the quantity
  regime_label: RegimeLabel    # current regime when intent was generated
  signal_sources: list         # which signals contributed, with their current weights
  confidence: float            # 0.0 - 1.0
  rationale_summary: str       # structured, human-readable trade thesis summary
  rationale_evidence: dict     # compact feature attribution / signal evidence, not raw chain-of-thought
  position_impact: dict        # what this trade does to portfolio (% of portfolio, sector exposure)
  paper_track_record: dict     # this strategy's paper performance (trades, win rate, Sharpe)
```

- **Gate 1 — Immutable Risk Constraints (Automatic):** Checks 5% position, 25% sector, 20% drawdown. If violated → trade blocked. Never bypassed.
- **Gate 2 — Paper Trading Shadow (Always On):** Every TradeIntent executes on paper simultaneously, regardless of live status. Creates continuous counterfactual benchmark. Includes human-vetoed trades (tracks "what would have happened"). Never stops running.
- **Gate 3 — Approval Gate (Graduated):** Starts fully manual, graduates based on rolling performance. Auto-approval threshold configurable (default: confidence >0.85 from validated skills).

#### Paper-to-Live Promotion Protocol
5 stages with metrics-driven transitions:
1. **Paper Training (min 90 trading days):** Full system on Alpaca paper. All metrics tracked. Min 50 completed trades. $0 at risk.
2. **Paper Validation (min 60 additional days):** Rolling 60-day Sharpe >0.5, max drawdown <15%, win rate >45%. Clock resets if any threshold breached.
3. **Micro-Live (min 30 days):** 5-10% of capital. Human approval for every trade. Paper shadow continues. Live returns within 2 std devs of paper returns.
4. **Partial-Live (min 60 days):** 25-50% of capital. Approval only for trades >$5K or untested skills. Auto-approval for confidence >0.85 from validated skills.
5. **Full-Live (ongoing):** 100% capital. Auto-approval within constraints. Human notification (not approval). Weekly orchestrator review.
- **Bidirectional:** 60-day Sharpe drops below 0.3 or drawdown >12% → auto-demotion. Progressive through stages. System protects capital by default.
- **Rolling Average Confirmation:** 60-day window for Sharpe, 30-day for drawdown, paper-to-live correlation >0.8.

#### Trade Notification System
Structured notification per TradeIntent:
- **Fields:** Action, Strategy Skill (with lineage), Regime (with confidence), Signal Sources (with tier and hit rate), Position Impact (vs. limits), Confidence Score (above/below auto-approval), Paper Track Record, Action Required
- **Channels:** Slack, Telegram, email, dashboard push notification
- **Approval workflow:** Reply APPROVE or REJECT within configurable timeout (default: 4 hours). No response = auto-reject.
- **Notification preferences:** User configures which channels, which trade sizes trigger notifications, quiet hours

### Deliverables
- [ ] Alpaca paper trading connected and executing trades
- [ ] TradeIntent object with all fields
- [ ] Gate 1: risk constraint check blocking violating trades
- [ ] Gate 2: paper shadow executing all intents
- [ ] Gate 3: graduated approval gate
- [ ] 5-stage promotion protocol with bidirectional transitions
- [ ] Rolling average confirmation metrics
- [ ] Trade notifications via Slack, Telegram, email, dashboard
- [ ] Approval workflow with configurable timeout
- [ ] Notification preferences system

### Testing
- Alpaca integration: submit paper order → verify execution and position update
- Gate 1: trade breaching 5% limit → blocked, logged, not executed
- Gate 2: vetoed trade still executes on paper → counterfactual tracked
- Gate 3: trade below auto-approval threshold → requires human approval
- Promotion: inject 90 days of qualifying metrics → verify system promotes to Paper Validation
- Demotion: inject degrading metrics → verify system demotes
- Notifications: trade intent → notification delivered to configured channels
- Approval: approve via Slack → trade proceeds; timeout → trade rejected

---

## Phase 7: Orchestrator
**Detailed plan:** [`docs/plans/phase-7-orchestrator.md`](phase-7-orchestrator.md)

### Goal
Build the meta-evolution orchestrator — an LLM-powered agent that sits above the entire system on a weekly/bi-weekly cadence, detecting cross-layer interactions, running counterfactual analysis, and tuning how the system evolves. **This is the core intellectual property and primary defensible asset of Evolve-Trader.** Existing systems do strategy evolution (OpenSpace) or multi-agent trading (AI-Trader) or signal tracking (Quiver, WhaleWisdom). None have a cross-layer orchestrator that watches the whole system evolve and adjusts the evolution process itself. Individual strategies can be copied; signal sources are public data. The orchestrator's learned understanding of how to coordinate evolution across strategy, signal, regime, sizing, and discovery layers — that's what can't be replicated.

### Key Activities

#### Orchestrator Agent
- LLM-powered agent running on slower cadence (weekly or bi-weekly)
- Ingests all monitoring metrics from past period: returns, drawdowns, evolution events, regime classifications, signal source performance, constraint proximity, trade frequency, conflict resolutions
- Subject to the same immutable risk constraints (cannot relax hard caps)

#### Evolution Pace Control
- Monitor evolution event log for each layer
- Detect cycles (FIX → un-FIX → re-FIX the same skill), oscillations, and convergence patterns
- Slow down layers that are oscillating (reduce evolution frequency)
- Speed up layers that are stagnant (increase evolution frequency)

#### Inter-Layer Tension Management
- Identify when two layers pull in opposite directions (e.g., aggressive strategies + sensitive regime classifier = whipsaw)
- Run counterfactual analysis: "what would last month's returns have been if the regime classifier was less sensitive?" vs. "what if position sizing was more conservative?"
- The counterfactual with better returns informs which layer to constrain

#### Counterfactual Replay
- Replay the last period's trades with proposed adjustments applied
- Estimate impact of each proposed change before applying it
- Only apply adjustments that pass counterfactual validation
- Log reasoning for deferred adjustments for later review
- This is computationally expensive — design for efficiency (batch replay, caching)

#### Monitoring Threshold Calibration
- All monitoring metrics have thresholds that trigger alerts or adjustments
- Orchestrator tunes these thresholds based on system-wide performance
- Consistently profitable → slightly relax thresholds (more aggressive behavior allowed)
- Drawdowns increasing → tighten thresholds
- This is the meta-evolution loop: parameters governing evolution are themselves evolved

#### Discovery Engine Tuning
- Adjust how aggressively the discovery engine (Phase 9) promotes new sources
- Active roster performing well → slow promotion (don't fix what isn't broken)
- Performance declining → accelerate promotion, faster demotion of underperformers

#### Cross-Layer Correlation Analysis
- Monitor whether improvements in one layer are offset by degradation in another
- System-level health dashboard tracking interaction effects
- Healthy: improving metrics across layers with minimal cross-layer interference
- Unhealthy: individual layers "improving" while overall performance degrades

#### Regime Classifier Decomposition
- Monitor which input signals the regime classifier is actually using (via LLM reasoning attribution from Phase 1)
- When post-execution analyzer sees classifier failing, let evolution engine DERIVE specialized variants
- May naturally split into macro/sentiment/sector sub-classifiers
- Each sub-classifier evolves independently
- Outputs compose into multi-dimensional regime vector rather than single label

### Deliverables
- [ ] Orchestrator agent running on configurable cadence
- [ ] Evolution pace control with oscillation/stagnation detection
- [ ] Inter-layer tension detection and resolution
- [ ] Counterfactual replay engine
- [ ] Monitoring threshold auto-calibration
- [ ] Discovery engine tuning interface (ready for Phase 9)
- [ ] Cross-layer correlation analysis
- [ ] Regime classifier decomposition support
- [ ] Orchestrator adjustment log (all decisions recorded with reasoning)

### Testing
- Oscillation detection: inject evolution log with FIX→un-FIX→re-FIX pattern → orchestrator detects and slows layer
- Tension detection: inject metrics showing aggressive sizing + sensitive classifier → orchestrator flags whipsaw risk
- Counterfactual: replay known trade sequence with parameter change → verify return estimate matches expected
- Threshold calibration: inject period of strong performance → verify thresholds relax slightly
- Cross-layer: inject improving strategy metrics + declining overall performance → orchestrator identifies interference

---

## Phase 8: Strategy Incubator
**Detailed plan:** [`docs/plans/phase-8-incubator.md`](phase-8-incubator.md)

### Goal
Build the concurrent tournament of candidate strategies. Start with small population (5-10) and 3 generation methods (mutation, crossover, regime-conditioned search). Implement fitness function, fossil record, and incubator monitoring. Add remaining generation methods incrementally.

### Key Activities

#### Concurrent Tournament Architecture
- Separate population of paper-trading agents, isolated from production
- Each candidate runs its own strategy SKILL.md against live market data
- Shares the production signal layer and regime classifier (read-only)
- Isolated paper portfolios tracked internally via AI-Trader's portfolio engine
- Alpaca paper accounts reserved for production — incubator uses internal simulation

#### Tournament Phases
1. **Seeding (Day 0):** New candidates from generation methods. Each starts with $10K paper portfolio. Max 10 new per generation.
2. **Incubation (30 trading days):** All candidates trade simultaneously. Full logging.
3. **Evaluation (Day 30):** Scored on Sharpe, max drawdown, win rate, trade count (min 8 for statistical relevance), consistency (std dev of weekly returns).
4. **Selection (Day 30):** Bottom 30% eliminated (archived in fossil record). Top 10% earn elite status.
5. **Reproduction (Day 31):** Elite produce offspring via mutation and crossover. New candidates fill remaining slots.
6. **Promotion (after 2+ surviving generations):** Survive 60+ trading days with Sharpe >0.5, drawdown <15% → graduate to production probation tier.

#### Fitness Function
- Risk-Adjusted Returns (40%): Sharpe ratio
- Drawdown Discipline (25%): Maximum drawdown during incubation
- Complementarity (20%): Correlation with existing production strategies (uncorrelated = higher score)
- Regime Specificity (15%): Performance in regimes where production library is weak (orchestrator communicates weaknesses)

#### Generation Methods (Initial 3)
1. **Mutation:** Modify one component of existing production strategy. LLM proposes change with hypothesis.
2. **Crossover:** Combine elements from 2+ strategies. LLM acts as coherence filter — incoherent hybrids rejected.
3. **Regime-Conditioned Search:** Explicitly search for strategies that work in specific regimes using historical replay.

#### Generation Methods (Added Incrementally)
4. **Academic Paper Mining:** Agent ingests papers from SSRN, arXiv (q-fin), Journal of Finance, JFE, RFS. Converts published anomaly research into candidate SKILL.md files.
5. **Anomaly Detection:** Scan market returns for statistical anomalies. LLM generates hypotheses for repeatability.
6. **Counter-Strategy Generation:** Invert logic of best-performing skills. Test if inverse has alpha on different timeframe.
7. **Meta-Strategy Discovery:** Exploit copycat behavior. NANC ETF rebalancing patterns. Fading Pelosi copycat surge. 13F disclosure date price pressure.

#### Fossil Record
- All eliminated strategies archived (not deleted)
- Includes market conditions under which they failed
- Orchestrator can resurrect strategies when conditions change to resemble environments where they showed promise
- Strategies eliminated during bull market may be needed when bear arrives

#### Incubator Monitoring Panel (Dashboard Extension)
- Current population size and generation number
- Fitness distribution histogram across active candidates
- Top 5 candidates by composite fitness with strategy descriptions
- Graduation rate: candidates per quarter surviving to production
- Diversity metrics: style distribution (momentum, mean-reversion, value, defensive, hybrid)
- Generation source effectiveness: which method produces the most graduates
- Orchestrator allocates more seeding slots to productive generation sources

#### Population Dynamics
- Target population size is an evolvable parameter managed by orchestrator
- Production performing well + low graduation rate → shrink population
- Production underperforming or unsuitable regime → expand population, accelerate cycle
- Adaptive exploration/exploitation balance

### Deliverables
- [ ] Concurrent tournament running with isolated paper portfolios
- [ ] Tournament phases: seeding → incubation → evaluation → selection → reproduction → promotion
- [ ] Fitness function with 4 weighted components
- [ ] 3 initial generation methods: mutation, crossover, regime-conditioned search
- [ ] Fossil record with archival and resurrection capability
- [ ] Incubator monitoring panel on dashboard
- [ ] Population dynamics managed by orchestrator
- [ ] 4 additional generation methods added incrementally

### Testing
- Tournament: run full cycle with 5 candidates → verify bottom 30% eliminated, top 10% reproduced
- Fitness function: known performance data → verify correct composite score and ranking
- Mutation: verify offspring differs from parent in exactly one component
- Crossover: verify coherence filter rejects nonsensical combinations
- Fossil record: eliminate strategy → change regime → verify orchestrator resurrects it
- Promotion: candidate surviving 60+ days with qualifying metrics → graduates to production
- Complementarity: strategy identical to existing production skill → low complementarity score

---

## Phase 9: Signal Expansion
**Detailed plan:** [`docs/plans/phase-9-signal-expansion.md`](phase-9-signal-expansion.md)

### Goal
Expand the signal layer with additional sources: ARK daily trades, prediction markets (Polymarket + Kalshi), options unusual activity, on-chain whale tracking, institutional investor letters, and news/macro feeds. Build the automated signal source discovery engine.

### Key Activities

#### Additional Signal Sources
Each produces typed SignalEvents through the existing framework:

- **ARK Invest Daily Trades:** Parse daily trade emails / API. Same-day latency. High-conviction growth/innovation thesis signals. Decay: ~10 days, fast exponential.

- **NANC/GOP Congressional ETFs (Unusual Whales):** Daily NAV tracking of Democratic (NANC) and Republican (GOP) congressional portfolio ETFs. Provides a real-time benchmark for congressional alpha without parsing individual disclosures. NANC +20.8% and GOP +18.8% in 2025, both beating S&P 500. Trivial to ingest via standard market data feeds. Useful as: (a) benchmark for our own congressional signal performance, (b) the NANC/GOP performance differential itself is a partisan sentiment signal, (c) rebalancing dates create predictable price patterns exploitable by the incubator's meta-strategy discovery.

- **Prediction Markets (Polymarket + Kalshi):**
  - Fastest signal layer — reprices within minutes of breaking news. Polymarket exceeds $2.5B weekly volume and received a $2B strategic investment from ICE (NYSE's parent company) at $9B valuation — crossed from novelty to institutional-grade signal infrastructure.
  - Categories: monetary policy, recession/growth, trade policy/tariffs, geopolitical, elections, crypto-specific
  - Cross-platform consensus signal: Polymarket (crypto-native, global) vs. Kalshi (CFTC-regulated, US)
  - Divergence between platforms is itself informative (regulatory risk, information asymmetry, liquidity differences)
  - API: `polymarket-apis` PyPI package (REST, WebSocket, CLOB, Pydantic validation, batch orders, settlement), Kalshi REST API, Bitquery GraphQL for on-chain Polymarket data on Polygon (includes Kafka streams for ultra-low-latency)
  - Prediction market whale tracking via Polymarket Analytics (same rolling scorecard methodology as hedge fund 13F tracking)

- **Options Unusual Activity:** Unusual volume, put/call ratio shifts, strike clustering. Pre-event positioning signals. Decay: ~2 days, very fast exponential. Sources: CBOE, Unusual Whales, Market Chameleon.

- **On-Chain Whale Movements:** Large wallet accumulation/distribution for crypto assets. Real-time. Decay: ~3 days, fast exponential. Sources: Arkham Intelligence, Whale Alert, Etherscan.

- **Institutional Investor Letters:** Ray Dalio (debt cycles), Howard Marks (credit cycles), Seth Klarman, Jeremy Grantham (bubble calls), Jamie Dimon, Larry Fink. Parsed via LLM. Thesis-level macro regime frameworks. Quarterly/annual cadence.

- **News & Macro Feeds:** Breaking events, Fed language, earnings surprises, geopolitical triggers. Real-time. Sources: Jina AI (**already in AI-Trader — wrap existing integration into SignalEvent format**), NewsAPI (new), FRED (new — macro economic data).

#### Signal Latency Hierarchy (Verified in System)
Prediction markets (minutes) → options unusual activity (hours) → news feeds (hours) → on-chain whale moves (hours-days) → ARK daily trades (same day) → Form 4 insider filings (days) → congressional STOCK Act (weeks) → 13F institutional filings (months)

#### Automated Signal Source Discovery Engine
5 discovery channels:

1. **WhaleWisdom Fund Performance Search:** Quarterly scan of 10,000+ institutional 13F filers. Filter: WhaleScore top decile + concentrated portfolios (top 10 holdings >60% AUM) + small-to-mid AUM ($50M-$2B) + low turnover (positions held 2+ quarters). API + Apify scrapers.

2. **SEC EDGAR Real-Time 13F Stream:** `sec-api` Python package with sub-200ms filing indexing. Monitor new 13F-HR filings each quarter. Flag untracked filers showing >30% trailing-twelve-month alpha. Kaleidoscope API (`api.kscope.io`) for push notifications. Backfill via EDGAR bulk data sets.

3. **Congressional Trader Emergence Detection:** Monitor newly seated members trading actively within first 6 months (via our congressional scraper). Monitor committee reassignment to trading-relevant sectors. **Highest priority:** members ascending to leadership positions (the Wei & Zhou inflection point).

4. **Insider Cluster Emergence:** Detect 3+ executives at different companies in same sector filing Form 4 purchases within 2-week window. Sector-level signal more robust than individual trades. HedgeFollow and Fintel for aggregation.

5. **Social/Media Signal Source Mining:** Track prominent financial commentators with verifiable, timestamped, falsifiable predictions. Highest bar: minimum 20 verifiable predictions with >60% hit rate before entering even Tier 3.

#### Discovery-to-Production Pipeline
Same lifecycle as Phase 3 (candidate → observation → probation → active → demotion) but with automated discovery feeding the candidate stage.

#### Scale Considerations
- Two-stage filter: fast coarse filter (simple return thresholds on structured data, runs weekly across full universe) + slower fine filter (LLM analysis of trading patterns, only on candidates passing coarse filter)
- Steady state: actively tracking 50-100 signal sources while continuously scanning full universe
- Compute-efficient: mirrors quantitative fund's factor zoo approach

### Deliverables
- [ ] ARK daily trade signal source
- [ ] Polymarket signal source with whale tracking
- [ ] Kalshi signal source with cross-platform consensus
- [ ] Options unusual activity signal source
- [ ] On-chain whale movement signal source
- [ ] Institutional investor letter parser
- [ ] News/macro feed signal source
- [ ] Signal latency hierarchy verified in monitoring
- [ ] Discovery engine: WhaleWisdom quarterly scan
- [ ] Discovery engine: EDGAR real-time 13F stream
- [ ] Discovery engine: Congressional emergence detection
- [ ] Discovery engine: Insider cluster emergence
- [ ] Discovery engine: Social/media mining
- [ ] NANC/GOP ETF signal source with partisan differential tracking
- [ ] Two-stage filter (coarse + fine) for discovery scalability

### Testing
- Each signal source: inject known data → verify correct SignalEvent output
- NANC/GOP ETFs: verify daily NAV ingestion and partisan differential signal
- Prediction market: Polymarket-Kalshi divergence → cross-platform signal fires
- Discovery: inject synthetic 13F with qualifying metrics → candidate flagged
- Congressional emergence: inject leadership promotion event → high-priority candidate created
- Insider cluster: inject 3 Form 4 buys in same sector → cluster signal fires
- Scale: 10,000 synthetic filers through coarse filter → verify performance and correct filtering
- Full integration: new signal sources feed into existing meta-selector and regime classifier

---

## Phase 10: Crowding, Contrarian & Synthetic Benchmarks
**Detailed plan:** [`docs/plans/phase-10-crowding-benchmarks.md`](phase-10-crowding-benchmarks.md)

### Goal
Build crowding detection and contrarian skill activation. Validate the system with broad historical distributions, quiet-market periods, and exploratory historical scenario packs.

### Key Activities

#### Cross-Source Convergence Scorer
- Compute rolling correlation matrix across all active signal sources
- When multiple independent sources converge on same trade → treat as single amplified signal with crowding flag (not additive independent signals)
- Above evolvable threshold → meta-selector reduces exposure or activates hedging skills
- **Monitoring metric:** Source Independence Score — average pairwise correlation. Rising = drift toward groupthink.

#### Contrarian Skill Family
- Dedicated SKILL.md files activated only when crowding is detected
- Defensive reasoning: reduce position sizes, add tail-risk hedges, increase cash allocation, initiate short positions in most crowded names
- Subject to evolution (FIX/DERIVED/CAPTURED) but only tested during crowding events

#### Historical Crowding Calibration
- Validate against known crowding episodes:
  - 2021 meme-stock squeeze (GME, AMC)
  - 2022 rate-shock tech selloff
  - 2020 COVID volatility spike
- Crowding detector should produce elevated scores during these periods

#### Exploratory Historical Scenario Packs
Use a small set of famous historical narratives as regression fixtures and diagnosis tools, not as hard acceptance gates:
1. **Soros Bank of England Short (1992):** Macro-stress exploratory fixture
2. **Burry Subprime Thesis (2005-2007):** Slow-burn deterioration exploratory fixture
3. **Druckenmiller Dot-Com Exit (early 2000):** Tech-crowding exploratory fixture
4. **Ackman COVID Hedges (Feb-Mar 2020):** Rapid regime-change exploratory fixture
5. **Paulson Gold Trade (2009-2011):** Inflation-protection exploratory fixture

#### "Quiet Market" Validation
Test against 3-4 periods where the correct answer was "stay invested, don't do anything clever":
- Mid-2014 to mid-2015
- Mid-2017
- Q3 2018
- **Unnecessary Trade Rate metric:** Number of regime-change signals per quarter during these periods. Target: ≤2-3. If more → regime classifier confidence thresholds need to go up.
- Capital Preservation should NOT dominate during quiet bull markets — the system should stay invested, not hide in cash
- Cash Deployment Rate during quiet periods should be <20% (system is deploying capital, not sitting idle)

#### Graceful Degradation Validation
Full integration test of system resilience:
- Kill all signal feeds → verify system continues operating on strategy library + Capital Preservation
- Kill individual signal sources → verify meta-selector adapts weights among remaining sources
- Simulate API schema change on one source → verify schema validation catches it and source is marked unhealthy
- Verify no trades are made based on stale/missing signals (the system should reduce confidence, not hallucinate)

### Deliverables
- [ ] Cross-source convergence scorer with crowding flag
- [ ] Source Independence Score monitoring metric
- [ ] Contrarian/hedging skill family (3+ skills)
- [ ] Historical crowding calibration against 2020/2021/2022
- [ ] Exploratory scenario pack: Soros 1992
- [ ] Exploratory scenario pack: Burry 2005-2007
- [ ] Exploratory scenario pack: Druckenmiller 2000
- [ ] Exploratory scenario pack: Ackman 2020
- [ ] Exploratory scenario pack: Paulson 2009-2011
- [ ] Quiet market validation against 3-4 periods with Unnecessary Trade Rate metric
- [ ] Graceful degradation validation (all signals killed → system survives)
- [ ] Benchmark results documented with identified weaknesses

### Testing
- Convergence scorer: inject 3 sources all buying same sector → crowding flag fires
- Contrarian activation: crowding detected → contrarian skills activate → exposure reduces
- Distributional validation: run broad historical windows → verify crowding controls improve robustness without increasing unnecessary trade rate
- Exploratory scenarios: run each narrative fixture → inspect whether failures are diagnostic, not whether the system "matches the legend"
- Quiet markets: run calm periods → verify Unnecessary Trade Rate ≤2-3 per quarter, Cash Deployment Rate <20%
- Graceful degradation: kill all signal feeds → system continues on strategy library alone
- Graceful degradation: kill single source → meta-selector redistributes weight among remaining sources
- Graceful degradation: simulate API schema change → source marked unhealthy, alert fires

---

## Phase 11: Live Trading & Hardening
**Detailed plan:** [`docs/plans/phase-11-live-trading.md`](phase-11-live-trading.md)

### Goal
Connect to Alpaca live trading. Operationalize the graduated promotion pipeline defined in Phase 6, build the kill switch with all trigger mechanisms, and harden the system for production.

### Key Activities

#### Alpaca Live Integration
- Switch from paper to live endpoint (`api.alpaca.markets`) — single config flag
- Paper shadow continues running in parallel (Gate 2 never stops)
- Fractional shares, extended hours, all order types supported
- Real-time P&L and position sync via WebSocket

#### Promotion Pipeline Operationalization
- Reuse the Phase 6 promotion protocol as the single source of truth for Paper -> Validation -> Micro -> Partial -> Full transitions
- Add production-only guardrails: live capital caps, approval latency SLAs, reviewer audit fields, and regime-diversity enforcement
- Tracks approval rate, response time, override frequency
- Human always has manual override capability
- Dashboard shows every auto-approved trade with rationale summary and supporting evidence, not raw chain-of-thought

#### Kill Switch
- **Dashboard button:** Prominent, always visible, one-click activation
- **Slack/Telegram command:** `/kill` or equivalent
- **API endpoint:** Programmatic trigger for automated systems
- **Auto-trigger:** 20% drawdown hard limit breached → automatic activation
- **Actions on activation:**
  1. Cancel all open orders across all connected brokers
  2. Close all positions at market (configurable: or hold positions but stop new trades)
  3. Revert system to Paper Training mode
  4. Send immediate notification with full portfolio state
- Never subject to AI override

#### Security Hardening
- API key management (never in code, environment variables or secrets manager)
- Rate limiting on all external API calls
- Audit log of all trade executions, approvals, overrides
- Encrypted storage of sensitive data
- Network security for cloud deployment
- Regular dependency vulnerability scanning
- Production logging and observability fully configured (structured JSON logs, log aggregation, alerting on CRITICAL/ERROR)
- Automated database backup schedule active (daily full + continuous WAL archiving)
- Disaster recovery runbook documented and tested

#### Regime Diversity Requirement (Promotion Gate)
- Strategy skills must demonstrate positive or neutral performance across at least 2 distinct regime classifications before production promotion
- Prevents accumulating regime-specific skills that appear strong in backtests but fail under regime change
- This is the final anti-overfitting defense at the production boundary

### Deliverables
- [ ] Alpaca live trading connected with paper shadow continuing
- [ ] Graduated approval gate operational
- [ ] Kill switch: dashboard button, Slack/Telegram command, API endpoint, auto-trigger
- [ ] Security hardening: key management, rate limiting, audit log, encryption
- [ ] Regime diversity requirement enforced at promotion
- [ ] Full promotion pipeline tested end-to-end (paper → micro → partial → full)
- [ ] Bidirectional demotion tested (degrading metrics → auto-demotion)
- [ ] Production logging and observability operational
- [ ] Automated database backups running and tested
- [ ] Disaster recovery runbook documented

### Testing
- Live integration: submit real order (micro-live with minimum capital) → verify execution
- Paper shadow: live trade executes → paper trade also executes → results compared
- Kill switch: trigger via each method → verify all orders cancelled, positions closed, system reverted
- Auto-trigger: inject 20% drawdown → kill switch fires automatically
- Security: verify API keys not in code, audit log captures all actions
- Regime diversity: strategy that only works in bull market → blocked from production

---

## Phase 12: Extensions & Polish
**Detailed plan:** [`docs/plans/phase-12-extensions.md`](phase-12-extensions.md)

### Goal
Extend the system with crypto-specific capabilities, additional brokerage support, prediction market direct trading, and address the open research questions.

### Key Activities

#### Crypto-Specific Regimes & BITWISE10
- Adapt regime classifier for crypto regimes: DeFi summer, NFT mania, regulatory crackdown, halving cycles
- Validate strategy evolution against crypto assets via Hyperliquid (AI-Trader has no BITWISE10 module — we define our own crypto universe)
- On-chain signals (from Phase 9) become primary signal layer for crypto strategies

#### IBKR Integration
- Interactive Brokers via TWS API or IB Gateway
- 170 markets in 40 countries, futures, forex, bonds, margin, short selling
- IB Gateway for headless server deployment
- Python via `ibapi` (native) or `ib_insync` (community wrapper)
- Port 7496 (live) / 7497 (paper)

#### Prediction Market Direct Trading
- Trade prediction markets as additional revenue source (not just signal input)
- When system's model diverges from Polymarket pricing → arbitrage opportunity
- Strategy incubator can evolve prediction-market-specific strategies
- Polymarket CLOB API for programmatic order placement via `polymarket-apis`

#### Open Research Questions
- **Optimal evolution cadence:** Empirical testing of daily vs. weekly vs. event-triggered evolution
- **Skill library carrying capacity:** What's the optimal number of active strategy skills?
- **Cross-market skill transfer:** Can NASDAQ-evolved momentum transfer to crypto?
- **LLM model sensitivity:** Are evolved skills model-specific or portable? Test via AI-Trader's multi-model arena
- **Adversarial robustness:** Can signal sources be gamed? Explore via TradeTrap framework

#### Dashboard v2 Refinements
- Based on user feedback from Phase 5 dashboard
- Performance optimizations for large datasets
- Mobile-optimized views for on-the-go monitoring
- Custom alert configuration
- LLM cost panel refinements (historical trend analysis, cost optimization recommendations — base panel already in Phase 5)

#### Open-Source Release Preparation (Optional)
If the project is to be released as open-source:
- Comprehensive documentation: architecture overview, getting started guide, API reference, configuration guide
- Community contribution guidelines (CONTRIBUTING.md)
- Issue templates and PR templates
- License verification (both OpenSpace and AI-Trader are MIT — verify all dependencies are compatible)
- Sensitive data audit: ensure no API keys, credentials, or personal trading data in repo history
- Example configurations and demo mode (runs with mock data, no API keys required)
- Docker Compose one-command demo setup

#### Disclaimer Integration
All system outputs that could be interpreted as financial guidance must include disclaimers:
- Dashboard footer: persistent disclaimer that this is a research tool, not investment advice
- Trade notifications: each notification includes a disclaimer footer
- Strategy skill outputs: no language implying guaranteed returns
- Full disclaimer text (from original plan document) accessible from dashboard "About" page

### Deliverables
- [ ] Crypto regime classifier and crypto universe validation
- [ ] IBKR integration with all supported order types
- [ ] Prediction market direct trading
- [ ] Research results documented for open questions
- [ ] Dashboard v2 with refinements and LLM cost panel
- [ ] Open-source release preparation (if applicable): docs, CONTRIBUTING.md, demo mode
- [ ] Disclaimer integration across all user-facing outputs

### Testing
- Crypto regimes: validate regime classifier identifies DeFi/NFT/regulatory/halving regimes on historical crypto data
- IBKR: submit paper order via IB Gateway → verify execution and position sync
- Prediction market trading: submit Polymarket order via CLOB API → verify execution
- Cross-market transfer: run NASDAQ-evolved strategy on crypto universe → document performance delta
- Model portability: run same evolved skill on Claude Sonnet vs GPT-4o vs Qwen → compare trade decisions
- Disclaimer: verify disclaimer appears on every dashboard page footer and every trade notification
- Demo mode: verify system runs end-to-end with mock data and no API keys

---

## Risk Mitigations as Testable Items

These risks from the original plan are mapped to specific verification tasks:

| Risk | Severity | Phase Tested | Verification |
|------|----------|-------------|--------------|
| Overfitting to historical data | Critical | Phase 1, 10 | Walk-forward validation, paper gate, distributional evaluation, complexity penalties, regime diversity — all 5 defenses verified independently |
| Signal source API instability | High | Phase 2, 9 | Fallback chains per source; system runs on strategy library alone if all signals fail (test: kill all signal feeds → system degrades gracefully) |
| LLM hallucination in strategy generation | High | Phase 1, 8 | Post-execution validation against actual returns; skills producing trades based on non-existent data auto-detected and purged |
| Regime classifier lag | Medium | Phase 3, 9 | Fast-moving signals (options, prediction markets) compensate for slow signals (13F); ensemble approach verified |
| Evolution runaway | Medium | Phase 1, 7 | OpenSpace anti-loop guards + orchestrator oscillation detection; no skill replacement until paper gate passed |
| Regulatory risk | Medium | Phase 11 | System positioned as research tool; all outputs include disclaimers; no live capital without explicit human authorization |
| Data cost escalation | Low | Phase 2, 9 | Core sources (EDGAR, FRED) free; premium APIs optional; cost monitoring in dashboard |
| LLM API cost overrun | Medium | Phase 1+ | Token usage tracking per component; budget caps with hard stop; model tiering (cheap models for incubator); cost-per-trade metric; monthly budget dashboard |
| Data loss / database corruption | Medium | Phase 11 | Automated daily backups + WAL archiving; quarterly recovery testing; skill library in git; export capability |
| Deployment downtime during market hours | Medium | Phase 11 | Cloud deployment with uptime monitoring; health checks; auto-restart on failure; kill switch as safety net |
| Signal source schema changes | Medium | Phase 2+ | Per-source schema validation on every fetch; auto-detection of breaking changes; source marked unhealthy on mismatch; alert fires |

---

## Academic Foundations (Reference)

All citations from the original plan document are preserved here for reference during implementation:

- Fan et al. (2025) — AI-Trader benchmarking framework
- Xu, Chen & Huang (2025) — OpenSpace self-evolving skill engine
- Jeng, Metrick & Zeckhauser (2003) — insider trading returns
- Wei & Zhou (2025) — congressional leaders outperform by 47% annually (NBER)
- Eggers & Hainmueller (2013) — alpha concentrates in leadership positions
- Sias (2004) — institutional herding precedes reversals
- Wermers (1999) — mutual fund herding price impact
- Kelly (1956) — optimal bet sizing
- Ang, Hodrick, Xing & Zhang (2006) — volatility-managed portfolios
- Hamilton (1989) — Markov-switching regime detection
- Ang & Bekaert (2002) — regime-aware allocation outperforms static
- Chen & Yeh (2001) — evolutionary strategy discovery in financial markets
- LeBaron (2006) — agent-based computational finance
- Pan & Poteshman (2006) — options volume predicts future prices
- Tetlock (2007) — media sentiment predicts stock returns
