# Phase 1 Implementation Spec

## Goal

Implement the core strategy evolution loop on Evolve-Trader's historical replay harness.

## Depends On

- Phase 0

## Owns

- `StrategySkill` schema
- Post-execution analyzer
- Walk-forward validation harness
- Capital-preservation skill
- Immutable risk constraints
- Stochastic fitness evaluation (using `numpy` + `scipy.stats` directly; `quantstats`/`empyrical` available for future use)
- Version DAG
- Shared `LLMUsageLogger` interface with file-backed persistence
- Initial seed strategy library

## Contracts

- Strategy execution and evaluation operate on out-of-sample validation, not only in-sample replay
- Capital preservation is always available as a fallback skill
- `LLMUsageLogger` lives at `src/evolve_trader/core/llm_logger.py` and survives into Phase 2

## Key Dependencies

- `numpy` + `scipy.stats` — Sharpe, drawdown, distributional metrics (direct implementation)
- `quantstats` + `empyrical` — available in phase1 extras for future use (HTML tearsheets, Monte Carlo)
- `yfinance` — historical market data (available; replay currently uses synthetic trades)
- `litellm` — LLM-driven evolution via configurable model routing
- Reference architectures for replay harness (both MIT licensed):
  - `DanRedelien/futures-backtesting-engine` — no-lookahead 6-phase bar loop, FastBar numpy optimization, Optuna walk-forward
  - `zachisit/july-backtester` — walk-forward overfitting detection, Monte Carlo robustness scoring, SQN/R-Multiple, multiprocessing

## Deferred

- Real market data replay via yfinance (currently uses synthetic trades based on skill parameters)
- OpenSpace deep integration for CAPTURED mode and tool-assisted evolution (deferred to Phase 7)

## Non-Goals

- PostgreSQL persistence
- Signal ingestion
- Live trading

## Acceptance Criteria

- Historical replay can execute strategies end to end
- Walk-forward validation gates promotion
- Risk constraints are enforced and non-bypassable
- Version lineage is tracked
- Seed skills parse and execute through the same schema

