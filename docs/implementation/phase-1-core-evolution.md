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
- Stochastic fitness evaluation (via `quantstats` + `empyrical` for metrics)
- Version DAG
- Shared `LLMUsageLogger` interface with file-backed persistence
- Initial seed strategy library

## Contracts

- Strategy execution and evaluation operate on out-of-sample validation, not only in-sample replay
- Capital preservation is always available as a fallback skill
- `LLMUsageLogger` lives at `src/evolve_trader/core/llm_logger.py` and survives into Phase 2

## Key Dependencies

- `quantstats` — Sharpe, Sortino, max drawdown, Monte Carlo, HTML tearsheets (~70 metrics)
- `empyrical` — lightweight programmatic metrics (alpha, beta, annual return)
- `yfinance` — historical market data for replay harness
- Reference architectures for replay harness (both MIT licensed):
  - `DanRedelien/futures-backtesting-engine` — no-lookahead 6-phase bar loop, FastBar numpy optimization, Optuna walk-forward
  - `zachisit/july-backtester` — walk-forward overfitting detection, Monte Carlo robustness scoring, SQN/R-Multiple, multiprocessing

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

