# Evolve-Trader AI — Instructions for Claude

## Profitability Contract (MANDATORY)

Before writing any code, implementing any feature, or making any architectural decision, read and comply with `docs/implementation/profitability-contract.md`. It overrides all phase specs.

Key constraints:
- **Every strategy/signal/layer must beat buy-and-hold SPY after costs** (spread, slippage, delay)
- **Narrow scope until first profitable quarter:** US large-cap only, days-to-weeks horizon, max 3 strategies + capital preservation, max 3 signal sources
- **Simplicity tax:** each new layer must prove incremental value over the simpler stack
- **LLMs generate structured hypotheses, NOT trade decisions.** Decision path is symbolic and testable.
- **Champion/challenger:** changes compete against current best across 3+ OOS windows
- **Paper/live deviation is the primary health metric** — auto-demotion if correlation < 0.8

If a proposed feature or change does not satisfy the profitability contract, do not implement it. Flag the conflict instead.

## Project Structure

- `docs/implementation/master-implementation-spec.md` — canonical build contracts
- `docs/implementation/profitability-contract.md` — overrides everything else
- `docs/implementation/phase-*.md` — per-phase specs (subordinate to profitability contract)
- `docs/plans/` — detailed task-level plans
- `docs/tooling-inventory.md` — evaluated packages and reference architectures
- `src/evolve_trader/` — Python source (core, strategies, signals, regime, sizing, selection, db, dashboard)
- `frontend/` — Vite + React + Tailwind dashboard
- `lib/openspace/`, `lib/ai-trader/` — git submodules (pinned upstream repos)

## Code Standards

- Python 3.12+, Pydantic for schemas, SQLAlchemy for ORM
- TDD: write tests before implementation
- All CI must pass: ruff, black, mypy (strict), pytest
- Commit messages: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`
- No raw prompts/responses stored as operational data
- Immutable risk constraints (5% position, 25% sector, 20% drawdown) are NEVER relaxable

## What NOT to Do

- Do not add features that haven't passed the simplicity tax
- Do not let LLMs make final trade decisions
- Do not expand scope beyond S&P 500 large-cap until first profitable quarter
- Do not evolve multiple components simultaneously in live mode
- Do not ship strategies whose expected edge is less than 2x estimated round-trip cost
