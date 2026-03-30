# Evolve-Trader AI — Instructions for Claude

## Profitability Contract (MANDATORY)

Before writing any code, implementing any feature, or making any architectural decision, read and comply with `docs/implementation/profitability-contract.md`. It overrides all phase specs.

Key constraints (11 sections — read the full doc):
1. **Baseline-beating:** strategy-class-matched benchmarks (SPY, sector ETF, beta-matched, low-net). Signal sources evaluated on marginal lift, not standalone returns.
2. **Executable alpha only:** after spread, slippage, delay, commissions. Reject if edge < 2x round-trip cost.
3. **Minimum statistical bar:** 30+ trades/window, 3+ windows, 2+ regime labels, deflated Sharpe, bootstrap CIs. No promotion if CIs overlap zero.
4. **Capacity and liquidity:** ADV participation cap (1% default), exit-capacity check, capacity-adjusted alpha must stay positive.
5. **Multiple-testing discipline:** experiment registry, search-breadth penalty, lower-confidence-bound Sharpe for promotion. More tests = higher bar.
6. **Narrow scope until first profitable quarter:** US large-cap, days-to-weeks, max 3 strategies, max 3 signal sources.
7. **Simplicity tax:** each layer must prove incremental Sharpe improvement >= 0.1 over the system without it.
8. **Champion/challenger:** one challenger at a time, must beat champion across 3+ OOS windows.
9. **LLM role boundaries:** structured hypotheses only, no trade decisions, benchmark against non-LLM baseline.
10. **Paper/live deviation:** primary health metric, auto-demotion if 30-day correlation < 0.8.
11. **Practical path:** find one edge, prove it survives costs, automate around it.

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
- Do not promote strategies without minimum 30 trades/window and deflated Sharpe
- Do not ignore capacity constraints — no live trading if ADV participation > 1%
- Do not cherry-pick from incubator/evolution without logging in experiment registry
