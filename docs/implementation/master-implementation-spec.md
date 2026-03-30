# Evolve-Trader Master Implementation Spec

## Purpose

This document defines the build contracts for Evolve-Trader. It is the canonical implementation reference above the phase specs in this directory.

**IMPORTANT:** All implementation must also satisfy the [Profitability Contract](profitability-contract.md). That document defines the constraints that determine whether any feature ships to live: baseline-beating after costs, narrow initial scope, simplicity tax, champion/challenger, LLM role boundaries, and paper/live deviation tracking. The profitability contract overrides any phase spec that conflicts with it.

## System Model

The system has four trading layers plus one supervisory layer:

1. Strategy skill library
2. Signal intelligence
3. Regime classification
4. Meta-selection and portfolio construction
5. Orchestrator supervision

Immutable risk constraints apply at every layer.

## Global Invariants

- Max single-position exposure: 5%
- Max single-sector exposure: 25%
- Max portfolio drawdown before forced de-risking: 20%
- AI may not disable or relax these limits
- No trade path may bypass audit logging
- No live-trading component may operate without broker reconciliation and market-session awareness
- No component may persist raw prompt/response transcripts or unrestricted chain-of-thought as an operational requirement

## Canonical Shared Contracts

These contracts are defined once and reused:

- `StrategySkill`: structured strategy metadata plus markdown body
- `SignalEvent`: typed signal object with source metadata, confidence, timestamps, and decay profile
- `RegimeLabel`: current regime classification with confidence and supporting evidence
- `TradeIntent`: execution intent containing strategy, sizing, regime, signals, confidence, `rationale_summary`, `rationale_evidence`, and projected portfolio impact
- `LLMUsageLogger`: shared interface for cost and token logging from Phase 1 onward
- `ReplayHarness`: Evolve-Trader's own historical replay engine (AI-Trader is a paper trading simulator, not a backtest engine)
- Promotion pipeline: single shared stage model spanning paper training through full live deployment
- Operator state API: status surfaces used by the dashboard; destructive control actions are owned by execution phases

## Logging Policy

- Store compact metadata, costs, rationale summaries, evidence, and validation traces
- Do not make raw prompt/response logging a default requirement
- Treat auditability as structured data, not transcript retention

## Dashboard Control Ownership

- Phase 5 owns read-heavy dashboard views and operator-state surfaces
- Phase 6 owns approval workflow actions
- Phase 11 owns kill-switch actions and live-trading controls

## Promotion Pipeline Ownership

- Phase 6 defines the single promotion pipeline and shared stage model
- Phase 11 operationalizes that pipeline for live capital
- No later phase may introduce a second promotion enum, gate, or parallel workflow

## Validation Policy

- Required acceptance tests are distributional, integration, and operational
- Historical named scenarios may be used as exploratory fixtures or regressions
- Famous-trade narratives are not hard production acceptance gates

## Phase Graph

- Phase 0 -> Phase 1 -> Phase 2 -> Phase 3 -> Phase 4 -> Phase 5 -> Phase 6 -> Phase 7 -> Phase 8
- Phase 9 depends on Phases 3 and 6; Phase 7 may later tune it
- Phase 10 depends on Phases 8 and 9
- Phase 11 depends on Phase 10 and reuses the Phase 6 promotion pipeline
- Phase 12 depends on Phase 11

## Required Exit Artifacts Per Phase

- Implemented modules for the phase-owned scope
- Unit tests for all new contracts
- Integration tests for phase boundaries
- Updated observability and audit paths where applicable
- Updated implementation docs if phase scope changed during delivery

