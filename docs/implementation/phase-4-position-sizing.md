# Phase 4 Implementation Spec

## Goal

Separate position sizing into an evolvable subsystem and enforce portfolio-level risk.

## Depends On

- Phase 3

## Owns

- Sizing-skill schema
- Kelly and volatility sizing baselines
- Correlation-aware and portfolio-aware overlays
- Composition interface between strategy and sizing
- Portfolio-level constraint enforcement
- Tax-aware mode hooks

## Contracts

- Strategy selection and sizing remain separate contracts
- Portfolio-level risk can veto or scale sizing outputs

## Non-Goals

- Paper broker integration
- Approval workflow

## Acceptance Criteria

- Strategies can be paired with independent sizing skills
- Portfolio constraints modify final exposure before execution
- Sizing decisions are auditable and replayable

