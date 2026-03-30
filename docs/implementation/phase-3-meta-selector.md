# Phase 3 Implementation Spec

## Goal

Route strategies using regime and signal intelligence rather than isolated strategy performance.

## Depends On

- Phase 2

## Owns

- Meta-selector
- Signal scoring and weighting
- Signal-source lifecycle stages
- Conflict resolution rules
- Post-signal return tracking
- Survivorship and popularity monitoring

## Contracts

- Signal-source lifecycle is the canonical source-promotion model later reused by discovery work
- Conflicting sources resolve through explicit weighting and fallback logic

## Non-Goals

- Position-sizing evolution
- Live broker execution

## Acceptance Criteria

- Strategy selection uses regime and signal context
- Signal weights evolve from observed performance
- Lifecycle states exist for source promotion and demotion
- Graceful degradation works when some signals are missing

