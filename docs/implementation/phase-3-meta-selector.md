# Phase 3 Implementation Spec

## Goal

Route strategies using regime and signal intelligence rather than isolated strategy performance.

## Depends On

- Phase 2

## Owns

- Meta-selector (regime + signals → weighted strategy allocation)
- Signal scoring and weighting (rolling scorecards with tier weights)
- Signal-source lifecycle stages (candidate → observation → probation → active → demoted)
- Conflict resolution rules (source-weighted dominance with capital preservation fallback)

## Deferred (tracked for later phases)

- Post-signal return tracking — needs real trade execution loop to measure actual returns
- Survivorship and popularity monitoring — needs longer running history
- Disclosure-to-executable spread tracking — needs live market data
- Multi-timeframe skill stacking (strategic/tactical/execution layers) — Phase 7+ orchestrator territory

## Contracts

- Signal-source lifecycle is the canonical source-promotion model later reused by discovery work
- Conflicting sources resolve through explicit weighting and fallback logic
- Meta-selector integrates scoring and lifecycle: demoted/candidate sources are filtered out

## Non-Goals

- Position-sizing evolution
- Live broker execution

## Acceptance Criteria

- Strategy selection uses regime and signal context
- Signal weights evolve from observed performance
- Lifecycle states exist for source promotion and demotion
- Graceful degradation works when some signals are missing
