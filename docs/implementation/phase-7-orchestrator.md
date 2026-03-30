# Phase 7 Implementation Spec

## Goal

Add the cross-layer orchestrator that supervises evolution and system tuning.

## Depends On

- Phase 6

## Owns

- Orchestrator agent
- Metrics aggregation
- Pace control
- Tension detection
- Counterfactual replay
- Threshold calibration
- Discovery tuning hooks
- Adjustment log
- CAPTURED evolution mode — detect emergent patterns from execution logs and create new skills (currently deferred from Phase 1; evaluate deep OpenSpace integration vs building our own)
- Auto-triggered evolution — background monitoring that triggers FIX/DERIVED when metrics degrade (replaces Phase 1's manual triggering)
- Tool-assisted evolution — give the LLM access to market data and test harness during the evolution agent loop (richer than Phase 1's single prompt/response)

## Contracts

- Orchestrator records structured rationales and cited metrics, not raw chain-of-thought
- All orchestrator adjustments pass through immutable risk constraints
- Discovery tuning hooks may tune Phase 9 but do not gate Phase 9 source integrations

## Non-Goals

- Direct broker actions outside existing execution paths

## Acceptance Criteria

- Orchestrator can ingest cross-layer metrics and emit bounded adjustments
- Counterfactual replay validates proposed changes
- All applied and deferred decisions are auditable

