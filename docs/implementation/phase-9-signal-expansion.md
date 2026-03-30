# Phase 9 Implementation Spec

## Goal

Expand signal coverage and add automated source discovery on top of the existing signal lifecycle.

## Depends On

- Phase 3
- Phase 6
- Phase 7 optional for tuning only

## Owns

- Additional named signal sources
- Prediction-market integrations
- Options, on-chain, and news/macro sources
- Source-discovery modules
- Discovery-to-production pipeline built on the Phase 3 lifecycle

## Contracts

- New sources must emit standard `SignalEvent` records
- Discovery uses the same candidate/observation/probation/active/demotion model from Phase 3
- Orchestrator tuning is optional and additive

## Non-Goals

- New promotion lifecycle definitions

## Acceptance Criteria

- New sources integrate without changing core signal contracts
- Discovery outputs enter the existing lifecycle pipeline
- Source latency, health, and degradation are observable

