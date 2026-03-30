# Phase 6 Implementation Spec

## Goal

Connect the system to Alpaca paper trading and define the single promotion pipeline used through live trading.

## Depends On

- Phase 5

## Owns

- Paper trading client
- Order mapping
- `TradeIntent`
- Gate 1: immutable risk enforcement
- Gate 2: paper-shadow execution
- Gate 3: approval workflow
- Shared promotion pipeline and stage model
- Notification delivery
- Approval workflow engine

## Contracts

- `TradeIntent` includes `rationale_summary` and `rationale_evidence`
- Order idempotency, broker reconciliation, and market-session awareness are mandatory
- This phase defines the only promotion pipeline; later phases extend it, not replace it

## Non-Goals

- Live-capital execution
- Kill-switch implementation

## Acceptance Criteria

- Paper orders execute and reconcile correctly
- All three execution gates operate in order
- Approval workflow functions across supported channels
- Shared promotion stages from paper to full live are implemented and tested

