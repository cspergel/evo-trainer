# Phase 5 Implementation Spec

## Goal

Provide operator visibility into system state through ops and trading dashboards.

## Depends On

- Phase 4
- Phase 2 for early skeleton work

## Owns

- FastAPI dashboard API layer
- WebSocket update layer
- Next.js frontend shell
- Ops panels
- Trading visibility panels
- Operator-state surface

## Contracts

- This phase is read-heavy
- Approval and kill-switch execution are not implemented here
- Operator-state APIs expose status, readiness, queues, and safe toggles only

## Non-Goals

- Trade approval actions
- Kill-switch actions
- Live trading

## Acceptance Criteria

- Portfolio, strategy, signal, and monitoring panels render from backend data
- Operator-state surfaces display approval queue, mode, override status, and kill-switch status
- UI contracts are stable enough for Phase 6 and Phase 11 to attach live actions later

