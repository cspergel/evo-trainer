# Phase 11 Implementation Spec

## Goal

Enable live trading, operationalize the shared promotion pipeline for real capital, and harden the system.

## Depends On

- Phase 10
- Phase 6 promotion pipeline

## Owns

- Live Alpaca client
- Kill-switch implementation
- Live-capital operational safeguards around promotion
- Regime-diversity enforcement
- Security hardening
- Production observability
- Backup and recovery

## Contracts

- Reuse the Phase 6 promotion pipeline and stage model
- No second promotion abstraction may be introduced
- Kill-switch actions are owned here and exposed to the Phase 5 dashboard surface
- Live trading requires reconciliation, session checks, and audit trails

## Non-Goals

- Replacing shared approval or promotion models

## Acceptance Criteria

- Live orders route correctly with paper shadow continuing
- Kill switch works from dashboard, messaging, API, and auto-trigger paths
- Live-capital promotion enforces regime diversity and operational checks
- Security, audit, backup, and recovery controls are verified

