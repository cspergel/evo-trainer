# Phase 12 Implementation Spec

## Goal

Add optional market extensions and polish after the live system is stable.

## Depends On

- Phase 11

## Owns

- Crypto-specific regime extensions
- Crypto universe validation (via Hyperliquid, not BITWISE10)
- IBKR integration
- Prediction-market direct trading
- Open research modules
- Dashboard refinements
- Disclaimer integration
- Optional open-source release prep

## Contracts

- Extensions may not weaken live-trading safety controls
- New broker or market integrations must conform to the same audit and execution contracts

## Non-Goals

- Reworking core system architecture without a new planning pass

## Acceptance Criteria

- Each extension integrates behind existing execution and audit contracts
- Research modules are isolated from production paths unless explicitly promoted
- Dashboard and disclaimer updates are consistent with production safety posture

