# Phase 4 Implementation Spec

## Goal

Separate position sizing into an evolvable subsystem and enforce portfolio-level risk.

## Depends On

- Phase 3

## Owns

- Sizing-skill schema (SizingSkill, SizingMethod, SizingContext, SizingResult)
- Kelly criterion sizing (fractional Kelly, configurable)
- Volatility targeting sizing (inverse to asset volatility)
- Regime-adjusted sizing (multiplier per regime)
- Fixed fractional sizing (fallback)
- Composition interface (TradeProposal + SizingResult → SizedTrade with constraint check)
- Portfolio-level constraint enforcement via compose_trade()
- Existing exposure cap (sizing capped by remaining portfolio capacity)

## Deferred (tracked for later phases)

- Correlation-aware sizing overlays — needs portfolio correlation matrix
- Tax-aware mode hooks — needs tax lot tracking infrastructure
- Paper-trading survival gate — Phase 6 (needs Alpaca paper trading)
- Integration with AllocationResult → specific ticker selection → TradeProposal generation

## Contracts

- Strategy selection and sizing remain separate contracts
- Portfolio-level risk can veto or scale sizing outputs
- Sells always pass constraint checks (reduce risk)

## Non-Goals

- Paper broker integration
- Approval workflow

## Acceptance Criteria

- Strategies can be paired with independent sizing skills
- Portfolio constraints modify final exposure before execution
- Sizing decisions are auditable and replayable (SizingResult.rationale)
- Existing exposure caps new positions
