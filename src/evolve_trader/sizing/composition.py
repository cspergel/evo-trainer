"""Composition interface — strategy + sizing → trade proposal.

Strategy skills output WHAT and WHEN.
Sizing skills output HOW MUCH.
Portfolio constraints can veto or scale the result.

This is the handoff point where a strategy's intent becomes a
concrete trade proposal with quantity and risk checks applied.
"""

from __future__ import annotations

from dataclasses import dataclass

from evolve_trader.core.risk_constraints import (
    PortfolioState,
    RiskConstraints,
    TradeCheckResult,
    check_trade_allowed,
)
from evolve_trader.sizing.models import SizingResult


@dataclass
class TradeProposal:
    """A strategy's intent to trade (before sizing)."""

    strategy_skill: str
    ticker: str
    direction: str  # "BUY", "SELL"
    sector: str
    confidence: float
    regime: str
    rationale: str


@dataclass
class SizedTrade:
    """A fully sized trade proposal ready for constraint checking."""

    proposal: TradeProposal
    sizing: SizingResult
    constraint_check: TradeCheckResult | None = None

    @property
    def is_approved(self) -> bool:
        """Trade passes all checks."""
        return self.constraint_check is not None and self.constraint_check.allowed


def compose_trade(
    proposal: TradeProposal,
    sizing: SizingResult,
    constraints: RiskConstraints,
    portfolio: PortfolioState,
) -> SizedTrade:
    """Compose a strategy proposal with sizing and apply constraints.

    This is the core composition pipeline:
    1. Strategy says what/when (TradeProposal)
    2. Sizing says how much (SizingResult)
    3. Constraints approve or veto (TradeCheckResult)
    """
    trade_value = sizing.position_value
    if proposal.direction == "SELL":
        trade_value = -trade_value

    constraint_check = check_trade_allowed(
        constraints=constraints,
        portfolio=portfolio,
        ticker=proposal.ticker,
        sector=proposal.sector,
        trade_value=trade_value,
    )

    return SizedTrade(
        proposal=proposal,
        sizing=sizing,
        constraint_check=constraint_check,
    )
