"""Immutable risk constraints — NEVER subject to AI override.

These are the hard safety limits that sit outside the evolution engine.
No evolved skill can remove, relax, or override these constraints.
The AI evolves everything else.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ConstraintViolation(Enum):
    """Types of constraint violations."""

    NONE = "none"
    POSITION_LIMIT = "position_limit_exceeded"
    SECTOR_LIMIT = "sector_limit_exceeded"
    DRAWDOWN_LIMIT = "drawdown_limit_exceeded"


@dataclass(frozen=True)
class RiskConstraints:
    """Hard risk limits. Frozen dataclass prevents modification."""

    max_position_pct: float = 0.05  # 5% max in any single position
    max_sector_pct: float = 0.25  # 25% max in any single sector
    max_drawdown_pct: float = 0.20  # 20% max drawdown before forced de-risk


@dataclass
class PortfolioState:
    """Current state of the portfolio for constraint checking."""

    total_value: float
    positions: dict[str, float]  # ticker -> current value
    sector_exposure: dict[str, float]  # sector -> fraction of portfolio
    current_drawdown: float  # 0.0 to 1.0


@dataclass
class TradeCheckResult:
    """Result of a constraint check on a proposed trade."""

    allowed: bool
    violation: ConstraintViolation = ConstraintViolation.NONE
    message: str = ""


def check_trade_allowed(
    constraints: RiskConstraints,
    portfolio: PortfolioState,
    ticker: str,
    sector: str,
    trade_value: float,
) -> TradeCheckResult:
    """Check if a proposed trade violates any immutable risk constraints.

    Called before EVERY trade, regardless of strategy, confidence, or
    automation level. This is the first gate in the execution pipeline.

    Check order: portfolio value -> drawdown -> position size -> sector.
    """
    if portfolio.total_value <= 0:
        return TradeCheckResult(
            allowed=False,
            violation=ConstraintViolation.DRAWDOWN_LIMIT,
            message="Portfolio value is zero or negative.",
        )

    # Check drawdown limit
    if portfolio.current_drawdown >= constraints.max_drawdown_pct:
        return TradeCheckResult(
            allowed=False,
            violation=ConstraintViolation.DRAWDOWN_LIMIT,
            message=(
                f"Current drawdown {portfolio.current_drawdown:.1%} "
                f"exceeds limit {constraints.max_drawdown_pct:.1%}. "
                f"Forced de-risk to Capital Preservation."
            ),
        )

    # Check position size limit
    existing_position = portfolio.positions.get(ticker, 0.0)
    new_position_value = existing_position + trade_value
    position_pct = new_position_value / portfolio.total_value

    if position_pct > constraints.max_position_pct:
        return TradeCheckResult(
            allowed=False,
            violation=ConstraintViolation.POSITION_LIMIT,
            message=(
                f"Position in {ticker} would be {position_pct:.1%} "
                f"of portfolio, exceeding {constraints.max_position_pct:.1%} limit."
            ),
        )

    # Check sector concentration limit
    existing_sector = portfolio.sector_exposure.get(sector, 0.0)
    new_sector_pct = existing_sector + (trade_value / portfolio.total_value)

    if new_sector_pct > constraints.max_sector_pct:
        return TradeCheckResult(
            allowed=False,
            violation=ConstraintViolation.SECTOR_LIMIT,
            message=(
                f"Sector {sector} would be {new_sector_pct:.1%} "
                f"of portfolio, exceeding {constraints.max_sector_pct:.1%} limit."
            ),
        )

    return TradeCheckResult(allowed=True)
