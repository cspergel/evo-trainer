"""Tests for immutable risk constraints.

These constraints are NEVER subject to AI override or evolution.
"""

import pytest

from evolve_trader.core.risk_constraints import (
    ConstraintViolation,
    PortfolioState,
    RiskConstraints,
    check_trade_allowed,
)


def test_position_size_limit():
    """Trade exceeding 5% of portfolio is blocked."""
    constraints = RiskConstraints()
    portfolio = PortfolioState(
        total_value=100_000,
        positions={"AAPL": 3000},
        sector_exposure={"Technology": 0.03},
        current_drawdown=0.0,
    )
    result = check_trade_allowed(
        constraints=constraints,
        portfolio=portfolio,
        ticker="AAPL",
        sector="Technology",
        trade_value=3000,
    )
    assert result.allowed is False
    assert result.violation == ConstraintViolation.POSITION_LIMIT


def test_position_size_within_limit():
    """Trade within 5% of portfolio is allowed."""
    constraints = RiskConstraints()
    portfolio = PortfolioState(
        total_value=100_000,
        positions={},
        sector_exposure={},
        current_drawdown=0.0,
    )
    result = check_trade_allowed(
        constraints=constraints,
        portfolio=portfolio,
        ticker="AAPL",
        sector="Technology",
        trade_value=4000,
    )
    assert result.allowed is True


def test_sector_concentration_limit():
    """Trade exceeding 25% sector exposure is blocked."""
    constraints = RiskConstraints()
    portfolio = PortfolioState(
        total_value=100_000,
        positions={"AAPL": 12000, "MSFT": 12000},
        sector_exposure={"Technology": 0.24},
        current_drawdown=0.0,
    )
    result = check_trade_allowed(
        constraints=constraints,
        portfolio=portfolio,
        ticker="GOOGL",
        sector="Technology",
        trade_value=2000,
    )
    assert result.allowed is False
    assert result.violation == ConstraintViolation.SECTOR_LIMIT


def test_drawdown_forces_capital_preservation():
    """20% drawdown forces de-risking."""
    constraints = RiskConstraints()
    portfolio = PortfolioState(
        total_value=80_000,
        positions={"AAPL": 5000},
        sector_exposure={"Technology": 0.0625},
        current_drawdown=0.20,
    )
    result = check_trade_allowed(
        constraints=constraints,
        portfolio=portfolio,
        ticker="MSFT",
        sector="Technology",
        trade_value=2000,
    )
    assert result.allowed is False
    assert result.violation == ConstraintViolation.DRAWDOWN_LIMIT


def test_constraints_are_immutable():
    """Risk constraints cannot be modified after creation."""
    constraints = RiskConstraints()
    with pytest.raises(AttributeError):
        constraints.max_position_pct = 0.10  # type: ignore[misc]


def test_zero_portfolio_value():
    """Zero portfolio value blocks all trades."""
    constraints = RiskConstraints()
    portfolio = PortfolioState(
        total_value=0,
        positions={},
        sector_exposure={},
        current_drawdown=1.0,
    )
    result = check_trade_allowed(
        constraints=constraints,
        portfolio=portfolio,
        ticker="AAPL",
        sector="Technology",
        trade_value=1000,
    )
    assert result.allowed is False


def test_multiple_constraints_drawdown_checked_first():
    """Drawdown is checked before position/sector limits."""
    constraints = RiskConstraints()
    portfolio = PortfolioState(
        total_value=100_000,
        positions={},
        sector_exposure={},
        current_drawdown=0.25,  # exceeds 20%
    )
    result = check_trade_allowed(
        constraints=constraints,
        portfolio=portfolio,
        ticker="AAPL",
        sector="Technology",
        trade_value=1000,  # small trade, but drawdown blocks it
    )
    assert result.allowed is False
    assert result.violation == ConstraintViolation.DRAWDOWN_LIMIT


def test_sell_always_allowed():
    """Sells (negative trade_value) reduce risk and are always allowed."""
    constraints = RiskConstraints()
    portfolio = PortfolioState(
        total_value=100_000,
        positions={"AAPL": 5000},
        sector_exposure={"Technology": 0.05},
        current_drawdown=0.25,  # even during forced de-risk
    )
    result = check_trade_allowed(
        constraints=constraints,
        portfolio=portfolio,
        ticker="AAPL",
        sector="Technology",
        trade_value=-3000,
    )
    assert result.allowed is True
