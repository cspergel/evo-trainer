"""Tests for the post-execution financial analyzer."""

import numpy as np
import pytest

from evolve_trader.core.analyzer import (
    TradeResult,
    analyze_failure_mode,
    analyze_strategy_performance,
    compute_max_drawdown,
    compute_sharpe_ratio,
)


def test_sharpe_ratio_known_values():
    """Sharpe ratio computation matches hand-calculated values."""
    # Generate returns with known mean and std
    rng = np.random.default_rng(42)
    daily_returns = list(rng.normal(loc=0.001, scale=0.01, size=252))
    sharpe = compute_sharpe_ratio(daily_returns, risk_free_rate=0.0)
    # With mean~0.001, std~0.01: annualized Sharpe ~ 1.58
    assert 0.5 < sharpe < 3.0  # reasonable range for these params


def test_sharpe_ratio_constant_returns():
    """Constant returns (zero std) produce zero Sharpe."""
    daily_returns = [0.001] * 100
    assert compute_sharpe_ratio(daily_returns) == 0.0


def test_sharpe_ratio_negative():
    """Negative returns produce negative Sharpe."""
    rng = np.random.default_rng(99)
    daily_returns = list(rng.normal(loc=-0.002, scale=0.01, size=100))
    sharpe = compute_sharpe_ratio(daily_returns, risk_free_rate=0.0)
    assert sharpe < 0


def test_sharpe_ratio_empty():
    """Empty returns produce zero Sharpe."""
    assert compute_sharpe_ratio([]) == 0.0


def test_max_drawdown_known_values():
    """Max drawdown matches hand-calculated value."""
    # Portfolio: 100 -> 120 -> 90 -> 110
    # Max drawdown = (120 - 90) / 120 = 25%
    equity_curve = [100, 110, 120, 100, 90, 95, 110]
    mdd = compute_max_drawdown(equity_curve)
    assert abs(mdd - 0.25) < 0.01


def test_max_drawdown_no_drawdown():
    """Monotonically increasing equity has zero drawdown."""
    equity_curve = [100, 101, 102, 103, 104]
    mdd = compute_max_drawdown(equity_curve)
    assert mdd == 0.0


def test_analyze_strategy_performance():
    """Full strategy analysis produces all required metrics."""
    trades = [
        TradeResult(
            ticker="AAPL",
            entry_price=150,
            exit_price=160,
            shares=10,
            entry_date="2025-01-01",
            exit_date="2025-01-15",
        ),
        TradeResult(
            ticker="MSFT",
            entry_price=300,
            exit_price=290,
            shares=5,
            entry_date="2025-01-05",
            exit_date="2025-01-20",
        ),
        TradeResult(
            ticker="GOOGL",
            entry_price=140,
            exit_price=155,
            shares=8,
            entry_date="2025-01-10",
            exit_date="2025-01-25",
        ),
    ]
    perf = analyze_strategy_performance(trades, initial_capital=100000)

    assert perf.win_rate == pytest.approx(2 / 3, abs=0.01)
    assert perf.total_trades == 3
    assert perf.mean_return > 0  # net positive trades
    assert perf.variance > 0  # non-zero variance with mixed outcomes
    assert perf.max_drawdown >= 0
    assert perf.total_pnl > 0  # net profitable


def test_analyze_empty_trades():
    """Empty trade list produces zero metrics."""
    perf = analyze_strategy_performance([], initial_capital=100000)
    assert perf.total_trades == 0
    assert perf.win_rate == 0.0


def test_analyze_failure_mode_entry_failure():
    """Failure tracing identifies entry logic as the problem for large losses."""
    trades = [
        TradeResult(
            ticker="TSLA",
            entry_price=200,
            exit_price=170,
            shares=10,
            entry_date="2025-01-01",
            exit_date="2025-01-02",
            reasoning="Entry based on RSI crossover",
        ),
    ]
    failure = analyze_failure_mode(trades)
    assert failure is not None
    assert failure.component == "entry_logic"


def test_analyze_failure_mode_no_losses():
    """No failure when all trades are profitable."""
    trades = [
        TradeResult(
            ticker="AAPL",
            entry_price=100,
            exit_price=110,
            shares=10,
            entry_date="2025-01-01",
            exit_date="2025-01-15",
        ),
    ]
    failure = analyze_failure_mode(trades)
    assert failure is None


def test_distributional_metrics():
    """Distribution metrics capture consistency."""
    trades_consistent = [
        TradeResult(
            ticker="SPY",
            entry_price=100,
            exit_price=101,
            shares=100,
            entry_date=f"2025-01-{i:02d}",
            exit_date=f"2025-01-{i + 1:02d}",
        )
        for i in range(1, 21)
    ]
    perf = analyze_strategy_performance(trades_consistent, initial_capital=100000)
    assert perf.variance < 0.01
