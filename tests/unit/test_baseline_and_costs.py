"""Tests for BaselineComparator and ExecutionCostModel."""

from datetime import date

import pytest

from evolve_trader.core.baseline import (
    BaselineResult,
    BenchmarkType,
    compute_baseline,
)
from evolve_trader.core.execution_costs import (
    CostEstimate,
    check_edge_vs_cost,
    estimate_costs,
)

# --- BaselineComparator tests ---


class TestBaseline:
    def test_spy_baseline_returns_result(self) -> None:
        """SPY baseline produces a valid result."""
        result = compute_baseline(
            BenchmarkType.SPY,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
        )
        assert isinstance(result, BaselineResult)
        assert result.ticker == "SPY"
        assert result.sharpe_ratio != 0  # Should have some Sharpe

    def test_cash_baseline(self) -> None:
        """Cash benchmark returns risk-free rate."""
        result = compute_baseline(
            BenchmarkType.CASH,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
        )
        assert result.ticker == "CASH"
        assert result.annualized_return == 0.05
        assert result.max_drawdown == 0.0

    def test_sector_etf_baseline(self) -> None:
        """Sector ETF benchmark uses the provided ticker."""
        result = compute_baseline(
            BenchmarkType.SECTOR_ETF,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
            sector_etf="XLK",
        )
        assert isinstance(result, BaselineResult)
        assert result.ticker == "XLK"


# --- ExecutionCostModel tests ---


class TestCosts:
    def test_large_cap_lower_spread(self) -> None:
        """Large-cap has lower spread than small-cap."""
        large = estimate_costs(10_000, market_cap_tier="large")
        small = estimate_costs(10_000, market_cap_tier="small")
        assert large.spread_bps < small.spread_bps

    def test_higher_participation_higher_slippage(self) -> None:
        """Larger order relative to ADV → more slippage."""
        small_order = estimate_costs(10_000, average_daily_volume=100_000_000)
        large_order = estimate_costs(100_000, average_daily_volume=100_000_000)
        assert large_order.slippage_bps > small_order.slippage_bps

    def test_signal_delay_adds_cost(self) -> None:
        """Signal delay adds per-day cost."""
        no_delay = estimate_costs(10_000, signal_delay_days=0)
        with_delay = estimate_costs(10_000, signal_delay_days=30)
        assert with_delay.delay_bps > no_delay.delay_bps
        assert with_delay.total_round_trip_bps > no_delay.total_round_trip_bps

    def test_total_round_trip(self) -> None:
        """Total is sum of components."""
        cost = estimate_costs(10_000, average_daily_volume=10_000_000)
        expected = cost.spread_bps + cost.slippage_bps + cost.commission_bps + cost.delay_bps
        assert cost.total_round_trip_bps == pytest.approx(expected)

    def test_round_trip_pct(self) -> None:
        """Percentage conversion is correct."""
        cost = CostEstimate(spread_bps=10, slippage_bps=5, commission_bps=0.3, delay_bps=0)
        assert cost.total_round_trip_pct == pytest.approx(15.3 / 10_000)

    def test_edge_vs_cost_passes(self) -> None:
        """Edge 3x cost → passes."""
        cost = CostEstimate(spread_bps=5, slippage_bps=3, commission_bps=0.3, delay_bps=0)
        assert check_edge_vs_cost(25, cost) is True  # 25 / 8.3 = 3x

    def test_edge_vs_cost_fails(self) -> None:
        """Edge 1.5x cost → fails."""
        cost = CostEstimate(spread_bps=10, slippage_bps=5, commission_bps=0.3, delay_bps=0)
        assert check_edge_vs_cost(20, cost) is False  # 20 / 15.3 = 1.3x

    def test_congressional_signal_delay(self) -> None:
        """Congressional signals have 30-45 day delay → significant cost."""
        cost = estimate_costs(10_000, signal_delay_days=35)
        assert cost.delay_bps == 70.0  # 35 days * 2 bps/day
        assert cost.total_round_trip_bps > 70  # delay dominates
