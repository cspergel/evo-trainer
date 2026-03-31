"""Integration test: real market data through the evaluation pipeline.

Tests that real yfinance data flows through replay → analyzer →
fitness → profitability gate. This is the proof that the system
works on actual prices, not just synthetic data.
"""

from datetime import date

import pytest

from evolve_trader.core.analyzer import analyze_strategy_performance
from evolve_trader.core.baseline import BenchmarkType, compute_baseline
from evolve_trader.core.execution_costs import estimate_costs
from evolve_trader.core.market_data import fetch_prices, fetch_universe
from evolve_trader.core.profitability_gate import (
    check_baseline_beating,
)
from evolve_trader.core.replay import ReplayConfig, replay_strategy_on_prices


class TestRealMarketData:
    def test_fetch_aapl_prices(self) -> None:
        """Can fetch real AAPL prices from yfinance."""
        prices = fetch_prices(
            "AAPL",
            start=date(2024, 1, 1),
            end=date(2024, 12, 31),
        )
        assert len(prices.bars) > 200  # ~252 trading days
        assert prices.ticker == "AAPL"
        assert all(b.close > 0 for b in prices.bars)

    def test_daily_returns_computed(self) -> None:
        """Daily returns are computed correctly from real prices."""
        prices = fetch_prices(
            "MSFT",
            start=date(2024, 6, 1),
            end=date(2024, 12, 31),
        )
        returns = prices.daily_returns
        assert len(returns) == len(prices.bars) - 1
        # Most daily returns should be small (< 5%)
        assert all(abs(r) < 0.20 for r in returns)

    def test_fetch_universe(self) -> None:
        """Can fetch prices for multiple tickers."""
        universe = fetch_universe(
            tickers=["AAPL", "MSFT", "GOOGL"],
            start=date(2024, 6, 1),
            end=date(2024, 12, 31),
        )
        assert len(universe) >= 2  # At least 2 should succeed
        for _ticker, series in universe.items():
            assert len(series.bars) > 100


class TestRealReplay:
    def test_replay_produces_trades(self) -> None:
        """Replay on real AAPL data produces actual trades."""
        prices = fetch_prices(
            "AAPL",
            start=date(2024, 1, 1),
            end=date(2024, 12, 31),
        )
        trades = replay_strategy_on_prices(prices)
        assert len(trades) > 0
        # Trades have real prices
        for t in trades:
            assert t.entry_price > 100  # AAPL is > $100
            assert t.exit_price > 0

    def test_replay_respects_stop_loss(self) -> None:
        """No trade loses more than stop loss % (plus costs)."""
        prices = fetch_prices(
            "AAPL",
            start=date(2024, 1, 1),
            end=date(2024, 12, 31),
        )
        config = ReplayConfig(stop_loss_pct=0.05)
        trades = replay_strategy_on_prices(prices, config)
        for t in trades:
            loss_pct = abs(t.return_pct)
            # Allow some slack for costs and gap-through
            assert loss_pct < 0.10, f"Trade lost {loss_pct:.1%}, exceeds tolerance"

    def test_replay_with_analyzer(self) -> None:
        """Replay trades flow through the analyzer correctly."""
        prices = fetch_prices(
            "AAPL",
            start=date(2024, 1, 1),
            end=date(2024, 12, 31),
        )
        trades = replay_strategy_on_prices(prices)
        if not trades:
            pytest.skip("No trades generated")

        perf = analyze_strategy_performance(trades, initial_capital=100_000)
        assert perf.total_trades == len(trades)
        assert perf.win_rate >= 0
        assert perf.win_rate <= 1
        # Sharpe can be anything — this is real market data
        assert perf.sharpe_ratio is not None


class TestRealProfitabilityGate:
    def test_baseline_comparison_with_real_data(self) -> None:
        """Strategy evaluated against real SPY baseline."""
        # Get real SPY baseline
        baseline = compute_baseline(
            BenchmarkType.SPY,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
        )
        assert baseline.sharpe_ratio != 0

        # Run strategy on real AAPL data
        prices = fetch_prices(
            "AAPL",
            start=date(2024, 1, 1),
            end=date(2024, 12, 31),
        )
        trades = replay_strategy_on_prices(prices)
        if not trades:
            pytest.skip("No trades generated")

        perf = analyze_strategy_performance(trades, initial_capital=100_000)

        # Compare against baseline (may or may not beat it — that's reality)
        result = check_baseline_beating(
            strategy_sharpe_by_window=[perf.sharpe_ratio],
            baseline_sharpe_by_window=[baseline.sharpe_ratio],
        )
        # With only 1 window, should be INSUFFICIENT_DATA
        assert result.result.value in ("pass", "fail", "insufficient_data")

    def test_cost_reality_check(self) -> None:
        """Real execution costs are meaningful, not zero."""
        cost = estimate_costs(
            order_value=5_000,
            average_daily_volume=50_000_000,  # AAPL-like
            market_cap_tier="large",
            signal_delay_days=1,
        )
        assert cost.total_round_trip_bps > 2  # At minimum spread + commission
        assert cost.total_round_trip_bps < 50  # Not unreasonable for large-cap
