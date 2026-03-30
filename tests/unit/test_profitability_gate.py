"""Tests for the profitability gate — runtime contract enforcement."""

from evolve_trader.core.profitability_gate import (
    GateResult,
    check_baseline_beating,
    check_capacity,
    check_executable_alpha,
    check_paper_live_correlation,
    check_scope_constraints,
    check_simplicity_tax,
    check_statistical_bar,
    run_promotion_gate,
)

# --- Section 1: Baseline ---


def test_baseline_beating_passes():
    """Strategy beats baseline in 2 of 3 windows."""
    result = check_baseline_beating(
        strategy_sharpe_by_window=[1.2, 0.8, 1.5],
        baseline_sharpe_by_window=[0.9, 0.9, 0.9],
    )
    assert result.result == GateResult.PASS


def test_baseline_beating_fails():
    """Strategy only beats baseline in 1 of 3 windows."""
    result = check_baseline_beating(
        strategy_sharpe_by_window=[0.5, 0.8, 1.5],
        baseline_sharpe_by_window=[0.9, 0.9, 0.9],
    )
    assert result.result == GateResult.FAIL


def test_baseline_insufficient_windows():
    """Not enough windows to evaluate."""
    result = check_baseline_beating(
        strategy_sharpe_by_window=[1.2],
        baseline_sharpe_by_window=[0.9],
    )
    assert result.result == GateResult.INSUFFICIENT_DATA


# --- Section 2: Executable alpha ---


def test_edge_exceeds_cost():
    """Edge is 3x the cost — passes."""
    result = check_executable_alpha(expected_edge_bps=30, estimated_round_trip_cost_bps=10)
    assert result.result == GateResult.PASS


def test_edge_too_small():
    """Edge is only 1.5x the cost — fails."""
    result = check_executable_alpha(expected_edge_bps=15, estimated_round_trip_cost_bps=10)
    assert result.result == GateResult.FAIL


# --- Section 3: Statistical bar ---


def test_statistical_bar_passes():
    """Sufficient trades, regimes, and PnL distribution."""
    result = check_statistical_bar(
        trades_per_window=[35, 40, 30],
        regime_labels_seen=3,
        pnl_by_window=[1000, 800, 1200],
    )
    assert result.result == GateResult.PASS


def test_statistical_bar_too_few_trades():
    """Not enough trades in one window."""
    result = check_statistical_bar(
        trades_per_window=[35, 10, 30],
        regime_labels_seen=2,
        pnl_by_window=[1000, 200, 800],
    )
    assert result.result == GateResult.FAIL
    assert "Window 1" in result.message


def test_statistical_bar_pnl_concentrated():
    """One window dominates PnL — fails."""
    result = check_statistical_bar(
        trades_per_window=[30, 30, 30],
        regime_labels_seen=2,
        pnl_by_window=[5000, 100, 100],
    )
    assert result.result == GateResult.FAIL
    assert "accounts for" in result.message


def test_statistical_bar_too_few_regimes():
    """Only one regime label — fails."""
    result = check_statistical_bar(
        trades_per_window=[40, 40, 40],
        regime_labels_seen=1,
        pnl_by_window=[1000, 1000, 1000],
    )
    assert result.result == GateResult.FAIL


# --- Section 4: Capacity ---


def test_capacity_within_limit():
    """Small order relative to ADV — passes."""
    result = check_capacity(order_value=50_000, average_daily_volume_dollars=10_000_000)
    assert result.result == GateResult.PASS


def test_capacity_exceeds_limit():
    """Order too large for daily volume — fails."""
    result = check_capacity(order_value=200_000, average_daily_volume_dollars=10_000_000)
    assert result.result == GateResult.FAIL


# --- Section 7: Simplicity tax ---


def test_simplicity_tax_passes():
    """Component adds 0.15 Sharpe — passes."""
    result = check_simplicity_tax(sharpe_with_component=1.0, sharpe_without_component=0.85)
    assert result.result == GateResult.PASS


def test_simplicity_tax_fails():
    """Component adds only 0.05 Sharpe — fails."""
    result = check_simplicity_tax(sharpe_with_component=1.0, sharpe_without_component=0.96)
    assert result.result == GateResult.FAIL


# --- Section 10: Paper/live ---


def test_paper_live_correlation_passes():
    """Correlation above threshold."""
    result = check_paper_live_correlation(0.92)
    assert result.result == GateResult.PASS


def test_paper_live_correlation_fails():
    """Correlation below threshold — auto-demotion."""
    result = check_paper_live_correlation(0.65)
    assert result.result == GateResult.FAIL


# --- Section 6: Scope ---


def test_scope_within_constraints():
    """Within scope limits."""
    result = check_scope_constraints(active_strategies=3, active_signal_sources=2, universe="sp500")
    assert result.result == GateResult.PASS


def test_scope_too_many_strategies():
    """Too many active strategies."""
    result = check_scope_constraints(active_strategies=5, active_signal_sources=2, universe="sp500")
    assert result.result == GateResult.FAIL


def test_scope_wrong_universe():
    """Wrong market universe."""
    result = check_scope_constraints(
        active_strategies=2, active_signal_sources=2, universe="nasdaq100"
    )
    assert result.result == GateResult.FAIL


# --- Full promotion gate ---


def test_full_promotion_passes():
    """Strategy passes all promotion checks."""
    report = run_promotion_gate(
        strategy_sharpe_by_window=[1.2, 1.0, 1.3],
        baseline_sharpe_by_window=[0.8, 0.8, 0.8],
        expected_edge_bps=25,
        estimated_cost_bps=8,
        trades_per_window=[35, 40, 32],
        regime_labels_seen=3,
        pnl_by_window=[1000, 900, 1100],
    )
    assert report.passed
    assert len(report.failed_checks) == 0


def test_full_promotion_fails_on_baseline():
    """Strategy fails baseline check — whole report fails."""
    report = run_promotion_gate(
        strategy_sharpe_by_window=[0.5, 0.6, 0.7],
        baseline_sharpe_by_window=[0.8, 0.8, 0.8],
        expected_edge_bps=25,
        estimated_cost_bps=8,
        trades_per_window=[35, 40, 32],
        regime_labels_seen=3,
        pnl_by_window=[1000, 900, 1100],
    )
    assert not report.passed
    assert any(c.name == "baseline_beating" for c in report.failed_checks)
