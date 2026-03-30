"""Tests for the orchestrator — metrics, adjustments, counterfactual."""

from evolve_trader.orchestrator.adjustment_log import (
    AdjustmentLog,
    AdjustmentStatus,
    AdjustmentType,
)
from evolve_trader.orchestrator.counterfactual import run_counterfactual
from evolve_trader.orchestrator.metrics import (
    SignalSourceMetrics,
    StrategyMetrics,
    SystemSnapshot,
)
from evolve_trader.orchestrator.orchestrator import Orchestrator


class TestMetrics:
    def test_snapshot_counts(self) -> None:
        """Snapshot computes derived counts."""
        snap = SystemSnapshot(
            active_strategies=[
                StrategyMetrics(name="a"),
                StrategyMetrics(name="b"),
            ],
            signal_sources=[
                SignalSourceMetrics(name="x", is_healthy=True),
                SignalSourceMetrics(name="y", is_healthy=False),
            ],
        )
        assert snap.strategy_count == 2
        assert snap.healthy_source_count == 1


class TestAdjustmentLog:
    def test_propose_and_apply(self) -> None:
        """Log tracks proposed and applied adjustments."""
        log = AdjustmentLog()
        adj = log.propose(
            AdjustmentType.EVOLUTION_PACE,
            "Low Sharpe, increase evolution",
            {"sharpe": 0.2},
        )
        assert adj.status == AdjustmentStatus.PROPOSED
        assert len(log.get_pending()) == 1

        log.apply(adj)
        assert adj.status == AdjustmentStatus.APPLIED
        assert len(log.get_applied()) == 1
        assert len(log.get_pending()) == 0

    def test_defer_and_reject(self) -> None:
        """Log supports defer and reject."""
        log = AdjustmentLog()
        adj1 = log.propose(AdjustmentType.STRATEGY_WEIGHT, "Test")
        adj2 = log.propose(AdjustmentType.SIGNAL_WEIGHT, "Test")

        log.defer(adj1, "Not enough evidence")
        log.reject(adj2, "Failed counterfactual")

        assert adj1.status == AdjustmentStatus.DEFERRED
        assert adj2.status == AdjustmentStatus.REJECTED
        assert "Deferred" in adj1.rationale
        assert "Rejected" in adj2.rationale


class TestCounterfactual:
    def test_improvement_passes(self) -> None:
        """Change that improves Sharpe by >= 0.1 passes."""
        result = run_counterfactual(
            baseline_sharpe_by_window=[0.5, 0.6, 0.4],
            adjusted_sharpe_by_window=[0.7, 0.8, 0.6],
        )
        assert result.passes_simplicity_tax
        assert result.improvement > 0.1

    def test_no_improvement_fails(self) -> None:
        """Change that doesn't improve Sharpe fails."""
        result = run_counterfactual(
            baseline_sharpe_by_window=[0.8, 0.9, 0.7],
            adjusted_sharpe_by_window=[0.81, 0.91, 0.71],
        )
        assert not result.passes_simplicity_tax
        assert result.improvement < 0.1

    def test_empty_data(self) -> None:
        """No data returns zero improvement."""
        result = run_counterfactual([], [])
        assert not result.passes_simplicity_tax


class TestOrchestrator:
    def test_proposes_risk_tightening_on_high_drawdown(self) -> None:
        """High drawdown triggers risk tightening proposal."""
        orch = Orchestrator()
        snap = SystemSnapshot(drawdown=0.18)
        proposals = orch.analyze_and_propose(snap)
        types = [p.adjustment_type for p in proposals]
        assert AdjustmentType.RISK_TIGHTENING in types

    def test_proposes_evolution_pace_on_low_sharpe(self) -> None:
        """Low Sharpe with few evolutions triggers pace increase."""
        orch = Orchestrator()
        snap = SystemSnapshot(overall_sharpe=0.1, total_evolution_events=2)
        proposals = orch.analyze_and_propose(snap)
        types = [p.adjustment_type for p in proposals]
        assert AdjustmentType.EVOLUTION_PACE in types

    def test_proposes_freeze_on_unhealthy_sources(self) -> None:
        """Unhealthy signal sources trigger freeze proposal."""
        orch = Orchestrator()
        snap = SystemSnapshot(
            signal_sources=[
                SignalSourceMetrics(name="congressional", is_healthy=False),
            ],
        )
        proposals = orch.analyze_and_propose(snap)
        types = [p.adjustment_type for p in proposals]
        assert AdjustmentType.FREEZE_COMPONENT in types

    def test_proposes_promotion_hold_on_low_correlation(self) -> None:
        """Low paper/live correlation triggers promotion hold."""
        orch = Orchestrator()
        snap = SystemSnapshot(paper_live_correlation=0.72)
        proposals = orch.analyze_and_propose(snap)
        types = [p.adjustment_type for p in proposals]
        assert AdjustmentType.PROMOTION_HOLD in types

    def test_no_proposals_when_healthy(self) -> None:
        """Healthy system produces no proposals."""
        orch = Orchestrator()
        snap = SystemSnapshot(
            drawdown=0.05,
            overall_sharpe=0.8,
            total_evolution_events=10,
            paper_live_correlation=0.95,
        )
        proposals = orch.analyze_and_propose(snap)
        assert len(proposals) == 0

    def test_validate_applies_on_pass(self) -> None:
        """Counterfactual pass → adjustment applied."""
        orch = Orchestrator()
        adj = orch.log.propose(AdjustmentType.STRATEGY_WEIGHT, "Test")
        result = orch.validate_adjustment(
            adj,
            baseline_sharpe=[0.5, 0.6, 0.4],
            adjusted_sharpe=[0.7, 0.8, 0.6],
        )
        assert result.passes_simplicity_tax
        assert adj.status == AdjustmentStatus.APPLIED

    def test_validate_defers_on_fail(self) -> None:
        """Counterfactual fail → adjustment deferred."""
        orch = Orchestrator()
        adj = orch.log.propose(AdjustmentType.SIGNAL_WEIGHT, "Test")
        result = orch.validate_adjustment(
            adj,
            baseline_sharpe=[0.8, 0.9, 0.7],
            adjusted_sharpe=[0.81, 0.91, 0.71],
        )
        assert not result.passes_simplicity_tax
        assert adj.status == AdjustmentStatus.DEFERRED
