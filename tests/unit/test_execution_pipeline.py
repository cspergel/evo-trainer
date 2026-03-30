"""Tests for the 3-gate execution pipeline and promotion pipeline."""

from evolve_trader.core.risk_constraints import PortfolioState
from evolve_trader.execution.pipeline import (
    ApprovalMode,
    GateDecision,
    PipelineConfig,
    run_pipeline,
)
from evolve_trader.execution.promotion import (
    PromotionStage,
    PromotionState,
    evaluate_promotion,
)
from evolve_trader.execution.trade_intent import TradeIntent


def _make_intent(**kwargs) -> TradeIntent:  # type: ignore[no-untyped-def]
    defaults = dict(
        ticker="AAPL",
        direction="BUY",
        quantity=10,
        strategy_skill="momentum-v1",
        confidence=0.9,
        regime_label="risk-on",
        position_impact={"sector": "Technology"},
    )
    defaults.update(kwargs)
    return TradeIntent(**defaults)


def _make_portfolio(**kwargs) -> PortfolioState:  # type: ignore[no-untyped-def]
    defaults = dict(
        total_value=100_000,
        positions={},
        sector_exposure={},
        current_drawdown=0.0,
    )
    defaults.update(kwargs)
    return PortfolioState(**defaults)


# --- Pipeline tests ---


class TestPipeline:
    def test_approved_trade_auto_mode(self) -> None:
        """Trade passes all gates in auto-all mode."""
        result = run_pipeline(
            _make_intent(),
            _make_portfolio(),
            PipelineConfig(approval_mode=ApprovalMode.AUTO_ALL),
        )
        assert result.approved
        assert result.blocked_at is None
        assert result.cost_estimate is not None

    def test_blocked_by_risk_constraints(self) -> None:
        """Trade blocked at Gate 1 when drawdown exceeds limit."""
        result = run_pipeline(
            _make_intent(),
            _make_portfolio(current_drawdown=0.25),
            PipelineConfig(approval_mode=ApprovalMode.AUTO_ALL),
        )
        assert not result.approved
        assert result.blocked_at == "gate1_risk"

    def test_manual_mode_needs_approval(self) -> None:
        """Manual mode always returns pending_approval."""
        result = run_pipeline(
            _make_intent(),
            _make_portfolio(),
            PipelineConfig(approval_mode=ApprovalMode.MANUAL),
        )
        assert not result.approved
        gate3 = [g for g in result.gate_results if g.gate_name == "gate3_approval"][0]
        assert gate3.decision == GateDecision.PENDING_APPROVAL

    def test_auto_high_confidence_approves(self) -> None:
        """High confidence auto-approves."""
        result = run_pipeline(
            _make_intent(confidence=0.90),
            _make_portfolio(),
            PipelineConfig(
                approval_mode=ApprovalMode.AUTO_HIGH_CONFIDENCE,
                auto_approval_confidence=0.85,
            ),
        )
        assert result.approved

    def test_auto_high_confidence_rejects_low(self) -> None:
        """Low confidence requires manual approval."""
        result = run_pipeline(
            _make_intent(confidence=0.60),
            _make_portfolio(),
            PipelineConfig(
                approval_mode=ApprovalMode.AUTO_HIGH_CONFIDENCE,
                auto_approval_confidence=0.85,
            ),
        )
        assert not result.approved

    def test_sell_passes_risk_gate(self) -> None:
        """Sells always pass Gate 1 even during drawdown."""
        result = run_pipeline(
            _make_intent(direction="SELL"),
            _make_portfolio(current_drawdown=0.25),
            PipelineConfig(approval_mode=ApprovalMode.AUTO_ALL),
        )
        assert result.approved

    def test_cost_estimate_populated(self) -> None:
        """Cost estimate is computed at Gate 2."""
        result = run_pipeline(
            _make_intent(),
            _make_portfolio(),
            PipelineConfig(approval_mode=ApprovalMode.AUTO_ALL),
        )
        assert result.cost_estimate is not None
        assert result.cost_estimate.total_round_trip_bps > 0


# --- Promotion pipeline tests ---


class TestPromotion:
    def test_paper_training_to_validation(self) -> None:
        """Promote after 90 days and 50 trades."""
        state = PromotionState(
            strategy_name="momentum-v1",
            stage=PromotionStage.PAPER_TRAINING,
            days_in_stage=95,
            trades_in_stage=55,
        )
        assert evaluate_promotion(state) == PromotionStage.PAPER_VALIDATION

    def test_paper_training_not_enough_trades(self) -> None:
        """Stay in training if not enough trades."""
        state = PromotionState(
            strategy_name="momentum-v1",
            stage=PromotionStage.PAPER_TRAINING,
            days_in_stage=100,
            trades_in_stage=30,
        )
        assert evaluate_promotion(state) == PromotionStage.PAPER_TRAINING

    def test_validation_to_micro_live(self) -> None:
        """Promote to micro-live with good Sharpe and low drawdown."""
        state = PromotionState(
            strategy_name="momentum-v1",
            stage=PromotionStage.PAPER_VALIDATION,
            days_in_stage=65,
            stage_sharpe=0.7,
            stage_max_drawdown=0.08,
        )
        assert evaluate_promotion(state) == PromotionStage.MICRO_LIVE

    def test_demotion_on_high_drawdown(self) -> None:
        """Demote when drawdown exceeds 12%."""
        state = PromotionState(
            strategy_name="momentum-v1",
            stage=PromotionStage.PARTIAL_LIVE,
            days_in_stage=30,
            stage_sharpe=0.8,
            stage_max_drawdown=0.15,  # exceeds 12%
        )
        assert evaluate_promotion(state) == PromotionStage.MICRO_LIVE

    def test_demotion_on_low_sharpe(self) -> None:
        """Demote when Sharpe drops below 0.3."""
        state = PromotionState(
            strategy_name="momentum-v1",
            stage=PromotionStage.FULL_LIVE,
            stage_sharpe=0.2,
            stage_max_drawdown=0.05,
        )
        assert evaluate_promotion(state) == PromotionStage.PARTIAL_LIVE

    def test_demotion_on_paper_live_divergence(self) -> None:
        """Demote when paper/live correlation drops below 0.8."""
        state = PromotionState(
            strategy_name="momentum-v1",
            stage=PromotionStage.MICRO_LIVE,
            days_in_stage=35,
            stage_sharpe=1.0,
            stage_max_drawdown=0.05,
            paper_live_correlation=0.65,  # below 0.80
        )
        assert evaluate_promotion(state) == PromotionStage.PAPER_VALIDATION

    def test_micro_to_partial_needs_correlation(self) -> None:
        """Micro-live to partial-live requires paper/live correlation >= 0.8."""
        state = PromotionState(
            strategy_name="momentum-v1",
            stage=PromotionStage.MICRO_LIVE,
            days_in_stage=35,
            stage_sharpe=1.0,
            stage_max_drawdown=0.05,
            paper_live_correlation=0.85,
        )
        assert evaluate_promotion(state) == PromotionStage.PARTIAL_LIVE

    def test_full_lifecycle(self) -> None:
        """Walk through all promotion stages."""
        state = PromotionState(strategy_name="test", stage=PromotionStage.PAPER_TRAINING)

        state.days_in_stage = 95
        state.trades_in_stage = 55
        assert evaluate_promotion(state) == PromotionStage.PAPER_VALIDATION

        state.stage = PromotionStage.PAPER_VALIDATION
        state.days_in_stage = 65
        state.stage_sharpe = 0.8
        state.stage_max_drawdown = 0.08
        assert evaluate_promotion(state) == PromotionStage.MICRO_LIVE

        state.stage = PromotionStage.MICRO_LIVE
        state.days_in_stage = 35
        state.paper_live_correlation = 0.90
        assert evaluate_promotion(state) == PromotionStage.PARTIAL_LIVE

        state.stage = PromotionStage.PARTIAL_LIVE
        state.days_in_stage = 65
        assert evaluate_promotion(state) == PromotionStage.FULL_LIVE
