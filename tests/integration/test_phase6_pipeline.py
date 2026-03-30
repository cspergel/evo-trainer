"""Phase 6 integration test: full execution pipeline end-to-end.

Proves the pieces work together as a phase boundary:
strategy → sizing → 3-gate pipeline → cost estimation →
profitability gate → promotion decision.
"""

from datetime import UTC, datetime

from evolve_trader.core.execution_costs import estimate_costs
from evolve_trader.core.profitability_gate import (
    run_promotion_gate,
)
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
from evolve_trader.regime.classifier import classify_regime
from evolve_trader.selection.meta_selector import MetaSelector
from evolve_trader.signals.types import DecayProfile, SignalEvent, SignalType
from evolve_trader.sizing.models import (
    SizingContext,
    SizingMethod,
    SizingSkill,
    compute_sizing,
)


def test_full_pipeline_signal_to_paper_execution():
    """End-to-end: signals → regime → meta-selector → sizing → pipeline → cost check.

    This is the Phase 6 boundary test. Every component participates.
    """
    # 1. Signals arrive (5+ needed for regime confidence > 0.5)
    base_signal = dict(
        timestamp=datetime(2025, 3, 15, tzinfo=UTC),
        confidence=0.85,
        decay_profile=DecayProfile(0.85, 20, "exponential"),
        signal_type=SignalType.CONVICTION,
        payload={"ticker": "AAPL", "action": "BUY"},
        metadata={"sector": "Technology"},
    )
    signals = [
        SignalEvent(source="congressional", source_entity="Pelosi", **base_signal),
        SignalEvent(source="form4_insider", source_entity="Cluster A", **base_signal),
        SignalEvent(source="edgar_13f", source_entity="Buffett", **base_signal),
        SignalEvent(source="form4_insider", source_entity="Cluster B", **base_signal),
        SignalEvent(source="congressional", source_entity="Crenshaw", **base_signal),
    ]

    # 2. Classify regime
    regime = classify_regime(signals)
    assert regime.primary_regime == "risk-on"

    # 3. Meta-selector picks strategies
    selector = MetaSelector(
        available_strategies=[
            "trend-following-v1",
            "mean-reversion-v1",
            "capital-preservation",
        ],
    )
    allocation = selector.select(regime, signals)
    assert not allocation.capital_preservation_active
    top_strategy = allocation.allocations[0].strategy
    top_weight = allocation.allocations[0].weight

    # 4. Sizing determines quantity
    sizing_ctx = SizingContext(
        portfolio_value=100_000,
        current_price=185.0,  # AAPL approximate
        win_rate=0.55,
        avg_win=0.06,
        avg_loss=0.04,
        recent_volatility=0.22,
        regime=regime.primary_regime,
        existing_exposure=0.1,
    )
    skill = SizingSkill("kelly-v1", SizingMethod.KELLY, "Quarter Kelly", {"kelly_fraction": 0.15})
    sizing = compute_sizing(sizing_ctx, skill)
    assert sizing.position_size_pct > 0
    assert sizing.shares > 0

    # 5. Build TradeIntent
    intent = TradeIntent(
        ticker="AAPL",
        direction="BUY",
        quantity=sizing.shares,
        price_estimate=185.0,
        strategy_skill=top_strategy,
        sizing_skill="kelly-v1",
        sizing_rationale=sizing.rationale,
        regime_label=regime.primary_regime,
        regime_confidence=regime.confidence,
        signal_sources=["congressional", "form4_insider"],
        confidence=regime.confidence * top_weight,
        rationale_summary="Congressional + insider cluster buy signal in risk-on regime",
        rationale_evidence={
            "signals": ["Pelosi BUY AAPL", "Insider cluster Technology"],
            "regime": regime.primary_regime,
        },
        position_impact={"sector": "Technology", "pct_of_portfolio": sizing.position_size_pct},
        estimated_cost_bps=0,
        created_at=datetime.now(UTC),
    )
    assert intent.notional_value > 0
    assert intent.client_order_id  # idempotency key exists

    # 6. Run through 3-gate pipeline
    portfolio = PortfolioState(
        total_value=100_000,
        positions={},
        sector_exposure={},
        current_drawdown=0.0,
    )
    pipeline_result = run_pipeline(
        intent,
        portfolio,
        PipelineConfig(
            approval_mode=ApprovalMode.AUTO_HIGH_CONFIDENCE,
            auto_approval_confidence=0.5,
            average_daily_volume=50_000_000,  # AAPL ADV
        ),
    )

    # Gate 1 (risk) should pass — small position, no drawdown
    assert pipeline_result.gate_results[0].decision == GateDecision.APPROVED

    # Gate 2 (paper + cost) should pass — liquid name
    assert pipeline_result.gate_results[1].decision == GateDecision.APPROVED
    assert pipeline_result.cost_estimate is not None
    assert pipeline_result.cost_estimate.total_round_trip_bps > 0

    # Gate 3 (approval) should auto-approve — confidence above threshold
    assert pipeline_result.approved

    # 7. Cost estimate is realistic for AAPL
    cost = pipeline_result.cost_estimate
    assert cost.spread_bps == 2.0  # large-cap
    assert cost.slippage_bps < 10.0  # small order, liquid name
    assert cost.commission_bps == 0.3

    # 8. Profitability gate (would run before promotion)
    gate_report = run_promotion_gate(
        strategy_sharpe_by_window=[0.9, 0.7, 1.1],
        baseline_sharpe_by_window=[0.6, 0.6, 0.6],
        expected_edge_bps=20,
        estimated_cost_bps=cost.total_round_trip_bps,
        trades_per_window=[35, 40, 30],
        regime_labels_seen=2,
        pnl_by_window=[800, 600, 1000],
    )
    assert gate_report.passed


def test_pipeline_blocks_during_drawdown():
    """Pipeline correctly blocks new buys during 20%+ drawdown."""
    intent = TradeIntent(
        ticker="AAPL",
        direction="BUY",
        quantity=10,
        price_estimate=185.0,
        strategy_skill="momentum-v1",
        confidence=0.95,
        created_at=datetime.now(UTC),
        position_impact={"sector": "Technology"},
    )
    portfolio = PortfolioState(
        total_value=80_000,
        positions={},
        sector_exposure={},
        current_drawdown=0.22,
    )
    result = run_pipeline(
        intent,
        portfolio,
        PipelineConfig(approval_mode=ApprovalMode.AUTO_ALL),
    )
    assert not result.approved
    assert result.blocked_at == "gate1_risk"


def test_promotion_requires_statistical_evidence():
    """Strategy cannot promote without meeting profitability gate."""
    # Good promotion state...
    state = PromotionState(
        strategy_name="momentum-v1",
        stage=PromotionStage.PAPER_TRAINING,
        days_in_stage=100,
        trades_in_stage=60,
    )
    # ...meets time/trade thresholds
    new_stage = evaluate_promotion(state)
    assert new_stage == PromotionStage.PAPER_VALIDATION

    # But profitability gate ALSO must pass before actual promotion.
    # If strategy doesn't beat baseline, gate blocks it.
    gate = run_promotion_gate(
        strategy_sharpe_by_window=[0.4, 0.3, 0.5],  # below baseline
        baseline_sharpe_by_window=[0.6, 0.6, 0.6],
        expected_edge_bps=5,
        estimated_cost_bps=10,  # edge < 2x cost
        trades_per_window=[20, 18, 22],  # below 30 minimum
        regime_labels_seen=1,
        pnl_by_window=[100, 50, 200],
    )
    assert not gate.passed
    assert len(gate.failed_checks) >= 1


def test_cost_model_reflects_signal_delay():
    """Congressional signal delay (35 days) adds significant cost."""
    cost = estimate_costs(
        order_value=5_000,
        average_daily_volume=50_000_000,
        market_cap_tier="large",
        signal_delay_days=35,
    )
    # 35 days * 2 bps/day = 70 bps delay alone
    assert cost.delay_bps == 70.0
    assert cost.total_round_trip_bps > 70
    # Edge must be > 2x this cost to be viable
    assert not (20 / cost.total_round_trip_bps >= 2.0)  # 20 bps edge insufficient


def test_killed_strategy_cannot_reenter_pipeline():
    """A killed strategy stays killed regardless of subsequent performance."""
    state = PromotionState(
        strategy_name="failed-strategy",
        stage=PromotionStage.KILLED,
        stage_sharpe=2.0,  # Even with amazing Sharpe
        stage_max_drawdown=0.01,
        days_in_stage=0,
        trades_in_stage=100,
    )
    assert evaluate_promotion(state) == PromotionStage.KILLED
