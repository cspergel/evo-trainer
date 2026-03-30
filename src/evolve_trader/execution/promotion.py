"""Promotion pipeline — the single shared stage model.

5 stages from paper training to full live. This is the only
promotion pipeline in the system; later phases extend it, not replace it.

Per profitability contract: promotion requires passing the
ProfitabilityGate at each transition.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class PromotionStage(Enum):
    """Strategy promotion stages."""

    PAPER_TRAINING = "paper_training"  # Min 90 trading days, 50 trades, $0 risk
    PAPER_VALIDATION = "paper_validation"  # Min 60 days, Sharpe > 0.5, DD < 15%
    MICRO_LIVE = "micro_live"  # 5-10% capital, human approval every trade
    PARTIAL_LIVE = "partial_live"  # 25-50% capital, approval for large trades
    FULL_LIVE = "full_live"  # 100% capital, auto-approval within constraints
    KILLED = "killed"  # Strategy permanently retired


@dataclass
class PromotionState:
    """Current promotion state for a strategy."""

    strategy_name: str
    stage: PromotionStage = PromotionStage.PAPER_TRAINING
    days_in_stage: int = 0
    trades_in_stage: int = 0
    stage_sharpe: float = 0.0
    stage_max_drawdown: float = 0.0
    stage_win_rate: float = 0.5
    paper_live_correlation: float | None = None
    promoted_at: datetime | None = None
    demoted_at: datetime | None = None
    consecutive_demotions: int = 0
    days_edge_below_zero: int = 0


# Promotion thresholds per stage
_THRESHOLDS: dict[PromotionStage, dict[str, float]] = {
    PromotionStage.PAPER_TRAINING: {
        "min_days": 90,
        "min_trades": 50,
    },
    PromotionStage.PAPER_VALIDATION: {
        "min_days": 60,
        "min_sharpe": 0.5,
        "max_drawdown": 0.15,
        "min_win_rate": 0.45,
    },
    PromotionStage.MICRO_LIVE: {
        "min_days": 30,
        "max_capital_pct": 0.10,
        "min_paper_live_correlation": 0.80,
    },
    PromotionStage.PARTIAL_LIVE: {
        "min_days": 60,
        "max_capital_pct": 0.50,
    },
}

# Demotion thresholds (any one triggers demotion)
_DEMOTION_TRIGGERS: dict[str, float] = {
    "sharpe_below": 0.3,  # 60-day Sharpe drops below 0.3
    "drawdown_above": 0.12,  # drawdown exceeds 12%
    "paper_live_correlation_below": 0.80,  # paper/live diverges
}


def evaluate_promotion(state: PromotionState) -> PromotionStage:
    """Evaluate whether a strategy should be promoted, demoted, or killed.

    Returns the new stage. Does NOT mutate state — caller decides
    whether to apply the change.
    """
    # Already killed — no recovery
    if state.stage == PromotionStage.KILLED:
        return PromotionStage.KILLED

    # Kill triggers (contract section 8) — any stage
    if state.days_edge_below_zero >= 30:
        return PromotionStage.KILLED
    if state.consecutive_demotions >= 3:
        return PromotionStage.KILLED

    # Check demotion first (any stage except paper training)
    if state.stage != PromotionStage.PAPER_TRAINING:
        if state.stage_sharpe < _DEMOTION_TRIGGERS["sharpe_below"]:
            return _demote(state.stage)
        if state.stage_max_drawdown > _DEMOTION_TRIGGERS["drawdown_above"]:
            return _demote(state.stage)
        if (
            state.paper_live_correlation is not None
            and state.paper_live_correlation < _DEMOTION_TRIGGERS["paper_live_correlation_below"]
        ):
            return _demote(state.stage)

    # Check promotion
    thresholds = _THRESHOLDS.get(state.stage, {})

    if state.stage == PromotionStage.PAPER_TRAINING:
        if state.days_in_stage >= thresholds.get(
            "min_days", 90
        ) and state.trades_in_stage >= thresholds.get("min_trades", 50):
            return PromotionStage.PAPER_VALIDATION

    elif state.stage == PromotionStage.PAPER_VALIDATION:
        if (
            state.days_in_stage >= thresholds.get("min_days", 60)
            and state.stage_sharpe >= thresholds.get("min_sharpe", 0.5)
            and state.stage_max_drawdown <= thresholds.get("max_drawdown", 0.15)
            and state.stage_win_rate >= thresholds.get("min_win_rate", 0.45)
        ):
            return PromotionStage.MICRO_LIVE

    elif state.stage == PromotionStage.MICRO_LIVE:
        if (
            state.days_in_stage >= thresholds.get("min_days", 30)
            and state.paper_live_correlation is not None
            and state.paper_live_correlation >= thresholds.get("min_paper_live_correlation", 0.80)
        ):
            return PromotionStage.PARTIAL_LIVE

    elif state.stage == PromotionStage.PARTIAL_LIVE:  # noqa: SIM102
        if (
            state.days_in_stage >= thresholds.get("min_days", 60)
            and state.stage_sharpe >= 0.5
            and state.stage_max_drawdown <= 0.15
        ):
            return PromotionStage.FULL_LIVE

    # No change
    return state.stage


def _demote(current: PromotionStage) -> PromotionStage:
    """Demote one stage down. Progressive — protects capital."""
    demotion_map = {
        PromotionStage.FULL_LIVE: PromotionStage.PARTIAL_LIVE,
        PromotionStage.PARTIAL_LIVE: PromotionStage.MICRO_LIVE,
        PromotionStage.MICRO_LIVE: PromotionStage.PAPER_VALIDATION,
        PromotionStage.PAPER_VALIDATION: PromotionStage.PAPER_TRAINING,
    }
    return demotion_map.get(current, PromotionStage.PAPER_TRAINING)
