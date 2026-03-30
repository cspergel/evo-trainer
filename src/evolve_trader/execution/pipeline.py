"""3-gate execution pipeline.

Gate 1: Immutable risk constraints (automatic, non-bypassable)
Gate 2: Paper trading shadow (always on)
Gate 3: Graduated approval (starts manual, graduates to auto)

Per profitability contract: paper/live deviation is tracked at Gate 2.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from evolve_trader.core.execution_costs import CostEstimate, estimate_costs
from evolve_trader.core.profitability_gate import check_capacity
from evolve_trader.core.risk_constraints import (
    PortfolioState,
    RiskConstraints,
    check_trade_allowed,
)
from evolve_trader.execution.trade_intent import TradeIntent


class GateDecision(Enum):
    """Result of a gate check."""

    APPROVED = "approved"
    BLOCKED = "blocked"
    PENDING_APPROVAL = "pending_approval"


@dataclass
class GateResult:
    """Result from a single gate."""

    gate_name: str
    decision: GateDecision
    message: str


@dataclass
class PipelineResult:
    """Full result from the 3-gate pipeline."""

    intent: TradeIntent
    gate_results: list[GateResult]
    cost_estimate: CostEstimate | None = None
    approved: bool = False

    @property
    def blocked_at(self) -> str | None:
        for g in self.gate_results:
            if g.decision == GateDecision.BLOCKED:
                return g.gate_name
        return None


class ApprovalMode(Enum):
    """How Gate 3 operates."""

    MANUAL = "manual"  # Human must approve every trade
    AUTO_HIGH_CONFIDENCE = "auto_high_confidence"  # Auto if confidence > threshold
    AUTO_ALL = "auto_all"  # Auto-approve all (full-live stage)


@dataclass
class PipelineConfig:
    """Configuration for the execution pipeline."""

    approval_mode: ApprovalMode = ApprovalMode.MANUAL
    auto_approval_confidence: float = 0.85
    average_daily_volume: float = 0.0  # For capacity check


def run_pipeline(
    intent: TradeIntent,
    portfolio: PortfolioState,
    config: PipelineConfig,
    constraints: RiskConstraints | None = None,
) -> PipelineResult:
    """Run a TradeIntent through the 3-gate execution pipeline.

    Gate 1: Risk constraints (automatic, non-bypassable)
    Gate 2: Paper shadow + cost estimate (always on)
    Gate 3: Approval gate (graduated based on mode)
    """
    if constraints is None:
        constraints = RiskConstraints()

    result = PipelineResult(intent=intent, gate_results=[])

    # --- Gate 1: Immutable risk constraints ---
    trade_value = intent.notional_value
    if intent.direction == "SELL":
        trade_value = -trade_value

    risk_check = check_trade_allowed(
        constraints=constraints,
        portfolio=portfolio,
        ticker=intent.ticker,
        sector=str(intent.position_impact.get("sector", "Unknown")),
        trade_value=trade_value,
    )

    if not risk_check.allowed:
        result.gate_results.append(
            GateResult(
                gate_name="gate1_risk",
                decision=GateDecision.BLOCKED,
                message=risk_check.message,
            )
        )
        return result

    result.gate_results.append(
        GateResult(
            gate_name="gate1_risk",
            decision=GateDecision.APPROVED,
            message="Risk constraints satisfied",
        )
    )

    # --- Gate 2: Paper shadow + cost estimation ---
    cost = estimate_costs(
        order_value=abs(trade_value),
        average_daily_volume=config.average_daily_volume,
        market_cap_tier="large",  # per scope constraint
    )
    result.cost_estimate = cost
    intent.estimated_cost_bps = cost.total_round_trip_bps

    # Capacity check (profitability contract section 4)
    if config.average_daily_volume > 0:
        cap_check = check_capacity(abs(trade_value), config.average_daily_volume)
        if cap_check.result.value != "pass":
            result.gate_results.append(
                GateResult(
                    gate_name="gate2_paper",
                    decision=GateDecision.BLOCKED,
                    message=f"Capacity: {cap_check.message}",
                )
            )
            return result

    result.gate_results.append(
        GateResult(
            gate_name="gate2_paper",
            decision=GateDecision.APPROVED,
            message=f"Paper shadow active. Cost: {cost.total_round_trip_bps:.1f}bps",
        )
    )

    # --- Gate 3: Approval ---
    if config.approval_mode == ApprovalMode.AUTO_ALL:
        result.gate_results.append(
            GateResult(
                gate_name="gate3_approval",
                decision=GateDecision.APPROVED,
                message="Auto-approved (full-live mode)",
            )
        )
        result.approved = True

    elif config.approval_mode == ApprovalMode.AUTO_HIGH_CONFIDENCE:
        if intent.confidence >= config.auto_approval_confidence:
            result.gate_results.append(
                GateResult(
                    gate_name="gate3_approval",
                    decision=GateDecision.APPROVED,
                    message=(
                        f"Auto-approved (confidence {intent.confidence:.0%} "
                        f">= {config.auto_approval_confidence:.0%})"
                    ),
                )
            )
            result.approved = True
        else:
            result.gate_results.append(
                GateResult(
                    gate_name="gate3_approval",
                    decision=GateDecision.PENDING_APPROVAL,
                    message=(
                        f"Needs approval (confidence {intent.confidence:.0%} "
                        f"< {config.auto_approval_confidence:.0%})"
                    ),
                )
            )

    else:
        # MANUAL mode — always needs human approval
        result.gate_results.append(
            GateResult(
                gate_name="gate3_approval",
                decision=GateDecision.PENDING_APPROVAL,
                message="Manual approval required",
            )
        )

    return result
