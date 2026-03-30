"""Orchestrator adjustment log — auditable record of all decisions.

Every orchestrator recommendation, whether applied or deferred,
is logged here with structured rationale and cited metrics.

Per profitability contract section 9: LLMs record structured
rationales, not free-form trade discretion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum


class AdjustmentType(Enum):
    """Types of orchestrator adjustments."""

    EVOLUTION_PACE = "evolution_pace"  # Speed up or slow down evolution
    STRATEGY_WEIGHT = "strategy_weight"  # Adjust meta-selector weights
    SIGNAL_WEIGHT = "signal_weight"  # Adjust source scoring
    REGIME_OVERRIDE = "regime_override"  # Override regime classification
    RISK_TIGHTENING = "risk_tightening"  # Tighten constraints (not relax)
    PROMOTION_HOLD = "promotion_hold"  # Hold promotion pending review
    FREEZE_COMPONENT = "freeze_component"  # Stop evolution on a component


class AdjustmentStatus(Enum):
    """Whether an adjustment was applied or deferred."""

    PROPOSED = "proposed"
    APPLIED = "applied"
    DEFERRED = "deferred"
    REJECTED = "rejected"


@dataclass
class Adjustment:
    """A single orchestrator adjustment record."""

    adjustment_type: AdjustmentType
    status: AdjustmentStatus
    rationale: str  # Structured reason, not free-form
    metrics_cited: dict[str, float] = field(default_factory=dict)
    proposed_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    applied_at: datetime | None = None
    counterfactual_result: dict[str, float] | None = None


class AdjustmentLog:
    """Append-only log of all orchestrator decisions."""

    def __init__(self) -> None:
        self._entries: list[Adjustment] = []

    def propose(
        self,
        adjustment_type: AdjustmentType,
        rationale: str,
        metrics_cited: dict[str, float] | None = None,
    ) -> Adjustment:
        """Record a proposed adjustment."""
        adj = Adjustment(
            adjustment_type=adjustment_type,
            status=AdjustmentStatus.PROPOSED,
            rationale=rationale,
            metrics_cited=metrics_cited or {},
        )
        self._entries.append(adj)
        return adj

    def apply(self, adjustment: Adjustment) -> None:
        """Mark an adjustment as applied."""
        adjustment.status = AdjustmentStatus.APPLIED
        adjustment.applied_at = datetime.now(UTC)

    def defer(self, adjustment: Adjustment, reason: str = "") -> None:
        """Mark an adjustment as deferred."""
        adjustment.status = AdjustmentStatus.DEFERRED
        if reason:
            adjustment.rationale += f" [Deferred: {reason}]"

    def reject(self, adjustment: Adjustment, reason: str = "") -> None:
        """Mark an adjustment as rejected."""
        adjustment.status = AdjustmentStatus.REJECTED
        if reason:
            adjustment.rationale += f" [Rejected: {reason}]"

    def get_all(self) -> list[Adjustment]:
        """Get all logged adjustments."""
        return list(self._entries)

    def get_applied(self) -> list[Adjustment]:
        """Get only applied adjustments."""
        return [a for a in self._entries if a.status == AdjustmentStatus.APPLIED]

    def get_pending(self) -> list[Adjustment]:
        """Get proposed but not yet decided adjustments."""
        return [a for a in self._entries if a.status == AdjustmentStatus.PROPOSED]
