"""Signal source lifecycle pipeline.

5-stage promotion model for all signal sources:
Candidate → Observation → Probation → Active → Demotion

This is the canonical source-promotion model reused by Phase 9 discovery.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum


class LifecycleStage(Enum):
    """Signal source lifecycle stages."""

    CANDIDATE = "candidate"  # Discovered, tracked, zero weight
    OBSERVATION = "observation"  # Min 5 trades observed, still zero weight
    PROBATION = "probation"  # Hit rate > 50%, low weight (0.5x Tier 3)
    ACTIVE = "active"  # Sustained performance, full tier weight
    DEMOTED = "demoted"  # Hit rate < 30% for 2 periods, weight zeroed


@dataclass
class SourceLifecycleState:
    """Tracks a source's current lifecycle position."""

    source_name: str
    stage: LifecycleStage = LifecycleStage.CANDIDATE
    observations: int = 0
    consecutive_underperform_periods: int = 0
    promoted_at: datetime | None = None
    demoted_at: datetime | None = None

    # Promotion thresholds (configurable)
    min_observations_for_probation: int = 5
    min_hit_rate_for_probation: float = 0.50
    min_hit_rate_for_active: float = 0.50
    demotion_hit_rate: float = 0.30
    demotion_consecutive_periods: int = 2


def evaluate_promotion(
    state: SourceLifecycleState,
    hit_rate: float,
    total_observations: int,
) -> LifecycleStage:
    """Evaluate whether a source should be promoted or demoted.

    Returns the new lifecycle stage based on current performance.
    """
    state.observations = total_observations

    if state.stage == LifecycleStage.CANDIDATE:
        if total_observations >= state.min_observations_for_probation:
            return LifecycleStage.OBSERVATION
        return LifecycleStage.CANDIDATE

    if state.stage == LifecycleStage.OBSERVATION:
        if hit_rate >= state.min_hit_rate_for_probation:
            state.promoted_at = datetime.now(UTC)
            return LifecycleStage.PROBATION
        return LifecycleStage.OBSERVATION

    if state.stage == LifecycleStage.PROBATION:
        if hit_rate >= state.min_hit_rate_for_active:
            state.promoted_at = datetime.now(UTC)
            return LifecycleStage.ACTIVE
        if hit_rate < state.demotion_hit_rate:
            state.consecutive_underperform_periods += 1
            if state.consecutive_underperform_periods >= state.demotion_consecutive_periods:
                state.demoted_at = datetime.now(UTC)
                return LifecycleStage.DEMOTED
        return LifecycleStage.PROBATION

    if state.stage == LifecycleStage.ACTIVE:
        if hit_rate < state.demotion_hit_rate:
            state.consecutive_underperform_periods += 1
            if state.consecutive_underperform_periods >= state.demotion_consecutive_periods:
                state.demoted_at = datetime.now(UTC)
                return LifecycleStage.DEMOTED
        else:
            state.consecutive_underperform_periods = 0
        return LifecycleStage.ACTIVE

    # Demoted sources stay demoted (can be re-promoted manually)
    return state.stage
