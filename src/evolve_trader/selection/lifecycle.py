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
    min_observations_for_observation: int = 5
    min_observations_for_probation: int = 10
    min_hit_rate_for_probation: float = 0.50
    min_hit_rate_for_active: float = 0.50
    demotion_hit_rate: float = 0.30
    demotion_consecutive_periods: int = 2
    re_promotion_hit_rate: float = 0.50
    re_promotion_min_observations: int = 10


def evaluate_promotion(
    state: SourceLifecycleState,
    hit_rate: float,
    total_observations: int,
) -> None:
    """Evaluate and update a source's lifecycle stage.

    Mutates state.stage and related fields. No return value —
    the state object is the single source of truth.
    """
    state.observations = total_observations

    if state.stage == LifecycleStage.CANDIDATE:
        if total_observations >= state.min_observations_for_observation:
            state.stage = LifecycleStage.OBSERVATION

    elif state.stage == LifecycleStage.OBSERVATION:
        if (
            hit_rate >= state.min_hit_rate_for_probation
            and total_observations >= state.min_observations_for_probation
        ):
            state.stage = LifecycleStage.PROBATION
            state.promoted_at = datetime.now(UTC)

    elif state.stage == LifecycleStage.PROBATION:
        if hit_rate >= state.min_hit_rate_for_active:
            state.stage = LifecycleStage.ACTIVE
            state.promoted_at = datetime.now(UTC)
            state.consecutive_underperform_periods = 0
        elif hit_rate < state.demotion_hit_rate:
            state.consecutive_underperform_periods += 1
            if state.consecutive_underperform_periods >= state.demotion_consecutive_periods:
                state.stage = LifecycleStage.DEMOTED
                state.demoted_at = datetime.now(UTC)

    elif state.stage == LifecycleStage.ACTIVE:
        if hit_rate < state.demotion_hit_rate:
            state.consecutive_underperform_periods += 1
            if state.consecutive_underperform_periods >= state.demotion_consecutive_periods:
                state.stage = LifecycleStage.DEMOTED
                state.demoted_at = datetime.now(UTC)
        else:
            state.consecutive_underperform_periods = 0

    elif state.stage == LifecycleStage.DEMOTED:  # noqa: SIM102
        # Re-promotion path: sustained recovery returns to OBSERVATION
        if (
            hit_rate >= state.re_promotion_hit_rate
            and total_observations >= state.re_promotion_min_observations
        ):
            state.stage = LifecycleStage.OBSERVATION
            state.consecutive_underperform_periods = 0
            state.demoted_at = None
