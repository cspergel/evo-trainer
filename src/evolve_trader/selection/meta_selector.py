"""Meta-selector — routes regime + signals to weighted strategy allocations.

Maps current RegimeLabel and active SignalEvents to a weighted set of
strategy skills with capital allocation percentages. Falls back to
Capital Preservation on low confidence or unresolved conflicts.

Integrates source scoring and lifecycle stages: signals from demoted or
candidate sources are excluded, and signal influence is weighted by
source effective_weight.
"""

from __future__ import annotations

from dataclasses import dataclass

from evolve_trader.regime.classifier import RegimeLabel
from evolve_trader.selection.lifecycle import LifecycleStage, SourceLifecycleState
from evolve_trader.selection.scoring import SourceScorecard
from evolve_trader.signals.types import SignalEvent

# Regime string constants (matching classifier output)
RISK_ON = "risk-on"
RISK_OFF = "risk-off"
TRANSITIONAL = "transitional"

CAPITAL_PRESERVATION = "capital-preservation"

# Default regime affinity for seed strategies
DEFAULT_REGIME_AFFINITY: dict[str, dict[str, float]] = {
    "trend-following-v1": {RISK_ON: 0.9, RISK_OFF: 0.1, TRANSITIONAL: 0.3},
    "momentum-sector-rotation-v1": {RISK_ON: 0.85, RISK_OFF: 0.15, TRANSITIONAL: 0.3},
    "moving-average-crossover-v1": {RISK_ON: 0.7, RISK_OFF: 0.2, TRANSITIONAL: 0.4},
    "mean-reversion-v1": {RISK_ON: 0.5, RISK_OFF: 0.6, TRANSITIONAL: 0.7},
    "pairs-trading-v1": {RISK_ON: 0.4, RISK_OFF: 0.5, TRANSITIONAL: 0.8},
    "rsi-divergence-v1": {RISK_ON: 0.5, RISK_OFF: 0.5, TRANSITIONAL: 0.6},
    "value-fundamental-v1": {RISK_ON: 0.4, RISK_OFF: 0.7, TRANSITIONAL: 0.5},
    "earnings-drift-v1": {RISK_ON: 0.6, RISK_OFF: 0.3, TRANSITIONAL: 0.5},
    "breakout-v1": {RISK_ON: 0.8, RISK_OFF: 0.1, TRANSITIONAL: 0.5},
    "defensive-low-volatility-v1": {RISK_ON: 0.2, RISK_OFF: 0.9, TRANSITIONAL: 0.5},
    CAPITAL_PRESERVATION: {RISK_ON: 0.1, RISK_OFF: 0.3, TRANSITIONAL: 0.4},
}

# Lifecycle stages that are eligible to influence selection
_ACTIVE_STAGES = {LifecycleStage.PROBATION, LifecycleStage.ACTIVE}


@dataclass
class StrategyAllocation:
    """A single strategy with its allocated weight."""

    strategy: str
    weight: float  # 0.0 to 1.0


@dataclass
class AllocationResult:
    """Complete allocation output from the meta-selector."""

    allocations: list[StrategyAllocation]
    regime: RegimeLabel
    confidence: float
    capital_preservation_active: bool = False

    @property
    def total_weight(self) -> float:
        return sum(a.weight for a in self.allocations)


def _cap_preservation(regime: RegimeLabel) -> AllocationResult:
    return AllocationResult(
        allocations=[StrategyAllocation(CAPITAL_PRESERVATION, 1.0)],
        regime=regime,
        confidence=regime.confidence,
        capital_preservation_active=True,
    )


class MetaSelector:
    """Routes regime + signals to weighted strategy allocations.

    Integrates source scoring and lifecycle: filters signals by lifecycle
    stage, weights signal influence by source effective_weight.
    """

    def __init__(
        self,
        available_strategies: list[str],
        strategy_regime_affinity: dict[str, dict[str, float]] | None = None,
        confidence_threshold: float = 0.5,
        max_strategies: int = 5,
    ) -> None:
        self._strategies = available_strategies
        self._affinity = strategy_regime_affinity or DEFAULT_REGIME_AFFINITY
        self._confidence_threshold = confidence_threshold
        self._max_strategies = max_strategies

    def select(
        self,
        regime: RegimeLabel,
        signals: list[SignalEvent],
        signal_conflicts: bool = False,
        source_scorecards: dict[str, SourceScorecard] | None = None,
        source_lifecycles: dict[str, SourceLifecycleState] | None = None,
    ) -> AllocationResult:
        """Select and weight strategies based on regime, signals, and source quality.

        Signals from demoted/candidate sources are excluded.
        Signal influence is weighted by source effective_weight.
        """
        if regime.confidence < self._confidence_threshold or signal_conflicts:
            return _cap_preservation(regime)

        # Filter signals by lifecycle stage
        active_signals = _filter_by_lifecycle(signals, source_lifecycles)

        # Compute weighted signal boost per source
        signal_boost = _compute_weighted_signal_boost(active_signals, source_scorecards)

        # Score each strategy by regime affinity + signal boost
        scores: list[tuple[str, float]] = []
        for strategy in self._strategies:
            affinity = self._affinity.get(strategy, {})
            score = affinity.get(regime.primary_regime, 0.3)
            scores.append((strategy, score * (1.0 + signal_boost)))

        scores.sort(key=lambda x: x[1], reverse=True)
        top = scores[: self._max_strategies]

        total_score = sum(s for _, s in top)
        if total_score == 0:
            return _cap_preservation(regime)

        allocations = [
            StrategyAllocation(strategy=name, weight=round(score / total_score, 4))
            for name, score in top
        ]

        return AllocationResult(
            allocations=allocations,
            regime=regime,
            confidence=regime.confidence,
        )


def _filter_by_lifecycle(
    signals: list[SignalEvent],
    lifecycles: dict[str, SourceLifecycleState] | None,
) -> list[SignalEvent]:
    """Exclude signals from sources not in active lifecycle stages."""
    if not lifecycles:
        return signals  # No lifecycle data → use all signals
    return [
        s
        for s in signals
        if s.source not in lifecycles or lifecycles[s.source].stage in _ACTIVE_STAGES
    ]


def _compute_weighted_signal_boost(
    signals: list[SignalEvent],
    scorecards: dict[str, SourceScorecard] | None,
) -> float:
    """Compute signal boost weighted by source effective_weight.

    Higher-quality sources contribute more to the boost.
    """
    if not signals:
        return 0.0

    total_weighted_confidence = 0.0
    total_weight = 0.0

    for signal in signals:
        source_weight = 1.0
        if scorecards and signal.source in scorecards:
            source_weight = scorecards[signal.source].effective_weight

        total_weighted_confidence += signal.confidence * source_weight
        total_weight += source_weight

    if total_weight == 0:
        return 0.0

    weighted_avg = total_weighted_confidence / total_weight
    count_factor = min(1.0, len(signals) / 5)
    return weighted_avg * count_factor * 0.5
