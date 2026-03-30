"""Meta-selector — routes regime + signals to weighted strategy allocations.

Maps current RegimeLabel and active SignalEvents to a weighted set of
strategy skills with capital allocation percentages. Falls back to
Capital Preservation on low confidence or unresolved conflicts.
"""

from __future__ import annotations

from dataclasses import dataclass

from evolve_trader.regime.classifier import RegimeLabel
from evolve_trader.signals.types import SignalEvent

# Regime string constants (matching classifier output)
RISK_ON = "risk-on"
RISK_OFF = "risk-off"
TRANSITIONAL = "transitional"

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
    "capital-preservation": {RISK_ON: 0.1, RISK_OFF: 0.3, TRANSITIONAL: 0.4},
}


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


class MetaSelector:
    """Routes regime + signals to weighted strategy allocations.

    Ensemble deployment: can activate multiple strategies simultaneously.
    Falls back to Capital Preservation on low confidence.
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
    ) -> AllocationResult:
        """Select and weight strategies based on regime and signals.

        Returns Capital Preservation when:
        - Regime confidence is below threshold
        - Signal conflicts are unresolved
        """
        # Fall back to capital preservation on low confidence or conflicts
        if regime.confidence < self._confidence_threshold or signal_conflicts:
            return AllocationResult(
                allocations=[StrategyAllocation("capital-preservation", 1.0)],
                regime=regime,
                confidence=regime.confidence,
                capital_preservation_active=True,
            )

        # Score each strategy by regime affinity
        scores: list[tuple[str, float]] = []
        for strategy in self._strategies:
            affinity = self._affinity.get(strategy, {})
            score = affinity.get(regime.primary_regime, 0.3)
            # Boost score based on signal strength
            signal_boost = _compute_signal_boost(signals)
            scores.append((strategy, score * (1.0 + signal_boost)))

        # Sort by score, take top N
        scores.sort(key=lambda x: x[1], reverse=True)
        top = scores[: self._max_strategies]

        # Normalize weights to sum to 1.0
        total_score = sum(s for _, s in top)
        if total_score == 0:
            return AllocationResult(
                allocations=[StrategyAllocation("capital-preservation", 1.0)],
                regime=regime,
                confidence=regime.confidence,
                capital_preservation_active=True,
            )

        allocations = [
            StrategyAllocation(strategy=name, weight=round(score / total_score, 4))
            for name, score in top
        ]

        return AllocationResult(
            allocations=allocations,
            regime=regime,
            confidence=regime.confidence,
        )


def _compute_signal_boost(signals: list[SignalEvent]) -> float:
    """Compute a boost factor from active signals.

    More high-confidence signals = higher boost (up to 0.5).
    """
    if not signals:
        return 0.0
    avg_confidence = sum(s.confidence for s in signals) / len(signals)
    count_factor = min(1.0, len(signals) / 5)
    return avg_confidence * count_factor * 0.5
