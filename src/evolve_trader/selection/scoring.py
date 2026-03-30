"""Signal source scoring engine with rolling scorecards.

Dynamic credibility system replacing static tier assignments.
Sources are scored on hit rate, regime alignment, and recency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

# Minimum observations before hit rate affects weight
_MIN_OBSERVATIONS_FOR_SCORING = 5


@dataclass
class SourceScorecard:
    """Rolling scorecard for a signal source."""

    source_name: str
    base_tier_weight: float = 1.0  # Tier 1: 3.0, Tier 2: 2.0, Tier 3: 1.0
    hit_rate: float = 0.5  # Rolling hit rate over lookback (default 50%)
    total_signals: int = 0
    winning_signals: int = 0
    regime_alignment_bonus: float = 0.0
    lookback_weeks: int = 12
    last_updated: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def effective_weight(self) -> float:
        """Compute dynamic weight from all scoring components.

        Uses base tier weight * hit rate multiplier, with cold streak
        penalty applied as a 0.5x factor when hit rate < 35%.
        Sources with fewer than MIN_OBSERVATIONS use base tier weight only.
        """
        # Not enough data — use base tier weight
        if self.total_signals < _MIN_OBSERVATIONS_FOR_SCORING:
            return self.base_tier_weight

        # Hit rate multiplier: 50% baseline = 1.0x, 100% = 2.0x, 25% = 0.5x
        hit_rate_multiplier = self.hit_rate / 0.5
        weight = self.base_tier_weight * hit_rate_multiplier
        weight += self.regime_alignment_bonus

        # Cold streak penalty: halve weight when hit rate < 35%
        if self.hit_rate < 0.35:
            weight *= 0.5

        return max(0.0, weight)

    def record_outcome(self, hit: bool) -> None:
        """Record whether a signal's prediction was correct."""
        self.total_signals += 1
        if hit:
            self.winning_signals += 1
        self.hit_rate = self.winning_signals / self.total_signals
        self.last_updated = datetime.now(UTC)


# Default tier assignments for known sources
SOURCE_TIERS: dict[str, float] = {
    # Tier 1 (3.0x): highest-signal sources
    "congressional": 3.0,
    "form4_insider": 3.0,
    # Tier 2 (2.0x): strong institutional signals
    "edgar_13f": 2.0,
    # Tier 3 (1.0x): supplementary signals
    "macro_news": 1.0,
    "options_unusual": 1.0,
}


def create_scorecard(source_name: str) -> SourceScorecard:
    """Create a new scorecard with default tier weight."""
    tier_weight = SOURCE_TIERS.get(source_name, 1.0)
    return SourceScorecard(source_name=source_name, base_tier_weight=tier_weight)
