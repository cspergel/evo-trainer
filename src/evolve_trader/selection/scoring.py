"""Signal source scoring engine with rolling scorecards.

Dynamic credibility system replacing static tier assignments.
Sources are scored on hit rate, regime alignment, and recency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass
class SourceScorecard:
    """Rolling scorecard for a signal source."""

    source_name: str
    base_tier_weight: float = 1.0  # Tier 1: 3.0, Tier 2: 2.0, Tier 3: 1.0
    hit_rate: float = 0.5  # Rolling hit rate over lookback
    total_signals: int = 0
    winning_signals: int = 0
    regime_alignment_bonus: float = 0.0
    cold_streak_penalty: float = 0.0  # 0.5x if hit rate < 35%
    lookback_weeks: int = 12
    last_updated: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def effective_weight(self) -> float:
        """Compute dynamic weight from all scoring components."""
        hit_rate_multiplier = self.hit_rate / 0.5 if self.hit_rate > 0 else 0.0
        weight = self.base_tier_weight * hit_rate_multiplier
        weight += self.regime_alignment_bonus
        weight *= 1.0 - self.cold_streak_penalty
        return max(0.0, weight)

    def record_outcome(self, hit: bool) -> None:
        """Record whether a signal's prediction was correct."""
        self.total_signals += 1
        if hit:
            self.winning_signals += 1
        self.hit_rate = self.winning_signals / self.total_signals
        self._update_penalties()
        self.last_updated = datetime.now(UTC)

    def _update_penalties(self) -> None:
        """Update cold streak penalty based on hit rate."""
        if self.total_signals >= 5 and self.hit_rate < 0.35:
            self.cold_streak_penalty = 0.5
        else:
            self.cold_streak_penalty = 0.0


# Default tier assignments for known sources
SOURCE_TIERS: dict[str, float] = {
    # Tier 1 (3.0x): highest-signal sources
    "congressional": 3.0,  # Congressional leadership trades
    "form4_insider": 3.0,  # Insider transaction clusters
    # Tier 2 (2.0x): strong institutional signals
    "edgar_13f": 2.0,  # Quarterly 13F holdings
    # Tier 3 (1.0x): supplementary signals
    "macro_news": 1.0,
    "options_unusual": 1.0,
}


def create_scorecard(source_name: str) -> SourceScorecard:
    """Create a new scorecard with default tier weight."""
    tier_weight = SOURCE_TIERS.get(source_name, 1.0)
    return SourceScorecard(source_name=source_name, base_tier_weight=tier_weight)
