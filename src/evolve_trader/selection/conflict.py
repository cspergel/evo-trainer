"""Signal conflict resolution.

Handles opposing signals using source-weighted confidence averaging.
Falls back to Capital Preservation when sources disagree equally.
"""

from __future__ import annotations

from dataclasses import dataclass

from evolve_trader.selection.scoring import SourceScorecard
from evolve_trader.signals.types import SignalEvent


@dataclass
class ConflictResult:
    """Result of conflict resolution between opposing signals."""

    resolved: bool
    net_direction: str  # "buy", "sell", "neutral"
    confidence: float
    conflicting_sources: list[str]
    resolution_method: str


def resolve_signal_conflicts(
    signals: list[SignalEvent],
    source_scorecards: dict[str, SourceScorecard] | None = None,
    dominance_threshold: float = 0.65,
) -> ConflictResult:
    """Resolve conflicts between opposing signals.

    Uses source-weighted confidence: a Tier 1 source with 3.0x weight
    dominates a Tier 3 source with 1.0x weight even at equal confidence.

    Args:
        signals: Active signal events.
        source_scorecards: Optional source scoring data. If provided,
            signal influence is weighted by effective_weight.
        dominance_threshold: Proportion threshold (0-1) for one side to win.
    """
    buy_weight = 0.0
    sell_weight = 0.0
    buy_sources: list[str] = []
    sell_sources: list[str] = []

    for signal in signals:
        action = str(signal.payload.get("action", "")).upper()
        # Weight by source quality if scorecards available
        source_weight = 1.0
        if source_scorecards and signal.source in source_scorecards:
            source_weight = source_scorecards[signal.source].effective_weight

        effective = signal.confidence * source_weight

        if action in ("BUY", "PURCHASE"):
            buy_weight += effective
            buy_sources.append(signal.source_entity)
        elif action in ("SELL", "SALE"):
            sell_weight += effective
            sell_sources.append(signal.source_entity)

    total = buy_weight + sell_weight
    if total == 0:
        return ConflictResult(
            resolved=True,
            net_direction="neutral",
            confidence=0.0,
            conflicting_sources=[],
            resolution_method="no_signals",
        )

    has_conflict = bool(buy_sources and sell_sources)

    if not has_conflict:
        direction = "buy" if buy_weight > 0 else "sell"
        return ConflictResult(
            resolved=True,
            net_direction=direction,
            confidence=max(buy_weight, sell_weight) / total,
            conflicting_sources=[],
            resolution_method="no_conflict",
        )

    dominance_ratio = max(buy_weight, sell_weight) / total

    if dominance_ratio > dominance_threshold:
        direction = "buy" if buy_weight > sell_weight else "sell"
        return ConflictResult(
            resolved=True,
            net_direction=direction,
            confidence=dominance_ratio,
            conflicting_sources=sell_sources if direction == "buy" else buy_sources,
            resolution_method="dominance",
        )

    return ConflictResult(
        resolved=False,
        net_direction="neutral",
        confidence=dominance_ratio,
        conflicting_sources=buy_sources + sell_sources,
        resolution_method="unresolved_conflict",
    )
