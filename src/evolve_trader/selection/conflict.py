"""Signal conflict resolution.

Handles opposing signals using confidence-weighted averaging.
Falls back to Capital Preservation when sources disagree equally.
"""

from __future__ import annotations

from dataclasses import dataclass

from evolve_trader.signals.types import SignalEvent


@dataclass
class ConflictResult:
    """Result of conflict resolution between opposing signals."""

    resolved: bool
    net_direction: str  # "buy", "sell", "neutral"
    confidence: float
    conflicting_sources: list[str]
    resolution_method: str


def resolve_signal_conflicts(signals: list[SignalEvent]) -> ConflictResult:
    """Resolve conflicts between opposing signals.

    Uses confidence-weighted averaging:
    - If one side dramatically outscores → that side wins
    - If scores are similar → conflict is unresolved (capital preservation)
    """
    buy_weight = 0.0
    sell_weight = 0.0
    buy_sources: list[str] = []
    sell_sources: list[str] = []

    for signal in signals:
        action = str(signal.payload.get("action", "")).upper()
        if action in ("BUY", "PURCHASE"):
            buy_weight += signal.confidence
            buy_sources.append(signal.source_entity)
        elif action in ("SELL", "SALE"):
            sell_weight += signal.confidence
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

    # Check if there's a genuine conflict (both sides present)
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

    # Conflict exists — check if one side dominates
    dominance_ratio = max(buy_weight, sell_weight) / total

    if dominance_ratio > 0.65:
        # One side clearly wins
        direction = "buy" if buy_weight > sell_weight else "sell"
        return ConflictResult(
            resolved=True,
            net_direction=direction,
            confidence=dominance_ratio,
            conflicting_sources=sell_sources if direction == "buy" else buy_sources,
            resolution_method="dominance",
        )

    # Signals are too close — unresolved conflict
    return ConflictResult(
        resolved=False,
        net_direction="neutral",
        confidence=dominance_ratio,
        conflicting_sources=buy_sources + sell_sources,
        resolution_method="unresolved_conflict",
    )
