"""Basic regime classifier — consumes SignalEvents, outputs RegimeLabels.

Hand-crafted heuristic for Phase 2. Subject to FIX/DERIVED/CAPTURED
evolution in later phases. Starts monolithic — later phases may
decompose into sub-classifiers.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from evolve_trader.signals.types import SignalEvent


@dataclass
class RegimeLabel:
    """Current market regime classification."""

    primary_regime: str  # "risk-on", "risk-off", "transitional"
    sector_bias: str  # "overweight tech", "underweight financials", etc.
    momentum_state: str  # "strengthening", "weakening", "transitional"
    confidence: float  # 0.0-1.0
    time_horizon: str = "medium-term (4-12 weeks)"
    timestamp: datetime | None = None


def classify_regime(signals: list[SignalEvent]) -> RegimeLabel:
    """Classify the current market regime from active signals.

    Simple heuristic:
    - Count buy vs sell signals weighted by confidence
    - If net bullish → risk-on; net bearish → risk-off; mixed → transitional
    - Sector bias from which sectors appear most in signals
    - Momentum from signal trend direction
    """
    if not signals:
        return RegimeLabel(
            primary_regime="transitional",
            sector_bias="neutral",
            momentum_state="transitional",
            confidence=0.3,
        )

    buy_weight = 0.0
    sell_weight = 0.0
    sector_counts: dict[str, float] = {}

    for signal in signals:
        action = str(signal.payload.get("action", "")).upper()
        weight = signal.confidence

        if action in ("BUY", "PURCHASE"):
            buy_weight += weight
        elif action in ("SELL", "SALE"):
            sell_weight += weight

        # Track sector exposure
        sector = str(signal.metadata.get("sector", ""))
        if sector:
            sector_counts[sector] = sector_counts.get(sector, 0) + weight

    total_weight = buy_weight + sell_weight
    if total_weight == 0:
        return RegimeLabel(
            primary_regime="transitional",
            sector_bias="neutral",
            momentum_state="transitional",
            confidence=0.3,
        )

    buy_ratio = buy_weight / total_weight

    # Determine primary regime
    if buy_ratio > 0.65:
        primary_regime = "risk-on"
        momentum = "strengthening"
    elif buy_ratio < 0.35:
        primary_regime = "risk-off"
        momentum = "weakening"
    else:
        primary_regime = "transitional"
        momentum = "transitional"

    # Determine sector bias
    if sector_counts:
        top_sector = max(sector_counts, key=sector_counts.get)  # type: ignore[arg-type]
        sector_bias = f"overweight {top_sector}"
    else:
        sector_bias = "neutral"

    # Confidence based on signal agreement and count
    agreement = max(buy_ratio, 1 - buy_ratio)
    count_factor = min(1.0, len(signals) / 5)
    confidence = agreement * count_factor

    return RegimeLabel(
        primary_regime=primary_regime,
        sector_bias=sector_bias,
        momentum_state=momentum,
        confidence=round(confidence, 3),
    )
