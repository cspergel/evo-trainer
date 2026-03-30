"""Integration test: full meta-selector pipeline.

Signals → scoring → regime → conflict check → meta-selector → allocation.
"""

from datetime import UTC, datetime

from evolve_trader.regime.classifier import classify_regime
from evolve_trader.selection.conflict import resolve_signal_conflicts
from evolve_trader.selection.meta_selector import MetaSelector
from evolve_trader.selection.scoring import create_scorecard
from evolve_trader.signals.types import DecayProfile, SignalEvent, SignalType


def _make_signal(
    source: str, entity: str, action: str, confidence: float, sector: str = ""
) -> SignalEvent:
    return SignalEvent(
        source=source,
        source_entity=entity,
        timestamp=datetime(2025, 3, 15, tzinfo=UTC),
        confidence=confidence,
        decay_profile=DecayProfile(initial_confidence=confidence, half_life_days=30),
        signal_type=SignalType.CONVICTION,
        payload={"action": action},
        metadata={"sector": sector} if sector else {},
    )


def test_full_pipeline_bullish_signals():
    """Bullish signals → risk-on regime → momentum strategies selected."""
    # 1. Create signals
    signals = [
        _make_signal("congressional", "Pelosi", "BUY", 0.85, "Technology"),
        _make_signal("edgar_13f", "Buffett", "BUY", 0.80, "Technology"),
        _make_signal("form4_insider", "Cluster", "BUY", 0.75, "Technology"),
    ]

    # 2. Score sources
    cards = {s.source: create_scorecard(s.source) for s in signals}
    for card in cards.values():
        for _ in range(8):
            card.record_outcome(hit=True)

    # 3. Classify regime
    regime = classify_regime(signals)
    assert regime.primary_regime == "risk-on"

    # 4. Check for conflicts
    conflict = resolve_signal_conflicts(signals)
    assert conflict.resolved

    # 5. Select strategies
    selector = MetaSelector(
        available_strategies=[
            "trend-following-v1",
            "mean-reversion-v1",
            "breakout-v1",
            "capital-preservation",
        ],
    )
    result = selector.select(regime, signals, signal_conflicts=not conflict.resolved)
    assert not result.capital_preservation_active
    assert result.allocations[0].strategy in ("trend-following-v1", "breakout-v1")


def test_full_pipeline_conflicting_signals():
    """Conflicting signals → unresolved → capital preservation."""
    signals = [
        _make_signal("congressional", "Pelosi", "BUY", 0.85),
        _make_signal("edgar_13f", "Buffett", "SELL", 0.80),
    ]

    conflict = resolve_signal_conflicts(signals)
    regime = classify_regime(signals)

    selector = MetaSelector(
        available_strategies=["momentum-v1", "capital-preservation"],
    )
    result = selector.select(regime, signals, signal_conflicts=not conflict.resolved)
    assert result.capital_preservation_active


def test_graceful_degradation_no_signals():
    """No signals → low confidence → capital preservation."""
    regime = classify_regime([])
    selector = MetaSelector(
        available_strategies=["momentum-v1", "capital-preservation"],
    )
    result = selector.select(regime, [])
    assert result.capital_preservation_active
