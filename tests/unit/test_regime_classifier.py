"""Tests for the basic regime classifier."""

from datetime import UTC, datetime

from evolve_trader.regime.classifier import classify_regime
from evolve_trader.signals.types import DecayProfile, SignalEvent, SignalType


def _make_signal(action: str, confidence: float = 0.8, sector: str = "") -> SignalEvent:
    """Helper to create a test signal."""
    return SignalEvent(
        source="test",
        source_entity="test_entity",
        timestamp=datetime(2025, 3, 15, tzinfo=UTC),
        confidence=confidence,
        decay_profile=DecayProfile(initial_confidence=confidence, half_life_days=30),
        signal_type=SignalType.CONVICTION,
        payload={"action": action},
        metadata={"sector": sector} if sector else {},
    )


def test_strong_buy_signals_produce_risk_on():
    """Mostly buy signals → risk-on regime."""
    signals = [
        _make_signal("BUY", 0.9),
        _make_signal("BUY", 0.8),
        _make_signal("BUY", 0.7),
        _make_signal("SELL", 0.3),
    ]
    regime = classify_regime(signals)
    assert regime.primary_regime == "risk-on"
    assert regime.momentum_state == "strengthening"


def test_strong_sell_signals_produce_risk_off():
    """Mostly sell signals → risk-off regime."""
    signals = [
        _make_signal("SELL", 0.9),
        _make_signal("SELL", 0.8),
        _make_signal("SELL", 0.7),
        _make_signal("BUY", 0.2),
    ]
    regime = classify_regime(signals)
    assert regime.primary_regime == "risk-off"
    assert regime.momentum_state == "weakening"


def test_mixed_signals_produce_transitional():
    """Balanced buy/sell → transitional regime."""
    signals = [
        _make_signal("BUY", 0.5),
        _make_signal("SELL", 0.5),
    ]
    regime = classify_regime(signals)
    assert regime.primary_regime == "transitional"


def test_no_signals_produce_low_confidence_transitional():
    """No signals → transitional with low confidence."""
    regime = classify_regime([])
    assert regime.primary_regime == "transitional"
    assert regime.confidence < 0.5


def test_sector_bias_detected():
    """Top sector by signal weight becomes the sector bias."""
    signals = [
        _make_signal("BUY", 0.9, sector="Technology"),
        _make_signal("BUY", 0.8, sector="Technology"),
        _make_signal("BUY", 0.3, sector="Healthcare"),
    ]
    regime = classify_regime(signals)
    assert "Technology" in regime.sector_bias


def test_confidence_increases_with_signal_count():
    """More signals increase confidence."""
    few = classify_regime([_make_signal("BUY", 0.9)])
    many = classify_regime([_make_signal("BUY", 0.9) for _ in range(10)])
    assert many.confidence >= few.confidence
