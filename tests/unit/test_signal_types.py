"""Tests for the SignalEvent type system and decay library."""

import pytest

from evolve_trader.signals.decay import (
    BUFFETT_13F_DECAY,
    CONGRESSIONAL_DECAY,
    compute_decayed_confidence,
    get_decay_profile,
)
from evolve_trader.signals.registry import SourceRegistry
from evolve_trader.signals.types import DecayProfile, SignalEvent, SignalType

# --- SignalEvent type tests ---


def test_signal_event_creation():
    """SignalEvent can be created with all required fields."""
    profile = DecayProfile(initial_confidence=0.8, half_life_days=30)
    signal = SignalEvent(
        source="edgar_13f",
        source_entity="Warren Buffett",
        timestamp=pytest.importorskip("datetime").datetime(2025, 3, 15),
        confidence=0.85,
        decay_profile=profile,
        signal_type=SignalType.CONVICTION,
        payload={"ticker": "AAPL", "action": "BUY"},
    )
    assert signal.source == "edgar_13f"
    assert signal.signal_type == SignalType.CONVICTION


def test_signal_type_enum():
    """SignalType enum has all expected values."""
    assert SignalType.REGIME_READ.value == "regime_read"
    assert SignalType.CONVICTION.value == "conviction"
    assert SignalType.EVENT_DRIVEN.value == "event_driven"
    assert SignalType.THESIS.value == "thesis"


# --- Decay function tests ---


def test_exponential_decay_at_half_life():
    """Exponential decay reaches ~50% at half-life."""
    profile = DecayProfile(initial_confidence=1.0, half_life_days=30, decay_type="exponential")
    confidence = compute_decayed_confidence(profile, days_elapsed=30)
    assert abs(confidence - 0.5) < 0.01


def test_linear_decay_at_half_life():
    """Linear decay reaches ~50% at half-life."""
    profile = DecayProfile(initial_confidence=1.0, half_life_days=30, decay_type="linear")
    confidence = compute_decayed_confidence(profile, days_elapsed=30)
    assert abs(confidence - 0.5) < 0.01


def test_linear_decay_reaches_zero():
    """Linear decay reaches zero at 2x half-life."""
    profile = DecayProfile(initial_confidence=1.0, half_life_days=30, decay_type="linear")
    confidence = compute_decayed_confidence(profile, days_elapsed=60)
    assert confidence == 0.0


def test_decay_respects_min_confidence():
    """Decay never goes below min_confidence."""
    profile = DecayProfile(
        initial_confidence=0.8, half_life_days=10, decay_type="exponential", min_confidence=0.1
    )
    confidence = compute_decayed_confidence(profile, days_elapsed=1000)
    assert confidence == 0.1


def test_no_decay_at_zero_days():
    """No time elapsed means full confidence."""
    profile = DecayProfile(initial_confidence=0.9, half_life_days=30)
    assert compute_decayed_confidence(profile, days_elapsed=0) == 0.9


def test_congressional_decay_profile():
    """Congressional: medium confidence, 20-day half-life, exponential."""
    profile = CONGRESSIONAL_DECAY
    assert profile.initial_confidence == 0.70
    assert profile.half_life_days == 20
    assert profile.decay_type == "exponential"


def test_buffett_13f_decay_profile():
    """Buffett 13F: high confidence, 90-day half-life, slow linear."""
    profile = BUFFETT_13F_DECAY
    assert profile.initial_confidence == 0.85
    assert profile.half_life_days == 90


def test_get_decay_profile_known():
    """Known source returns its profile."""
    assert get_decay_profile("edgar_13f") == BUFFETT_13F_DECAY
    assert get_decay_profile("congressional") == CONGRESSIONAL_DECAY


def test_get_decay_profile_unknown():
    """Unknown source raises ValueError."""
    with pytest.raises(ValueError, match="Unknown signal source"):
        get_decay_profile("nonexistent_source")


# --- Source registry tests ---


def test_registry_register_and_list():
    """Sources can be registered and listed."""

    class FakeSource:
        @property
        def name(self) -> str:
            return "fake_source"

        @property
        def description(self) -> str:
            return "A fake source for testing"

        async def fetch_signals(self) -> list[SignalEvent]:
            return []

    registry = SourceRegistry()
    registry.register(FakeSource())

    assert "fake_source" in registry.list_sources()
    assert registry.get("fake_source") is not None


def test_registry_health_tracking():
    """Registry tracks source health."""

    class FakeSource:
        @property
        def name(self) -> str:
            return "test_source"

        @property
        def description(self) -> str:
            return "Test"

        async def fetch_signals(self) -> list[SignalEvent]:
            return []

    registry = SourceRegistry()
    registry.register(FakeSource())

    registry.record_success("test_source", 5)
    health = registry.get_health("test_source")
    assert health is not None
    assert health.status == "healthy"
    assert health.total_signals_fetched == 5

    for _ in range(5):
        registry.record_failure("test_source")

    health = registry.get_health("test_source")
    assert health is not None
    assert health.status == "unhealthy"
    assert "test_source" not in registry.get_healthy_sources()
