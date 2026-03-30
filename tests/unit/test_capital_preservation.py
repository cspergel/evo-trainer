"""Tests for the Capital Preservation (do nothing) skill."""

from evolve_trader.strategies.capital_preservation import (
    CapitalPreservationConfig,
    should_activate_capital_preservation,
)


def test_activates_below_confidence_threshold():
    """Capital Preservation activates when confidence is below threshold."""
    config = CapitalPreservationConfig(confidence_threshold=0.6)
    assert should_activate_capital_preservation(regime_confidence=0.4, config=config) is True


def test_does_not_activate_above_threshold():
    """Capital Preservation does not activate when confidence is sufficient."""
    config = CapitalPreservationConfig(confidence_threshold=0.6)
    assert should_activate_capital_preservation(regime_confidence=0.8, config=config) is False


def test_activates_at_exactly_threshold():
    """At exactly the threshold, Capital Preservation activates (conservative)."""
    config = CapitalPreservationConfig(confidence_threshold=0.6)
    assert should_activate_capital_preservation(regime_confidence=0.6, config=config) is True


def test_activates_on_signal_conflict():
    """Capital Preservation activates when signals conflict."""
    config = CapitalPreservationConfig(confidence_threshold=0.6)
    assert (
        should_activate_capital_preservation(
            regime_confidence=0.9,
            config=config,
            unresolved_conflicts=True,
        )
        is True
    )


def test_threshold_is_configurable():
    """Different thresholds produce different activation behavior."""
    strict = CapitalPreservationConfig(confidence_threshold=0.8)
    loose = CapitalPreservationConfig(confidence_threshold=0.3)

    assert should_activate_capital_preservation(regime_confidence=0.5, config=strict) is True
    assert should_activate_capital_preservation(regime_confidence=0.5, config=loose) is False
