"""Capital Preservation — the 'do nothing' skill.

Holds cash and makes no trades. Activated when the regime classifier's
confidence is below threshold or when signal source conflicts are unresolved.
The confidence threshold is itself evolvable.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CapitalPreservationConfig:
    """Configuration for Capital Preservation activation."""

    confidence_threshold: float = 0.6


def should_activate_capital_preservation(
    regime_confidence: float,
    config: CapitalPreservationConfig,
    unresolved_conflicts: bool = False,
) -> bool:
    """Determine if Capital Preservation should be the active strategy.

    Returns True (activate Capital Preservation) when:
    - Regime confidence is at or below the threshold
    - There are unresolved signal source conflicts
    """
    if unresolved_conflicts:
        return True
    return regime_confidence <= config.confidence_threshold
