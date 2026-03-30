"""Signal decay function library.

Per-source decay profiles with configurable half-lives.
Confidence decays over time — older signals are less actionable.
"""

from __future__ import annotations

import math

from evolve_trader.signals.types import DecayProfile

# --- Pre-configured decay profiles for known sources ---

BUFFETT_13F_DECAY = DecayProfile(
    initial_confidence=0.85,
    half_life_days=90,
    decay_type="linear",
    min_confidence=0.1,
)

FORM4_INSIDER_DECAY = DecayProfile(
    initial_confidence=0.80,
    half_life_days=30,
    decay_type="linear",
    min_confidence=0.05,
)

CONGRESSIONAL_DECAY = DecayProfile(
    initial_confidence=0.70,
    half_life_days=20,
    decay_type="exponential",
    min_confidence=0.05,
)

ARK_DAILY_DECAY = DecayProfile(
    initial_confidence=0.65,
    half_life_days=10,
    decay_type="exponential",
    min_confidence=0.0,
)

ONCHAIN_WHALE_DECAY = DecayProfile(
    initial_confidence=0.75,
    half_life_days=3,
    decay_type="exponential",
    min_confidence=0.0,
)

OPTIONS_UNUSUAL_DECAY = DecayProfile(
    initial_confidence=0.80,
    half_life_days=2,
    decay_type="exponential",
    min_confidence=0.0,
)

MACRO_NEWS_DECAY = DecayProfile(
    initial_confidence=0.60,
    half_life_days=45,
    decay_type="linear",
    min_confidence=0.1,
)

# Registry mapping source names to their default profiles
DECAY_PROFILES: dict[str, DecayProfile] = {
    "edgar_13f": BUFFETT_13F_DECAY,
    "form4_insider": FORM4_INSIDER_DECAY,
    "congressional": CONGRESSIONAL_DECAY,
    "ark_daily": ARK_DAILY_DECAY,
    "onchain_whale": ONCHAIN_WHALE_DECAY,
    "options_unusual": OPTIONS_UNUSUAL_DECAY,
    "macro_news": MACRO_NEWS_DECAY,
}


def get_decay_profile(source: str) -> DecayProfile:
    """Get the default decay profile for a source."""
    if source not in DECAY_PROFILES:
        raise ValueError(f"Unknown signal source: {source}")
    return DECAY_PROFILES[source]


def compute_decayed_confidence(profile: DecayProfile, days_elapsed: float) -> float:
    """Compute the current confidence after decay.

    Args:
        profile: The decay configuration.
        days_elapsed: Days since the signal was generated.

    Returns:
        Decayed confidence value, floored at min_confidence.
    """
    if days_elapsed <= 0:
        return profile.initial_confidence

    if profile.decay_type == "exponential":
        decay_rate = math.log(2) / profile.half_life_days
        confidence = profile.initial_confidence * math.exp(-decay_rate * days_elapsed)

    elif profile.decay_type == "linear":
        # Linear: reaches zero at 2 * half_life_days
        total_life = profile.half_life_days * 2
        fraction_remaining = max(0.0, 1.0 - days_elapsed / total_life)
        confidence = profile.initial_confidence * fraction_remaining

    else:
        # Default to exponential for unknown types
        decay_rate = math.log(2) / profile.half_life_days
        confidence = profile.initial_confidence * math.exp(-decay_rate * days_elapsed)

    return max(profile.min_confidence, confidence)
