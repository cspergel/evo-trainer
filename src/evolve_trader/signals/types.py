"""SignalEvent type system — the common interface for all signal sources.

Every signal source (EDGAR 13F, Form 4, congressional, options, etc.)
produces SignalEvent objects conforming to this schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class SignalType(Enum):
    """Classification of signal intent."""

    REGIME_READ = "regime_read"  # Sector tilts, macro positioning
    CONVICTION = "conviction"  # High-confidence directional bet
    EVENT_DRIVEN = "event_driven"  # Catalyst-based (earnings, filings)
    THESIS = "thesis"  # Long-term structural view


@dataclass
class DecayProfile:
    """Per-source decay configuration."""

    initial_confidence: float  # Starting confidence (0.0-1.0)
    half_life_days: float  # Days until confidence halves
    decay_type: str = "exponential"  # "exponential", "linear", "step"
    min_confidence: float = 0.0  # Floor — never decays below this


@dataclass
class SignalEvent:
    """A typed signal from any ingestion source.

    This is the canonical contract that all signal sources must produce.
    """

    source: str  # e.g., "edgar_13f", "capitol_trades", "form4_insider"
    source_entity: str  # e.g., "Warren Buffett", "Nancy Pelosi"
    timestamp: datetime  # when the signal was generated/ingested
    confidence: float  # 0.0-1.0
    decay_profile: DecayProfile
    signal_type: SignalType
    payload: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    trade_date: datetime | None = None  # when the actual trade occurred
    filing_date: datetime | None = None  # when the filing was made public
