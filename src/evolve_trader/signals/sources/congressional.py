"""Congressional trading signal source.

House data via congressional-trading package (PTR parser).
Senate data via Capitol Trades scraper (future).
Committee enrichment via ProPublica Congress API (future).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum

from evolve_trader.signals.decay import CONGRESSIONAL_DECAY
from evolve_trader.signals.types import DecayProfile, SignalEvent, SignalType


class LeadershipRole(Enum):
    """Congressional leadership roles — the Wei & Zhou inflection point."""

    NONE = "none"
    COMMITTEE_CHAIR = "committee_chair"
    COMMITTEE_RANKING = "committee_ranking"
    SPEAKER = "speaker"
    SPEAKER_EMERITUS = "speaker_emeritus"
    MAJORITY_LEADER = "majority_leader"
    MINORITY_LEADER = "minority_leader"
    WHIP = "whip"


@dataclass
class CongressionalMember:
    """A member on the congressional trading watchlist."""

    name: str
    party: str  # "D" or "R"
    state: str
    chamber: str  # "House" or "Senate"
    committees: list[str]
    leadership_role: LeadershipRole = LeadershipRole.NONE


CONGRESSIONAL_WATCHLIST = [
    CongressionalMember(
        "Nancy Pelosi",
        "D",
        "CA",
        "House",
        ["Intelligence", "Financial Services"],
        LeadershipRole.SPEAKER_EMERITUS,
    ),
    CongressionalMember(
        "Dan Crenshaw",
        "R",
        "TX",
        "House",
        ["Armed Services", "Intelligence"],
    ),
    CongressionalMember(
        "Ron Wyden",
        "D",
        "OR",
        "Senate",
        ["Finance", "Intelligence"],
        LeadershipRole.COMMITTEE_CHAIR,
    ),
    CongressionalMember(
        "Josh Gottheimer",
        "D",
        "NJ",
        "House",
        ["Intelligence", "Financial Services"],
    ),
    CongressionalMember(
        "Marjorie Taylor Greene",
        "R",
        "GA",
        "House",
        [],
    ),
    CongressionalMember(
        "Tommy Tuberville",
        "R",
        "AL",
        "Senate",
        ["Armed Services", "Agriculture"],
    ),
    CongressionalMember(
        "Markwayne Mullin",
        "R",
        "OK",
        "Senate",
        ["Armed Services", "Energy"],
    ),
    CongressionalMember(
        "Warren Davidson",
        "R",
        "OH",
        "House",
        ["Financial Services"],
    ),
    CongressionalMember(
        "Donald Norcross",
        "D",
        "NJ",
        "House",
        ["Armed Services", "Education"],
    ),
    CongressionalMember(
        "Rick Scott",
        "R",
        "FL",
        "Senate",
        ["Armed Services", "Commerce"],
        LeadershipRole.COMMITTEE_RANKING,
    ),
]


@dataclass
class CongressionalTrade:
    """A congressional stock trade disclosure."""

    member_name: str
    party: str
    state: str
    chamber: str
    ticker: str
    transaction_type: str  # "purchase" or "sale"
    amount_range: str  # e.g. "$1,001 - $15,000"
    trade_date: datetime
    filing_date: datetime
    committees: list[str]
    leadership_role: LeadershipRole = LeadershipRole.NONE
    owner: str = "self"  # "self", "spouse", "child"


def congressional_trade_to_signal(
    trade: CongressionalTrade,
    decay_profile: DecayProfile | None = None,
) -> SignalEvent:
    """Convert a congressional trade to a SignalEvent."""
    profile = decay_profile or CONGRESSIONAL_DECAY
    base_confidence = profile.initial_confidence

    # Boost confidence for leadership roles
    if trade.leadership_role != LeadershipRole.NONE:
        base_confidence = min(1.0, base_confidence + 0.15)

    # Check committee relevance (e.g., Armed Services member buying defense)
    committee_relevant = _is_committee_relevant(trade.ticker, trade.committees)
    if committee_relevant:
        base_confidence = min(1.0, base_confidence + 0.10)

    return SignalEvent(
        source="congressional",
        source_entity=trade.member_name,
        timestamp=datetime.now(UTC),
        confidence=base_confidence,
        decay_profile=profile,
        signal_type=SignalType.CONVICTION,
        trade_date=trade.trade_date,
        filing_date=trade.filing_date,
        payload={
            "ticker": trade.ticker,
            "action": trade.transaction_type,
            "amount_range": trade.amount_range,
            "owner": trade.owner,
        },
        metadata={
            "party": trade.party,
            "state": trade.state,
            "chamber": trade.chamber,
            "committees": trade.committees,
            "leadership_role": trade.leadership_role.value,
            "committee_relevant": committee_relevant,
        },
    )


def _is_committee_relevant(ticker: str, committees: list[str]) -> bool:
    """Check if a trade might be informed by committee membership.

    Simple heuristic — will be enriched with sector mapping later.
    """
    # Defense/aerospace tickers + Armed Services/Intelligence committees
    defense_tickers = {"LMT", "RTX", "NOC", "GD", "BA", "LHX"}
    defense_committees = {"Armed Services", "Intelligence"}
    if ticker.upper() in defense_tickers and defense_committees & set(committees):
        return True

    # Finance tickers + Financial Services/Finance committees
    finance_tickers = {"JPM", "GS", "BAC", "MS", "C", "WFC"}
    finance_committees = {"Financial Services", "Finance", "Banking"}
    return bool(ticker.upper() in finance_tickers and finance_committees & set(committees))


class CongressionalTradeSource:
    """Congressional trading signal source."""

    @property
    def name(self) -> str:
        return "congressional"

    @property
    def description(self) -> str:
        return "Congressional stock trades (House + Senate STOCK Act disclosures)"

    async def fetch_signals(self) -> list[SignalEvent]:
        """Fetch latest congressional trading signals.

        House data: via congressional-trading package (PTR parser).
        Senate data: via Capitol Trades scraper (future).
        Committee enrichment: via ProPublica Congress API (future).
        """
        # TODO: Wire up congressional-trading parser for House data
        # TODO: Build Capitol Trades scraper for Senate data
        # TODO: Add ProPublica committee enrichment
        return []
