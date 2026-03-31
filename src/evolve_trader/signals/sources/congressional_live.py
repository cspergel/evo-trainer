"""Live congressional trading data — fetches real House disclosures.

Uses the congressional-trading package's XML index parser to find
PTR filings, then converts to our SignalEvent format.

For Phase 1 wiring: fetches the filing index, extracts trade metadata,
produces SignalEvents. Full PDF parsing deferred until proven valuable.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import httpx

from evolve_trader.signals.sources.congressional import (
    CongressionalTrade,
    congressional_trade_to_signal,
)
from evolve_trader.signals.types import SignalEvent

# House Clerk XML index URL
_HOUSE_FD_INDEX_URL = "https://disclosures-clerk.house.gov/public_disc/financial-pdfs/{year}FD.xml"


def fetch_house_filing_index(year: int | None = None) -> list[dict[str, str]]:
    """Fetch the House Financial Disclosure XML index for a given year.

    Returns a list of filing records with keys:
    doc_id, prefix, first, last, filing_type, filing_date, etc.
    """
    if year is None:
        year = datetime.now(UTC).year

    url = _HOUSE_FD_INDEX_URL.format(year=year)

    try:
        from congressional_trading.scraper.downloader import parse_xml_index

        resp = httpx.get(url, timeout=30, follow_redirects=True)
        resp.raise_for_status()
        filings = parse_xml_index(resp.content)
        return filings  # type: ignore[return-value]
    except Exception as e:
        return [{"error": str(e)}]


def filter_ptr_filings(filings: list[dict[str, str]]) -> list[dict[str, str]]:
    """Filter to only Periodic Transaction Reports (PTRs)."""
    return [
        f for f in filings if f.get("filing_type", "").upper() in ("P", "PTR") and "error" not in f
    ]


def filing_to_congressional_trade(
    filing: dict[str, str],
) -> CongressionalTrade | None:
    """Convert a House filing index record to a CongressionalTrade.

    The filing index has member name, filing date, and doc_id
    but NOT individual trade details (those are in the PDF).
    We create a basic signal from the filing metadata.
    """
    first = filing.get("first", "")
    last = filing.get("last", "")
    if not last:
        return None

    name = f"{first} {last}".strip()
    filing_date_str = filing.get("filing_date", "")

    try:
        filing_date = datetime.strptime(filing_date_str, "%m/%d/%Y").replace(tzinfo=UTC)
    except (ValueError, TypeError):
        filing_date = datetime.now(UTC)

    # Estimate trade date as ~30 days before filing (STOCK Act delay)
    trade_date = filing_date - timedelta(days=30)

    return CongressionalTrade(
        member_name=name,
        party="",  # Not in filing index
        state="",
        chamber="House",
        ticker="",  # Not in filing index — would need PDF parsing
        transaction_type="disclosure",
        amount_range="",
        trade_date=trade_date,
        filing_date=filing_date,
        committees=[],
    )


def fetch_recent_congressional_signals(
    year: int | None = None,
    limit: int = 50,
) -> list[SignalEvent]:
    """Fetch real congressional trading signals from House Clerk.

    Returns SignalEvents for recent PTR filings.
    """
    filings = fetch_house_filing_index(year)
    ptrs = filter_ptr_filings(filings)

    signals: list[SignalEvent] = []
    for filing in ptrs[:limit]:
        trade = filing_to_congressional_trade(filing)
        if trade and trade.member_name:
            signal = congressional_trade_to_signal(trade)
            signals.append(signal)

    return signals
