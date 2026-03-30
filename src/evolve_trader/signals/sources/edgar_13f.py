"""SEC EDGAR 13F signal source.

Wraps piboufilings to parse 13F-HR quarterly holdings filings.
Produces SignalEvent objects for portfolio changes by tracked managers.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from evolve_trader.signals.decay import BUFFETT_13F_DECAY
from evolve_trader.signals.types import DecayProfile, SignalEvent, SignalType

# Named manager watchlist: CIK -> name
MANAGER_WATCHLIST: dict[str, str] = {
    "1067983": "Warren Buffett (Berkshire Hathaway)",
    "1350694": "Ray Dalio (Bridgewater)",
    "1336528": "Bill Ackman (Pershing Square)",
    "1029160": "George Soros (Soros Fund Mgmt)",
    "1535392": "Stanley Druckenmiller (Duquesne)",
    "1649339": "Michael Burry (Scion Asset)",
    "1167483": "Chase Coleman (Tiger Global)",
    "1099281": "David Tepper (Appaloosa)",
    "1061768": "Seth Klarman (Baupost)",
    "911974": "Howard Marks (Oaktree)",
    "1040273": "Dan Loeb (Third Point)",
    "921669": "Carl Icahn (Icahn Enterprises)",
}


def parse_13f_holdings_csv(csv_path: str | Path) -> pd.DataFrame:
    """Parse a 13F holdings CSV produced by piboufilings.

    Expected columns: cik, company_name, cusip, value, shares, ...
    """
    path = Path(csv_path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def holdings_to_signals(
    holdings: pd.DataFrame,
    filer_cik: str,
    filer_name: str,
    filing_date: datetime,
    report_date: datetime,
    decay_profile: DecayProfile | None = None,
) -> list[SignalEvent]:
    """Convert 13F holdings into SignalEvent objects.

    Each holding with a CUSIP becomes a signal. Large position changes
    relative to prior filings would be CONVICTION signals; overall sector
    tilts are REGIME_READ signals.
    """
    if holdings.empty:
        return []

    profile = decay_profile or BUFFETT_13F_DECAY
    signals: list[SignalEvent] = []

    for _, row in holdings.iterrows():
        cusip = str(row.get("cusip", ""))
        value = float(row.get("value", 0))
        shares = float(row.get("shares", 0))
        issuer = str(row.get("company_name", row.get("issuer", "")))

        if not cusip or value <= 0:
            continue

        signals.append(
            SignalEvent(
                source="edgar_13f",
                source_entity=filer_name,
                timestamp=datetime.now(UTC),
                confidence=profile.initial_confidence,
                decay_profile=profile,
                signal_type=SignalType.CONVICTION,
                trade_date=report_date,
                filing_date=filing_date,
                payload={
                    "cusip": cusip,
                    "issuer": issuer,
                    "value_thousands": value,
                    "shares": shares,
                },
                metadata={
                    "cik": filer_cik,
                    "filer": filer_name,
                },
            )
        )

    return signals


class Edgar13FSource:
    """EDGAR 13F signal source using piboufilings."""

    @property
    def name(self) -> str:
        return "edgar_13f"

    @property
    def description(self) -> str:
        return "SEC EDGAR 13F quarterly institutional holdings"

    async def fetch_signals(self) -> list[SignalEvent]:
        """Fetch latest 13F signals.

        In production, this would:
        1. Call piboufilings.get_filings() for tracked CIKs
        2. Parse the output CSVs
        3. Diff against prior quarter's holdings
        4. Produce SignalEvents for significant changes

        Requires piboufilings data to be downloaded first.
        """
        # TODO: Wire up piboufilings.get_filings() for automated polling
        return []
