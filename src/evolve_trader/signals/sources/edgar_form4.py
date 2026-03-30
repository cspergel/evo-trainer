"""SEC EDGAR Form 4 insider transaction signal source.

Wraps piboufilings FormSection16Parser for insider trade parsing.
Detects insider clusters: 3+ insiders in same sector buying within 2 weeks.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from evolve_trader.signals.decay import FORM4_INSIDER_DECAY
from evolve_trader.signals.types import DecayProfile, SignalEvent, SignalType


@dataclass
class InsiderTrade:
    """A single insider transaction from Form 4."""

    filer_name: str
    issuer: str
    ticker: str
    transaction_date: datetime
    transaction_code: str  # P=purchase, S=sale
    shares: float
    price: float
    sector: str = ""


def detect_insider_clusters(
    trades: list[InsiderTrade],
    window_days: int = 14,
    min_cluster_size: int = 3,
) -> list[dict[str, object]]:
    """Detect clusters of insider purchases in the same sector.

    A cluster is 3+ insiders at DIFFERENT companies in the same sector
    filing purchases within a 2-week window.
    """
    # Group purchases by sector
    purchases_by_sector: dict[str, list[InsiderTrade]] = defaultdict(list)
    for trade in trades:
        if trade.transaction_code == "P" and trade.sector:
            purchases_by_sector[trade.sector].append(trade)

    clusters: list[dict[str, object]] = []

    for sector, sector_trades in purchases_by_sector.items():
        # Sort by date
        sector_trades.sort(key=lambda t: t.transaction_date)

        # Sliding window
        for i, anchor in enumerate(sector_trades):
            window_end = anchor.transaction_date + timedelta(days=window_days)
            window_trades = [t for t in sector_trades[i:] if t.transaction_date <= window_end]

            # Count unique companies
            unique_companies = {t.issuer for t in window_trades}
            if len(unique_companies) >= min_cluster_size:
                clusters.append(
                    {
                        "sector": sector,
                        "companies": list(unique_companies),
                        "trade_count": len(window_trades),
                        "window_start": anchor.transaction_date.isoformat(),
                        "window_end": window_end.isoformat(),
                    }
                )
                break  # One cluster per sector per scan

    return clusters


def insider_trades_to_signals(
    trades: list[InsiderTrade],
    decay_profile: DecayProfile | None = None,
) -> list[SignalEvent]:
    """Convert insider trades to SignalEvents.

    Individual trades become CONVICTION signals.
    Detected clusters become stronger REGIME_READ signals.
    """
    profile = decay_profile or FORM4_INSIDER_DECAY
    signals: list[SignalEvent] = []

    # Individual trade signals
    for trade in trades:
        signals.append(
            SignalEvent(
                source="form4_insider",
                source_entity=trade.filer_name,
                timestamp=datetime.now(UTC),
                confidence=profile.initial_confidence,
                decay_profile=profile,
                signal_type=SignalType.CONVICTION,
                trade_date=trade.transaction_date,
                payload={
                    "ticker": trade.ticker,
                    "action": "BUY" if trade.transaction_code == "P" else "SELL",
                    "shares": trade.shares,
                    "price": trade.price,
                },
                metadata={
                    "issuer": trade.issuer,
                    "sector": trade.sector,
                    "filer": trade.filer_name,
                },
            )
        )

    # Cluster signals (stronger)
    clusters = detect_insider_clusters(trades)
    for cluster in clusters:
        signals.append(
            SignalEvent(
                source="form4_insider",
                source_entity=f"Insider cluster ({cluster['sector']})",
                timestamp=datetime.now(UTC),
                confidence=min(1.0, profile.initial_confidence + 0.15),
                decay_profile=profile,
                signal_type=SignalType.REGIME_READ,
                payload=cluster,
                metadata={"cluster": True, "sector": str(cluster["sector"])},
            )
        )

    return signals


class EdgarForm4Source:
    """EDGAR Form 4 insider transaction signal source."""

    @property
    def name(self) -> str:
        return "form4_insider"

    @property
    def description(self) -> str:
        return "SEC EDGAR Form 4 insider transactions with cluster detection"

    async def fetch_signals(self) -> list[SignalEvent]:
        """Fetch latest Form 4 signals.

        In production, calls piboufilings for Section 16 filings,
        parses transactions, detects clusters.
        """
        # TODO: Wire up piboufilings FormSection16Parser for automated polling
        return []
