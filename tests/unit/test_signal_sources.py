"""Tests for signal sources: EDGAR 13F, Form 4, Congressional."""

from datetime import UTC, datetime

import pandas as pd

from evolve_trader.signals.sources.congressional import (
    CongressionalTrade,
    CongressionalTradeSource,
    LeadershipRole,
    congressional_trade_to_signal,
)
from evolve_trader.signals.sources.edgar_13f import (
    Edgar13FSource,
    holdings_to_signals,
)
from evolve_trader.signals.sources.edgar_form4 import (
    EdgarForm4Source,
    InsiderTrade,
    detect_insider_clusters,
    insider_trades_to_signals,
)
from evolve_trader.signals.types import SignalType

# --- EDGAR 13F tests ---


def test_13f_holdings_to_signals():
    """Holdings DataFrame produces SignalEvents."""
    holdings = pd.DataFrame(
        {
            "cusip": ["037833100", "594918104"],
            "company_name": ["Apple Inc", "Microsoft Corp"],
            "value": [50000, 30000],
            "shares": [100000, 60000],
        }
    )
    signals = holdings_to_signals(
        holdings,
        filer_cik="1067983",
        filer_name="Warren Buffett",
        filing_date=datetime(2025, 5, 15, tzinfo=UTC),
        report_date=datetime(2025, 3, 31, tzinfo=UTC),
    )
    assert len(signals) == 2
    assert signals[0].source == "edgar_13f"
    assert signals[0].source_entity == "Warren Buffett"
    assert signals[0].signal_type == SignalType.CONVICTION


def test_13f_empty_holdings():
    """Empty holdings produce no signals."""
    signals = holdings_to_signals(
        pd.DataFrame(),
        filer_cik="1067983",
        filer_name="Test",
        filing_date=datetime(2025, 5, 15, tzinfo=UTC),
        report_date=datetime(2025, 3, 31, tzinfo=UTC),
    )
    assert len(signals) == 0


def test_13f_source_interface():
    """Edgar13FSource implements SignalSource protocol."""
    source = Edgar13FSource()
    assert source.name == "edgar_13f"
    assert source.description


# --- Form 4 tests ---


def test_insider_cluster_detection():
    """3+ insiders buying in same sector within 2 weeks triggers cluster."""
    trades = [
        InsiderTrade(
            "CEO A",
            "Company A",
            "AAA",
            datetime(2025, 3, 1, tzinfo=UTC),
            "P",
            1000,
            50.0,
            "Technology",
        ),
        InsiderTrade(
            "CEO B",
            "Company B",
            "BBB",
            datetime(2025, 3, 5, tzinfo=UTC),
            "P",
            2000,
            75.0,
            "Technology",
        ),
        InsiderTrade(
            "CEO C",
            "Company C",
            "CCC",
            datetime(2025, 3, 10, tzinfo=UTC),
            "P",
            500,
            30.0,
            "Technology",
        ),
    ]
    clusters = detect_insider_clusters(trades)
    assert len(clusters) >= 1
    assert clusters[0]["sector"] == "Technology"


def test_no_cluster_below_threshold():
    """2 insiders don't trigger a cluster (need 3+)."""
    trades = [
        InsiderTrade(
            "CEO A",
            "Company A",
            "AAA",
            datetime(2025, 3, 1, tzinfo=UTC),
            "P",
            1000,
            50.0,
            "Technology",
        ),
        InsiderTrade(
            "CEO B",
            "Company B",
            "BBB",
            datetime(2025, 3, 5, tzinfo=UTC),
            "P",
            2000,
            75.0,
            "Technology",
        ),
    ]
    clusters = detect_insider_clusters(trades)
    assert len(clusters) == 0


def test_insider_trades_to_signals():
    """Insider trades produce both individual and cluster signals."""
    trades = [
        InsiderTrade(
            "CEO A",
            "Company A",
            "AAA",
            datetime(2025, 3, 1, tzinfo=UTC),
            "P",
            1000,
            50.0,
            "Technology",
        ),
        InsiderTrade(
            "CEO B",
            "Company B",
            "BBB",
            datetime(2025, 3, 5, tzinfo=UTC),
            "P",
            2000,
            75.0,
            "Technology",
        ),
        InsiderTrade(
            "CEO C",
            "Company C",
            "CCC",
            datetime(2025, 3, 10, tzinfo=UTC),
            "P",
            500,
            30.0,
            "Technology",
        ),
    ]
    signals = insider_trades_to_signals(trades)
    # 3 individual + 1 cluster
    assert len(signals) == 4
    cluster_signals = [s for s in signals if s.signal_type == SignalType.REGIME_READ]
    assert len(cluster_signals) == 1


def test_form4_source_interface():
    """EdgarForm4Source implements SignalSource protocol."""
    source = EdgarForm4Source()
    assert source.name == "form4_insider"


# --- Congressional tests ---


def test_congressional_trade_to_signal():
    """Congressional trade produces a SignalEvent."""
    trade = CongressionalTrade(
        member_name="Nancy Pelosi",
        party="D",
        state="CA",
        chamber="House",
        ticker="NVDA",
        transaction_type="purchase",
        amount_range="$1,001 - $15,000",
        trade_date=datetime(2025, 3, 1, tzinfo=UTC),
        filing_date=datetime(2025, 3, 15, tzinfo=UTC),
        committees=["Intelligence", "Financial Services"],
        leadership_role=LeadershipRole.SPEAKER_EMERITUS,
    )
    signal = congressional_trade_to_signal(trade)
    assert signal.source == "congressional"
    assert signal.source_entity == "Nancy Pelosi"
    # Leadership role boosts confidence
    assert signal.confidence > 0.70


def test_congressional_committee_relevance_boosts_confidence():
    """Defense stock by Armed Services member gets confidence boost."""
    trade = CongressionalTrade(
        member_name="Dan Crenshaw",
        party="R",
        state="TX",
        chamber="House",
        ticker="LMT",
        transaction_type="purchase",
        amount_range="$15,001 - $50,000",
        trade_date=datetime(2025, 3, 1, tzinfo=UTC),
        filing_date=datetime(2025, 3, 15, tzinfo=UTC),
        committees=["Armed Services", "Intelligence"],
    )
    signal = congressional_trade_to_signal(trade)
    # Committee relevance + base confidence
    assert signal.confidence > 0.70
    assert signal.metadata["committee_relevant"] is True


def test_congressional_source_interface():
    """CongressionalTradeSource implements SignalSource protocol."""
    source = CongressionalTradeSource()
    assert source.name == "congressional"
