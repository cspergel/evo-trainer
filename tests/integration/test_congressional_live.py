"""Integration test: real congressional data from House Clerk."""

from evolve_trader.regime.classifier import classify_regime
from evolve_trader.signals.sources.congressional_live import (
    fetch_house_filing_index,
    fetch_recent_congressional_signals,
    filter_ptr_filings,
)


def test_fetch_house_index():
    """Can fetch the House Clerk filing index."""
    filings = fetch_house_filing_index(2025)
    assert len(filings) > 100  # Hundreds of filings per year
    assert "error" not in filings[0]


def test_filter_ptrs():
    """PTR filtering reduces total filings to transaction reports."""
    filings = fetch_house_filing_index(2025)
    ptrs = filter_ptr_filings(filings)
    assert len(ptrs) > 0
    assert len(ptrs) < len(filings)  # Filtered down


def test_fetch_signals():
    """Real congressional signals are produced."""
    signals = fetch_recent_congressional_signals(2025, limit=10)
    assert len(signals) > 0
    assert signals[0].source == "congressional"
    assert signals[0].source_entity  # Has a name


def test_signals_flow_to_regime():
    """Real congressional signals feed into regime classifier."""
    signals = fetch_recent_congressional_signals(2025, limit=20)
    regime = classify_regime(signals)
    # With disclosure-type signals (no BUY/SELL), regime will be transitional
    assert regime.primary_regime in ("risk-on", "risk-off", "transitional")
    assert regime.confidence >= 0
