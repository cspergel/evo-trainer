"""Integration test: full signal pipeline end-to-end.

Signal source emits → decay applied → stored in DB → regime classifier consumes.
"""

from datetime import UTC, datetime

from evolve_trader.db.engine import create_db_engine, create_tables, get_session_factory
from evolve_trader.db.models import SignalEventRecord
from evolve_trader.db.repositories import SignalEventRepository
from evolve_trader.regime.classifier import classify_regime
from evolve_trader.signals.decay import CONGRESSIONAL_DECAY, compute_decayed_confidence
from evolve_trader.signals.registry import SourceRegistry
from evolve_trader.signals.sources.congressional import (
    CongressionalTrade,
    CongressionalTradeSource,
    LeadershipRole,
    congressional_trade_to_signal,
)
from evolve_trader.signals.sources.edgar_13f import Edgar13FSource
from evolve_trader.signals.sources.edgar_form4 import EdgarForm4Source


def test_full_pipeline_congressional_to_regime():
    """Congressional trade → SignalEvent → DB → decay → regime classification."""
    # 1. Source produces signal
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
        committees=["Intelligence"],
        leadership_role=LeadershipRole.SPEAKER_EMERITUS,
    )
    signal = congressional_trade_to_signal(trade)
    assert signal.source == "congressional"

    # 2. Persist to database
    engine = create_db_engine("sqlite:///:memory:")
    create_tables(engine)
    session = get_session_factory(engine)()
    repo = SignalEventRepository(session)

    record = SignalEventRecord(
        source=signal.source,
        source_entity=signal.source_entity,
        timestamp=signal.timestamp,
        confidence=signal.confidence,
        decay_type=signal.decay_profile.decay_type,
        half_life_days=signal.decay_profile.half_life_days,
        signal_type=signal.signal_type.value,
        payload=signal.payload,
        metadata_=signal.metadata,
    )
    repo.add(record)
    session.commit()

    # 3. Verify persistence
    stored = repo.get_by_source("congressional")
    assert len(stored) == 1
    assert stored[0].source_entity == "Nancy Pelosi"

    # 4. Apply decay (simulate 10 days later)
    decayed_confidence = compute_decayed_confidence(CONGRESSIONAL_DECAY, days_elapsed=10)
    assert decayed_confidence < CONGRESSIONAL_DECAY.initial_confidence

    # 5. Feed to regime classifier
    regime = classify_regime([signal])
    assert regime.primary_regime in ("risk-on", "transitional", "risk-off")
    assert regime.confidence > 0


def test_source_registry_with_all_sources():
    """All three sources register and report healthy."""
    registry = SourceRegistry()
    registry.register(Edgar13FSource())
    registry.register(EdgarForm4Source())
    registry.register(CongressionalTradeSource())

    assert len(registry.list_sources()) == 3
    assert "edgar_13f" in registry.list_sources()
    assert "form4_insider" in registry.list_sources()
    assert "congressional" in registry.list_sources()

    # All start healthy
    healthy = registry.get_healthy_sources()
    assert len(healthy) == 3
