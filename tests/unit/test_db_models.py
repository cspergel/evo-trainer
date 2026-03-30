"""Tests for database models and repositories."""

from datetime import UTC, datetime

from sqlalchemy.orm import Session

from evolve_trader.db.engine import create_db_engine, create_tables, get_session_factory
from evolve_trader.db.models import (
    EvolutionEventRecord,
    LLMUsageRecord,
    SignalEventRecord,
    TradeLog,
)
from evolve_trader.db.repositories import (
    EvolutionEventRepository,
    LLMUsageRepository,
    SignalEventRepository,
    TradeLogRepository,
)


def _make_session() -> Session:
    """Create an in-memory SQLite session for testing."""
    engine = create_db_engine("sqlite:///:memory:")
    create_tables(engine)
    factory = get_session_factory(engine)
    return factory()


def test_trade_log_roundtrip():
    """TradeLog can be inserted and retrieved."""
    session = _make_session()
    repo = TradeLogRepository(session)

    trade = TradeLog(
        strategy_skill="momentum-v1",
        ticker="AAPL",
        direction="BUY",
        quantity=10.0,
        entry_price=150.0,
        entry_date=datetime(2025, 1, 1, tzinfo=UTC),
    )
    repo.add(trade)
    session.commit()

    retrieved = repo.get_by_ticker("AAPL")
    assert len(retrieved) == 1
    assert retrieved[0].strategy_skill == "momentum-v1"


def test_signal_event_roundtrip():
    """SignalEventRecord can be inserted and queried by source."""
    session = _make_session()
    repo = SignalEventRepository(session)

    signal = SignalEventRecord(
        source="edgar_13f",
        source_entity="Warren Buffett",
        timestamp=datetime(2025, 3, 15, tzinfo=UTC),
        confidence=0.85,
        decay_type="linear",
        half_life_days=90,
        signal_type="CONVICTION",
        payload={"ticker": "AAPL", "action": "BUY"},
        metadata_={"sector": "Technology"},
    )
    repo.add(signal)
    session.commit()

    results = repo.get_by_source("edgar_13f")
    assert len(results) == 1
    assert results[0].source_entity == "Warren Buffett"
    assert results[0].confidence == 0.85


def test_evolution_event_lineage():
    """EvolutionEventRecord lineage can be traced."""
    session = _make_session()
    repo = EvolutionEventRepository(session)

    repo.add(
        EvolutionEventRecord(
            event_type="SEED",
            parent_skill=None,
            child_skill="momentum-v1",
            trigger_reason="Initial seed",
        )
    )
    repo.add(
        EvolutionEventRecord(
            event_type="FIX",
            parent_skill="momentum-v1",
            child_skill="momentum-v2",
            trigger_reason="Poor Sharpe ratio",
            performance_before={"sharpe": 0.3},
        )
    )
    session.commit()

    lineage = repo.get_lineage("momentum-v2")
    assert len(lineage) == 2
    assert lineage[0].child_skill == "momentum-v1"
    assert lineage[1].child_skill == "momentum-v2"


def test_llm_usage_cost_tracking():
    """LLM usage repository tracks costs by component."""
    session = _make_session()
    repo = LLMUsageRepository(session)

    repo.add(
        LLMUsageRecord(
            model="claude-sonnet-4",
            component="evolution",
            input_tokens=1000,
            output_tokens=500,
            cost_usd=0.01,
        )
    )
    repo.add(
        LLMUsageRecord(
            model="claude-sonnet-4",
            component="analysis",
            input_tokens=2000,
            output_tokens=800,
            cost_usd=0.02,
        )
    )
    repo.add(
        LLMUsageRecord(
            model="gpt-4o-mini",
            component="evolution",
            input_tokens=500,
            output_tokens=200,
            cost_usd=0.001,
        )
    )
    session.commit()

    assert repo.total_cost() == 0.031
    by_comp = repo.cost_by_component()
    assert by_comp["evolution"] == 0.011
    assert by_comp["analysis"] == 0.02


def test_trade_log_query_by_strategy():
    """Can query trades by strategy name."""
    session = _make_session()
    repo = TradeLogRepository(session)

    for ticker in ["AAPL", "MSFT", "GOOGL"]:
        repo.add(
            TradeLog(
                strategy_skill="momentum-v1",
                ticker=ticker,
                direction="BUY",
                quantity=10.0,
                entry_price=100.0,
                entry_date=datetime(2025, 1, 1, tzinfo=UTC),
            )
        )
    repo.add(
        TradeLog(
            strategy_skill="mean-reversion-v1",
            ticker="TSLA",
            direction="BUY",
            quantity=5.0,
            entry_price=200.0,
            entry_date=datetime(2025, 1, 1, tzinfo=UTC),
        )
    )
    session.commit()

    momentum_trades = repo.get_by_strategy("momentum-v1")
    assert len(momentum_trades) == 3
