"""Tests for domain-to-ORM converters."""

from datetime import UTC, datetime

from evolve_trader.core.analyzer import TradeResult
from evolve_trader.core.version_dag import EvolutionEvent, EvolutionMode
from evolve_trader.db.converters import (
    evolution_event_to_record,
    signal_event_to_record,
    trade_result_to_log,
)
from evolve_trader.signals.types import DecayProfile, SignalEvent, SignalType


def test_trade_result_to_log():
    """TradeResult converts to TradeLog with all fields."""
    trade = TradeResult(
        ticker="AAPL",
        entry_price=150.0,
        exit_price=160.0,
        shares=10,
        entry_date="2025-01-01",
        exit_date="2025-01-15",
        reasoning="Momentum signal",
    )
    log = trade_result_to_log(
        trade,
        strategy_skill="momentum-v1",
        regime_label="risk-on",
        signal_sources=["edgar_13f"],
    )
    assert log.ticker == "AAPL"
    assert log.strategy_skill == "momentum-v1"
    assert log.pnl == trade.pnl
    assert log.signal_sources == ["edgar_13f"]
    assert log.rationale_summary == "Momentum signal"


def test_signal_event_to_record():
    """SignalEvent converts to SignalEventRecord with decay flattened."""
    signal = SignalEvent(
        source="congressional",
        source_entity="Pelosi",
        timestamp=datetime(2025, 3, 15, tzinfo=UTC),
        confidence=0.85,
        decay_profile=DecayProfile(0.85, 20, "exponential"),
        signal_type=SignalType.CONVICTION,
        payload={"ticker": "NVDA"},
        metadata={"party": "D"},
    )
    record = signal_event_to_record(signal)
    assert record.source == "congressional"
    assert record.decay_type == "exponential"
    assert record.half_life_days == 20
    assert record.signal_type == "conviction"
    assert record.payload["ticker"] == "NVDA"


def test_evolution_event_to_record():
    """EvolutionEvent converts to EvolutionEventRecord."""
    event = EvolutionEvent(
        parent="momentum-v1",
        child="momentum-v2",
        mode=EvolutionMode.FIX,
        reason="Poor Sharpe ratio",
        metrics={"sharpe_before": 0.3},
    )
    record = evolution_event_to_record(event)
    assert record.event_type == "FIX"
    assert record.parent_skill == "momentum-v1"
    assert record.child_skill == "momentum-v2"
    assert record.performance_before == {"sharpe_before": 0.3}
