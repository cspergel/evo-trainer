"""Converters between in-memory domain types and ORM models.

Bridges the gap between Phase 1 dataclasses and Phase 2 database records.
"""

from __future__ import annotations

from datetime import datetime

from evolve_trader.core.analyzer import TradeResult
from evolve_trader.core.version_dag import EvolutionEvent
from evolve_trader.db.models import (
    EvolutionEventRecord,
    SignalEventRecord,
    TradeLog,
)
from evolve_trader.db.models import (
    LLMUsageRecord as LLMUsageDBRecord,
)
from evolve_trader.signals.types import SignalEvent


def trade_result_to_log(
    trade: TradeResult,
    strategy_skill: str,
    regime_label: str = "",
    signal_sources: list[str] | None = None,
) -> TradeLog:
    """Convert an in-memory TradeResult to a database TradeLog."""
    return TradeLog(
        strategy_skill=strategy_skill,
        ticker=trade.ticker,
        direction="BUY" if trade.return_pct >= 0 else "SELL",
        quantity=trade.shares,
        entry_price=trade.entry_price,
        exit_price=trade.exit_price,
        entry_date=datetime.fromisoformat(trade.entry_date),
        exit_date=datetime.fromisoformat(trade.exit_date),
        pnl=trade.pnl,
        return_pct=trade.return_pct,
        regime_label=regime_label,
        signal_sources=signal_sources or [],
        rationale_summary=trade.reasoning,
    )


def signal_event_to_record(signal: SignalEvent) -> SignalEventRecord:
    """Convert an in-memory SignalEvent to a database SignalEventRecord."""
    return SignalEventRecord(
        source=signal.source,
        source_entity=signal.source_entity,
        timestamp=signal.timestamp,
        trade_date=signal.trade_date,
        filing_date=signal.filing_date,
        confidence=signal.confidence,
        decay_type=signal.decay_profile.decay_type,
        half_life_days=signal.decay_profile.half_life_days,
        signal_type=signal.signal_type.value,
        payload=signal.payload,
        metadata_=signal.metadata,
    )


def evolution_event_to_record(event: EvolutionEvent) -> EvolutionEventRecord:
    """Convert an in-memory EvolutionEvent to a database record."""
    return EvolutionEventRecord(
        event_type=event.mode.value.upper(),
        parent_skill=event.parent,
        child_skill=event.child,
        trigger_reason=event.reason,
        market_conditions={},
        performance_before=event.metrics,
    )


def llm_usage_to_db(
    model: str,
    component: str,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
) -> LLMUsageDBRecord:
    """Create a database LLM usage record."""
    return LLMUsageDBRecord(
        model=model,
        component=component,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
    )
