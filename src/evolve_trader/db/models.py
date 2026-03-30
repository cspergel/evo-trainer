"""SQLAlchemy ORM models for Evolve-Trader.

Supports both SQLite (dev/test) and PostgreSQL (production).
JSON columns use SQLAlchemy's JSON type which maps to JSONB on PostgreSQL.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import DateTime, Float, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import JSON


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    type_annotation_map = {dict[str, Any]: JSON, list[str]: JSON}


class TradeLog(Base):
    """Audit trail for every trade execution."""

    __tablename__ = "trade_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    strategy_skill: Mapped[str] = mapped_column(String(200))
    ticker: Mapped[str] = mapped_column(String(20), index=True)
    direction: Mapped[str] = mapped_column(String(10))  # BUY, SELL, SHORT, COVER
    quantity: Mapped[float] = mapped_column(Float)
    entry_price: Mapped[float] = mapped_column(Float)
    exit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    entry_date: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    exit_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    pnl: Mapped[float | None] = mapped_column(Float, nullable=True)
    return_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    regime_label: Mapped[str | None] = mapped_column(String(100), nullable=True)
    signal_sources: Mapped[list[str]] = mapped_column(JSON, default=list)
    rationale_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )

    __table_args__ = (Index("ix_trade_logs_strategy_date", "strategy_skill", "entry_date"),)


class SignalEventRecord(Base):
    """Persisted signal events from all ingestion sources."""

    __tablename__ = "signal_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(String(100), index=True)
    source_entity: Mapped[str] = mapped_column(String(200))
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    trade_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    filing_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    confidence: Mapped[float] = mapped_column(Float)
    decay_type: Mapped[str] = mapped_column(String(50))
    half_life_days: Mapped[float] = mapped_column(Float)
    signal_type: Mapped[str] = mapped_column(String(50))
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )

    __table_args__ = (Index("ix_signal_events_source_time", "source", "timestamp"),)


class EvolutionEventRecord(Base):
    """Tracks FIX/DERIVED/CAPTURED evolution events with lineage."""

    __tablename__ = "evolution_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_type: Mapped[str] = mapped_column(String(20))  # FIX, DERIVED, CAPTURED
    parent_skill: Mapped[str | None] = mapped_column(String(200), nullable=True)
    child_skill: Mapped[str] = mapped_column(String(200))
    trigger_reason: Mapped[str] = mapped_column(Text)
    market_conditions: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    performance_before: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    performance_after: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class PortfolioSnapshot(Base):
    """Periodic portfolio state snapshots for monitoring."""

    __tablename__ = "portfolio_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    total_value: Mapped[float] = mapped_column(Float)
    cash: Mapped[float] = mapped_column(Float)
    positions: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    sector_exposure: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    drawdown: Mapped[float] = mapped_column(Float)
    sharpe_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class LLMUsageRecord(Base):
    """LLM API call records for cost tracking."""

    __tablename__ = "llm_usage"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model: Mapped[str] = mapped_column(String(100))
    component: Mapped[str] = mapped_column(String(100), index=True)
    input_tokens: Mapped[int] = mapped_column(Integer)
    output_tokens: Mapped[int] = mapped_column(Integer)
    cost_usd: Mapped[float] = mapped_column(Float)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
