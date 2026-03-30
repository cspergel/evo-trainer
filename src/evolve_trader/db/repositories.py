"""Typed repository pattern for database access.

All database operations go through these repositories.
Callers never use raw SQLAlchemy sessions directly.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from evolve_trader.db.models import (
    EvolutionEventRecord,
    LLMUsageRecord,
    PortfolioSnapshot,
    SignalEventRecord,
    TradeLog,
)


class TradeLogRepository:
    """Repository for trade log records."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, trade: TradeLog) -> TradeLog:
        self._session.add(trade)
        self._session.flush()
        return trade

    def get_by_id(self, trade_id: int) -> TradeLog | None:
        return self._session.get(TradeLog, trade_id)

    def get_by_strategy(self, strategy: str) -> list[TradeLog]:
        stmt = select(TradeLog).where(TradeLog.strategy_skill == strategy)
        return list(self._session.scalars(stmt).all())

    def get_by_ticker(self, ticker: str) -> list[TradeLog]:
        stmt = select(TradeLog).where(TradeLog.ticker == ticker)
        return list(self._session.scalars(stmt).all())


class SignalEventRepository:
    """Repository for signal event records."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, signal: SignalEventRecord) -> SignalEventRecord:
        self._session.add(signal)
        self._session.flush()
        return signal

    def get_by_source(self, source: str, limit: int = 100) -> list[SignalEventRecord]:
        stmt = (
            select(SignalEventRecord)
            .where(SignalEventRecord.source == source)
            .order_by(SignalEventRecord.timestamp.desc())
            .limit(limit)
        )
        return list(self._session.scalars(stmt).all())

    def get_recent(self, hours: int = 24) -> list[SignalEventRecord]:
        cutoff = datetime.now(UTC).replace(
            hour=datetime.now(UTC).hour - min(hours, datetime.now(UTC).hour)
        )
        stmt = (
            select(SignalEventRecord)
            .where(SignalEventRecord.timestamp >= cutoff)
            .order_by(SignalEventRecord.timestamp.desc())
        )
        return list(self._session.scalars(stmt).all())


class EvolutionEventRepository:
    """Repository for evolution event records."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, event: EvolutionEventRecord) -> EvolutionEventRecord:
        self._session.add(event)
        self._session.flush()
        return event

    def get_by_child(self, child_skill: str) -> list[EvolutionEventRecord]:
        stmt = select(EvolutionEventRecord).where(EvolutionEventRecord.child_skill == child_skill)
        return list(self._session.scalars(stmt).all())

    def get_lineage(self, skill_name: str) -> list[EvolutionEventRecord]:
        """Get the full evolution chain for a skill."""
        events: list[EvolutionEventRecord] = []
        current = skill_name
        visited: set[str] = set()

        while current and current not in visited:
            visited.add(current)
            stmt = select(EvolutionEventRecord).where(EvolutionEventRecord.child_skill == current)
            event = self._session.scalars(stmt).first()
            if event:
                events.append(event)
                current = event.parent_skill or ""
            else:
                break

        events.reverse()
        return events


class LLMUsageRepository:
    """Repository for LLM usage records."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, record: LLMUsageRecord) -> LLMUsageRecord:
        self._session.add(record)
        self._session.flush()
        return record

    def total_cost(self) -> float:
        records = self._session.scalars(select(LLMUsageRecord)).all()
        return sum(r.cost_usd for r in records)

    def cost_by_component(self) -> dict[str, float]:
        records = self._session.scalars(select(LLMUsageRecord)).all()
        totals: dict[str, float] = {}
        for r in records:
            totals[r.component] = totals.get(r.component, 0.0) + r.cost_usd
        return totals


class PortfolioSnapshotRepository:
    """Repository for portfolio snapshots."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, snapshot: PortfolioSnapshot) -> PortfolioSnapshot:
        self._session.add(snapshot)
        self._session.flush()
        return snapshot

    def get_latest(self) -> PortfolioSnapshot | None:
        stmt = select(PortfolioSnapshot).order_by(PortfolioSnapshot.recorded_at.desc()).limit(1)
        return self._session.scalars(stmt).first()
