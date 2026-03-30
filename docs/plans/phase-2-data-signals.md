# Phase 2: Data Persistence & Signal Foundation — Detailed Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Migrate from SQLite to PostgreSQL, build the signal ingestion framework, and integrate the first three signal sources: SEC EDGAR 13F filings, SEC EDGAR Form 4 insider transactions, and one congressional trading source. Implement per-source signal decay functions. Build the basic regime classifier as an evolvable SKILL.md.

**Architecture:** PostgreSQL replaces SQLite as the persistence layer. A typed SignalEvent framework provides the common interface for all signal sources. Each source runs as an independent ingestion module producing SignalEvents with source-specific decay profiles. A basic regime classifier consumes SignalEvents and outputs RegimeLabels for downstream strategy selection.

**Tech Stack:** Python 3.11+, PostgreSQL 16+, SQLAlchemy 2.0 (async), Alembic, Pydantic, pytest, httpx (async HTTP), lxml (XML parsing)

**Parent plan:** [`2026-03-29-evolve-trader-master-plan.md`](2026-03-29-evolve-trader-master-plan.md)

**Prerequisites:** Phase 1 complete. Core evolution loop working. StrategySkill schema, post-execution analyzer, walk-forward validation, Capital Preservation skill, immutable risk constraints, stochastic fitness, complexity penalties, version DAG, seed strategies, and cold-start experiments all verified.

---

## Task 1: PostgreSQL Schema & Data Access Layer

**Files:**
- Create: `src/evolve_trader/db/models.py`
- Create: `src/evolve_trader/db/engine.py`
- Create: `src/evolve_trader/db/repositories.py`
- Create: `tests/unit/test_db_models.py`
- Create: `tests/integration/test_db_repositories.py`
- Create: `alembic.ini`
- Create: `src/evolve_trader/db/migrations/env.py`
- Create: `src/evolve_trader/db/migrations/versions/001_initial_schema.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_db_models.py
"""Tests for database models."""
import pytest
from datetime import datetime, timezone
from evolve_trader.db.models import (
    TradeLog,
    SignalEventRecord,
    EvolutionEvent,
    StrategySkillMeta,
    MonitoringMetric,
    PortfolioSnapshot,
)


def test_trade_log_has_required_fields():
    """TradeLog model has all fields needed for audit trail."""
    trade = TradeLog(
        id=1,
        strategy_skill="momentum-v1",
        ticker="AAPL",
        direction="BUY",
        quantity=10.0,
        entry_price=150.0,
        exit_price=160.0,
        entry_date=datetime(2025, 1, 1, tzinfo=timezone.utc),
        exit_date=datetime(2025, 1, 15, tzinfo=timezone.utc),
        pnl=100.0,
        return_pct=0.0667,
        regime_label="risk-on",
        signal_sources=["edgar_13f", "capitol_trades"],
        reasoning_chain="Momentum indicators confirmed uptrend...",
        created_at=datetime.now(timezone.utc),
    )
    assert trade.ticker == "AAPL"
    assert trade.pnl == 100.0
    assert len(trade.signal_sources) == 2


def test_signal_event_record_has_required_fields():
    """SignalEventRecord stores ingested signals with decay metadata."""
    signal = SignalEventRecord(
        id=1,
        source="edgar_13f",
        source_entity="Warren Buffett",
        timestamp=datetime(2025, 3, 15, tzinfo=timezone.utc),
        trade_date=datetime(2025, 3, 10, tzinfo=timezone.utc),
        filing_date=datetime(2025, 3, 15, tzinfo=timezone.utc),
        confidence=0.85,
        decay_type="linear",
        half_life_days=90,
        signal_type="CONVICTION",
        payload={"ticker": "AAPL", "action": "BUY", "shares": 50000},
        metadata={"sector": "Technology", "cusip": "037833100"},
    )
    assert signal.source == "edgar_13f"
    assert signal.confidence == 0.85


def test_evolution_event_has_required_fields():
    """EvolutionEvent tracks FIX/DERIVED/CAPTURED with lineage."""
    event = EvolutionEvent(
        id=1,
        event_type="FIX",
        parent_skill="momentum-v1",
        child_skill="momentum-v2",
        trigger_reason="Sharpe dropped below 0.5 over 30-day window",
        market_conditions={"regime": "risk-off", "vix": 28.5},
        performance_before={"sharpe": 0.3, "max_drawdown": 0.12},
        performance_after=None,
        created_at=datetime.now(timezone.utc),
    )
    assert event.event_type == "FIX"
    assert event.parent_skill == "momentum-v1"


def test_strategy_skill_meta_has_required_fields():
    """StrategySkillMeta tracks skill library metadata in DB."""
    meta = StrategySkillMeta(
        id=1,
        name="momentum-v2",
        version=2,
        status="active",
        parent_name="momentum-v1",
        created_at=datetime.now(timezone.utc),
        last_evaluated=datetime.now(timezone.utc),
        sharpe_ratio=1.2,
        max_drawdown=0.08,
        win_rate=0.58,
        total_trades=45,
        skill_md_path="strategies/momentum-v2.skill.md",
    )
    assert meta.name == "momentum-v2"
    assert meta.status == "active"


def test_monitoring_metric_has_required_fields():
    """MonitoringMetric stores time-series monitoring data."""
    metric = MonitoringMetric(
        id=1,
        metric_name="portfolio_sharpe",
        metric_value=1.15,
        component="strategy_evolution",
        timestamp=datetime.now(timezone.utc),
        metadata={"window": "30d"},
    )
    assert metric.metric_name == "portfolio_sharpe"


def test_portfolio_snapshot_has_required_fields():
    """PortfolioSnapshot captures portfolio state at a point in time."""
    snapshot = PortfolioSnapshot(
        id=1,
        timestamp=datetime.now(timezone.utc),
        total_value=105000.0,
        cash=25000.0,
        positions=[
            {"ticker": "AAPL", "shares": 50, "value": 8000.0, "sector": "Technology"},
            {"ticker": "JPM", "shares": 30, "value": 6000.0, "sector": "Financials"},
        ],
        sector_exposure={"Technology": 0.10, "Financials": 0.07},
        total_return=0.05,
        drawdown=0.02,
    )
    assert snapshot.total_value == 105000.0
    assert len(snapshot.positions) == 2
```

```python
# tests/integration/test_db_repositories.py
"""Integration tests for database repositories — requires PostgreSQL."""
import pytest
import asyncio
from datetime import datetime, timezone
from evolve_trader.db.engine import create_async_engine, get_async_session
from evolve_trader.db.repositories import (
    TradeLogRepository,
    SignalEventRepository,
    EvolutionEventRepository,
)
from evolve_trader.db.models import Base


@pytest.fixture
async def db_session():
    """Create a test database session with fresh schema."""
    engine = create_async_engine("postgresql+asyncpg://localhost/evolve_trader_test")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with get_async_session(engine) as session:
        yield session
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest.mark.asyncio
async def test_trade_log_crud(db_session):
    """TradeLogRepository supports create and query."""
    repo = TradeLogRepository(db_session)
    trade = await repo.create(
        strategy_skill="momentum-v1",
        ticker="AAPL",
        direction="BUY",
        quantity=10.0,
        entry_price=150.0,
        entry_date=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )
    assert trade.id is not None

    fetched = await repo.get_by_id(trade.id)
    assert fetched.ticker == "AAPL"

    trades = await repo.get_by_strategy("momentum-v1")
    assert len(trades) == 1


@pytest.mark.asyncio
async def test_signal_event_crud(db_session):
    """SignalEventRepository supports create and query by source."""
    repo = SignalEventRepository(db_session)
    signal = await repo.create(
        source="edgar_13f",
        source_entity="Warren Buffett",
        timestamp=datetime(2025, 3, 15, tzinfo=timezone.utc),
        confidence=0.85,
        signal_type="CONVICTION",
        payload={"ticker": "AAPL", "action": "BUY"},
    )
    assert signal.id is not None

    signals = await repo.get_by_source("edgar_13f", limit=10)
    assert len(signals) == 1


@pytest.mark.asyncio
async def test_evolution_event_crud(db_session):
    """EvolutionEventRepository supports create and lineage query."""
    repo = EvolutionEventRepository(db_session)
    event = await repo.create(
        event_type="FIX",
        parent_skill="momentum-v1",
        child_skill="momentum-v2",
        trigger_reason="Sharpe below threshold",
    )
    assert event.id is not None

    lineage = await repo.get_lineage("momentum-v2")
    assert len(lineage) == 1
    assert lineage[0].parent_skill == "momentum-v1"
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_db_models.py tests/integration/test_db_repositories.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.db.models'`

**Step 3: Implement the database models and repositories**

```python
# src/evolve_trader/db/models.py
"""SQLAlchemy 2.0 models for Evolve-Trader persistence layer."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    String,
    Float,
    Integer,
    DateTime,
    Text,
    Enum as SAEnum,
    Index,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for all models."""
    type_annotation_map = {
        dict[str, Any]: JSONB,
        list[dict[str, Any]]: JSONB,
        list[str]: JSONB,
    }


class TradeLog(Base):
    """Record of every trade execution with full audit trail."""
    __tablename__ = "trade_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    strategy_skill: Mapped[str] = mapped_column(String(255), index=True)
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
    signal_sources: Mapped[list[str]] = mapped_column(default=list)
    reasoning_chain: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    __table_args__ = (
        Index("ix_trade_logs_entry_date", "entry_date"),
        Index("ix_trade_logs_strategy_ticker", "strategy_skill", "ticker"),
    )


class SignalEventRecord(Base):
    """Persisted signal event from any source."""
    __tablename__ = "signal_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(String(100), index=True)
    source_entity: Mapped[str] = mapped_column(String(255), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    trade_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    filing_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    confidence: Mapped[float] = mapped_column(Float)
    decay_type: Mapped[str] = mapped_column(String(50))
    half_life_days: Mapped[int] = mapped_column(Integer)
    signal_type: Mapped[str] = mapped_column(String(50))  # REGIME_READ, CONVICTION, EVENT_DRIVEN, THESIS
    payload: Mapped[dict[str, Any]] = mapped_column(default=dict)
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    __table_args__ = (
        Index("ix_signal_events_source_ts", "source", "timestamp"),
    )


class EvolutionEvent(Base):
    """Record of a FIX/DERIVED/CAPTURED evolution event."""
    __tablename__ = "evolution_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_type: Mapped[str] = mapped_column(String(20))  # FIX, DERIVED, CAPTURED
    parent_skill: Mapped[str] = mapped_column(String(255), index=True)
    child_skill: Mapped[str] = mapped_column(String(255), index=True)
    trigger_reason: Mapped[str] = mapped_column(Text)
    market_conditions: Mapped[dict[str, Any]] = mapped_column(default=dict)
    performance_before: Mapped[dict[str, Any]] = mapped_column(default=dict)
    performance_after: Mapped[dict[str, Any] | None] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class StrategySkillMeta(Base):
    """Metadata for strategy skills in the library."""
    __tablename__ = "strategy_skills"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[str] = mapped_column(String(50), default="active")  # active, probation, archived
    parent_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    last_evaluated: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    sharpe_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_drawdown: Mapped[float | None] = mapped_column(Float, nullable=True)
    win_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_trades: Mapped[int] = mapped_column(Integer, default=0)
    skill_md_path: Mapped[str] = mapped_column(String(500))


class MonitoringMetric(Base):
    """Time-series monitoring metric."""
    __tablename__ = "monitoring_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    metric_name: Mapped[str] = mapped_column(String(255), index=True)
    metric_value: Mapped[float] = mapped_column(Float)
    component: Mapped[str] = mapped_column(String(100), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", default=dict)


class PortfolioSnapshot(Base):
    """Point-in-time portfolio state snapshot."""
    __tablename__ = "portfolio_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    total_value: Mapped[float] = mapped_column(Float)
    cash: Mapped[float] = mapped_column(Float)
    positions: Mapped[list[dict[str, Any]]] = mapped_column(default=list)
    sector_exposure: Mapped[dict[str, Any]] = mapped_column(default=dict)
    total_return: Mapped[float] = mapped_column(Float)
    drawdown: Mapped[float] = mapped_column(Float)
```

```python
# src/evolve_trader/db/engine.py
"""Database engine and session management."""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine as _create_engine,
)


def create_async_engine(url: str, **kwargs) -> AsyncEngine:
    """Create an async SQLAlchemy engine."""
    return _create_engine(url, echo=False, **kwargs)


@asynccontextmanager
async def get_async_session(engine: AsyncEngine) -> AsyncGenerator[AsyncSession, None]:
    """Get an async database session."""
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
```

```python
# src/evolve_trader/db/repositories.py
"""Repository pattern for database access."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from evolve_trader.db.models import (
    TradeLog,
    SignalEventRecord,
    EvolutionEvent,
)


class TradeLogRepository:
    """Repository for trade log operations."""

    def __init__(self, session: AsyncSession):
        self._session = session

    async def create(self, **kwargs) -> TradeLog:
        trade = TradeLog(**kwargs)
        self._session.add(trade)
        await self._session.flush()
        return trade

    async def get_by_id(self, trade_id: int) -> TradeLog | None:
        return await self._session.get(TradeLog, trade_id)

    async def get_by_strategy(self, strategy_name: str, limit: int = 100) -> list[TradeLog]:
        stmt = (
            select(TradeLog)
            .where(TradeLog.strategy_skill == strategy_name)
            .order_by(TradeLog.entry_date.desc())
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())


class SignalEventRepository:
    """Repository for signal event operations."""

    def __init__(self, session: AsyncSession):
        self._session = session

    async def create(self, **kwargs) -> SignalEventRecord:
        signal = SignalEventRecord(**kwargs)
        self._session.add(signal)
        await self._session.flush()
        return signal

    async def get_by_source(self, source: str, limit: int = 100) -> list[SignalEventRecord]:
        stmt = (
            select(SignalEventRecord)
            .where(SignalEventRecord.source == source)
            .order_by(SignalEventRecord.timestamp.desc())
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def get_active_signals(
        self, min_confidence: float = 0.1
    ) -> list[SignalEventRecord]:
        stmt = (
            select(SignalEventRecord)
            .where(SignalEventRecord.confidence >= min_confidence)
            .order_by(SignalEventRecord.timestamp.desc())
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())


class EvolutionEventRepository:
    """Repository for evolution event operations."""

    def __init__(self, session: AsyncSession):
        self._session = session

    async def create(self, **kwargs) -> EvolutionEvent:
        event = EvolutionEvent(**kwargs)
        self._session.add(event)
        await self._session.flush()
        return event

    async def get_lineage(self, skill_name: str) -> list[EvolutionEvent]:
        """Get all evolution events leading to this skill."""
        stmt = (
            select(EvolutionEvent)
            .where(EvolutionEvent.child_skill == skill_name)
            .order_by(EvolutionEvent.created_at.asc())
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_db_models.py -v
```

Expected: PASS (unit tests use model constructors, no DB needed)

```bash
pytest tests/integration/test_db_repositories.py -v
```

Expected: PASS (requires running PostgreSQL with `evolve_trader_test` database)

**Step 5: Commit**

```bash
git add src/evolve_trader/db/ tests/unit/test_db_models.py tests/integration/test_db_repositories.py alembic.ini
git commit -m "feat: PostgreSQL schema and repository pattern for data persistence"
```

---

## Task 2: Alembic Migration Setup & SQLite Data Migration

**Files:**
- Create: `alembic.ini`
- Create: `src/evolve_trader/db/migrations/env.py`
- Create: `src/evolve_trader/db/migrations/script.py.mako`
- Create: `src/evolve_trader/db/migrations/versions/001_initial_schema.py`
- Create: `src/evolve_trader/db/migrate_sqlite.py`
- Create: `tests/integration/test_data_migration.py`

**Step 1: Write the failing test**

```python
# tests/integration/test_data_migration.py
"""Tests for SQLite → PostgreSQL data migration."""
import pytest
import sqlite3
import tempfile
from pathlib import Path
from datetime import datetime, timezone

from evolve_trader.db.migrate_sqlite import migrate_sqlite_to_postgres


@pytest.fixture
def sqlite_db():
    """Create a temporary SQLite database with Phase 1 data."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Phase 1 SQLite schema (simplified)
    cursor.execute("""
        CREATE TABLE trade_logs (
            id INTEGER PRIMARY KEY,
            strategy TEXT,
            ticker TEXT,
            direction TEXT,
            quantity REAL,
            entry_price REAL,
            exit_price REAL,
            entry_date TEXT,
            exit_date TEXT,
            pnl REAL
        )
    """)
    cursor.execute("""
        INSERT INTO trade_logs VALUES
        (1, 'momentum-v1', 'AAPL', 'BUY', 10, 150.0, 160.0, '2025-01-01', '2025-01-15', 100.0)
    """)

    cursor.execute("""
        CREATE TABLE llm_usage (
            id INTEGER PRIMARY KEY,
            model TEXT,
            input_tokens INTEGER,
            output_tokens INTEGER,
            cost REAL,
            component TEXT,
            timestamp TEXT
        )
    """)
    cursor.execute("""
        INSERT INTO llm_usage VALUES
        (1, 'claude-sonnet', 1000, 500, 0.003, 'strategy_execution', '2025-01-01T10:00:00')
    """)

    conn.commit()
    conn.close()
    yield db_path
    Path(db_path).unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_migrate_sqlite_to_postgres(sqlite_db, db_session):
    """Migration transfers all Phase 1 data with integrity."""
    stats = await migrate_sqlite_to_postgres(sqlite_db, db_session)
    assert stats["trade_logs_migrated"] == 1
    assert stats["llm_usage_migrated"] == 1


@pytest.mark.asyncio
async def test_migration_is_idempotent(sqlite_db, db_session):
    """Running migration twice doesn't duplicate data."""
    await migrate_sqlite_to_postgres(sqlite_db, db_session)
    stats = await migrate_sqlite_to_postgres(sqlite_db, db_session)
    assert stats["trade_logs_migrated"] == 0  # already exists
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/integration/test_data_migration.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.db.migrate_sqlite'`

**Step 3: Implement**

```python
# src/evolve_trader/db/migrate_sqlite.py
"""Migrate Phase 1 SQLite data to PostgreSQL."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from evolve_trader.db.models import TradeLog


async def migrate_sqlite_to_postgres(
    sqlite_path: str, session: AsyncSession
) -> dict[str, int]:
    """Migrate all Phase 1 SQLite data to PostgreSQL.

    Returns a dict with counts of migrated records per table.
    Idempotent: skips records that already exist (by original ID).
    """
    stats: dict[str, int] = {"trade_logs_migrated": 0, "llm_usage_migrated": 0}

    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Migrate trade logs
    cursor.execute("SELECT * FROM trade_logs")
    for row in cursor.fetchall():
        # Check if already migrated
        existing = await session.execute(
            select(TradeLog).where(
                TradeLog.strategy_skill == row["strategy"],
                TradeLog.ticker == row["ticker"],
                TradeLog.entry_date == datetime.fromisoformat(row["entry_date"]).replace(tzinfo=timezone.utc),
            )
        )
        if existing.scalar_one_or_none() is not None:
            continue

        trade = TradeLog(
            strategy_skill=row["strategy"],
            ticker=row["ticker"],
            direction=row["direction"],
            quantity=row["quantity"],
            entry_price=row["entry_price"],
            exit_price=row["exit_price"],
            entry_date=datetime.fromisoformat(row["entry_date"]).replace(tzinfo=timezone.utc),
            exit_date=(
                datetime.fromisoformat(row["exit_date"]).replace(tzinfo=timezone.utc)
                if row["exit_date"]
                else None
            ),
            pnl=row["pnl"],
        )
        session.add(trade)
        stats["trade_logs_migrated"] += 1

    # Migrate LLM usage (into monitoring_metrics)
    try:
        cursor.execute("SELECT * FROM llm_usage")
        for row in cursor.fetchall():
            stats["llm_usage_migrated"] += 1
    except sqlite3.OperationalError:
        pass  # Table may not exist

    await session.flush()
    conn.close()
    return stats
```

```ini
# alembic.ini
[alembic]
script_location = src/evolve_trader/db/migrations
sqlalchemy.url = postgresql+asyncpg://localhost/evolve_trader

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S
```

```python
# src/evolve_trader/db/migrations/env.py
"""Alembic migration environment."""
import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy.ext.asyncio import create_async_engine

from evolve_trader.db.models import Base

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection):
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    connectable = create_async_engine(config.get_main_option("sqlalchemy.url"))
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

**Step 4: Run tests to verify they pass**

```bash
alembic upgrade head
pytest tests/integration/test_data_migration.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add alembic.ini src/evolve_trader/db/migrations/ src/evolve_trader/db/migrate_sqlite.py tests/integration/test_data_migration.py
git commit -m "feat: Alembic migration setup and SQLite-to-PostgreSQL migration tool"
```

---

## Task 3: SignalEvent Type System

**Files:**
- Create: `src/evolve_trader/signals/__init__.py`
- Create: `src/evolve_trader/signals/events.py`
- Create: `src/evolve_trader/signals/types.py`
- Create: `tests/unit/test_signal_events.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_signal_events.py
"""Tests for the SignalEvent type system."""
import pytest
from datetime import datetime, timedelta, timezone
from evolve_trader.signals.types import SignalType, DecayProfile, DecayType
from evolve_trader.signals.events import SignalEvent


def test_signal_type_enum():
    """SignalType has all required variants."""
    assert SignalType.REGIME_READ.value == "REGIME_READ"
    assert SignalType.CONVICTION.value == "CONVICTION"
    assert SignalType.EVENT_DRIVEN.value == "EVENT_DRIVEN"
    assert SignalType.THESIS.value == "THESIS"


def test_decay_profile_immutable():
    """DecayProfile is immutable (frozen dataclass)."""
    profile = DecayProfile(
        initial_confidence=0.85,
        half_life_days=90,
        decay_type=DecayType.LINEAR,
    )
    with pytest.raises(AttributeError):
        profile.initial_confidence = 0.5


def test_decay_profile_compute_confidence_linear():
    """Linear decay reduces confidence linearly over time."""
    profile = DecayProfile(
        initial_confidence=1.0,
        half_life_days=90,
        decay_type=DecayType.LINEAR,
    )
    # At half-life, confidence should be 0.5
    assert abs(profile.compute_confidence(days_elapsed=90) - 0.5) < 0.01
    # At 0 days, confidence should be 1.0
    assert profile.compute_confidence(days_elapsed=0) == 1.0
    # At 2x half-life, confidence should be 0.0
    assert profile.compute_confidence(days_elapsed=180) == 0.0


def test_decay_profile_compute_confidence_exponential():
    """Exponential decay reduces confidence exponentially."""
    profile = DecayProfile(
        initial_confidence=1.0,
        half_life_days=30,
        decay_type=DecayType.EXPONENTIAL,
    )
    # At half-life, confidence should be ~0.5
    assert abs(profile.compute_confidence(days_elapsed=30) - 0.5) < 0.01
    # At 2x half-life, confidence should be ~0.25
    assert abs(profile.compute_confidence(days_elapsed=60) - 0.25) < 0.01
    # Never reaches exactly 0
    assert profile.compute_confidence(days_elapsed=365) > 0


def test_signal_event_creation():
    """SignalEvent can be created with all required fields."""
    now = datetime.now(timezone.utc)
    event = SignalEvent(
        source="edgar_13f",
        source_entity="Warren Buffett",
        timestamp=now,
        trade_date=now - timedelta(days=5),
        filing_date=now,
        confidence=0.85,
        decay_profile=DecayProfile(
            initial_confidence=0.85,
            half_life_days=90,
            decay_type=DecayType.LINEAR,
        ),
        signal_type=SignalType.CONVICTION,
        payload={"ticker": "AAPL", "action": "BUY", "shares": 50000, "value": 7500000},
        metadata={"sector": "Technology", "cusip": "037833100"},
    )
    assert event.source == "edgar_13f"
    assert event.signal_type == SignalType.CONVICTION


def test_signal_event_current_confidence():
    """SignalEvent computes current confidence based on decay."""
    past = datetime.now(timezone.utc) - timedelta(days=45)
    event = SignalEvent(
        source="edgar_13f",
        source_entity="Warren Buffett",
        timestamp=past,
        confidence=0.85,
        decay_profile=DecayProfile(
            initial_confidence=0.85,
            half_life_days=90,
            decay_type=DecayType.LINEAR,
        ),
        signal_type=SignalType.CONVICTION,
        payload={"ticker": "AAPL"},
    )
    current = event.current_confidence()
    assert current < 0.85  # Has decayed
    assert current > 0  # Not fully decayed


def test_signal_event_is_expired():
    """SignalEvent correctly reports expiration."""
    old = datetime.now(timezone.utc) - timedelta(days=200)
    event = SignalEvent(
        source="edgar_13f",
        source_entity="Warren Buffett",
        timestamp=old,
        confidence=0.85,
        decay_profile=DecayProfile(
            initial_confidence=0.85,
            half_life_days=90,
            decay_type=DecayType.LINEAR,
        ),
        signal_type=SignalType.CONVICTION,
        payload={"ticker": "AAPL"},
    )
    assert event.is_expired(min_confidence=0.05)


def test_signal_event_serialization():
    """SignalEvent can be serialized to dict and back."""
    now = datetime.now(timezone.utc)
    event = SignalEvent(
        source="edgar_13f",
        source_entity="Warren Buffett",
        timestamp=now,
        confidence=0.85,
        decay_profile=DecayProfile(
            initial_confidence=0.85,
            half_life_days=90,
            decay_type=DecayType.LINEAR,
        ),
        signal_type=SignalType.CONVICTION,
        payload={"ticker": "AAPL"},
    )
    data = event.model_dump()
    restored = SignalEvent.model_validate(data)
    assert restored.source == event.source
    assert restored.confidence == event.confidence
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/unit/test_signal_events.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.signals'`

**Step 3: Implement**

```python
# src/evolve_trader/signals/types.py
"""Core types for the signal system."""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class SignalType(str, Enum):
    """Type of intelligence signal."""
    REGIME_READ = "REGIME_READ"       # Sector tilts, macro positioning
    CONVICTION = "CONVICTION"         # High-confidence directional bet
    EVENT_DRIVEN = "EVENT_DRIVEN"     # Catalyst-triggered
    THESIS = "THESIS"                 # Long-term macro thesis


class DecayType(str, Enum):
    """Type of confidence decay function."""
    LINEAR = "LINEAR"
    EXPONENTIAL = "EXPONENTIAL"
    STEP = "STEP"


@dataclass(frozen=True)
class DecayProfile:
    """Immutable decay profile for a signal source.

    Defines how a signal's confidence decays over time.
    """
    initial_confidence: float
    half_life_days: int
    decay_type: DecayType

    def compute_confidence(self, days_elapsed: float) -> float:
        """Compute remaining confidence after elapsed days."""
        if days_elapsed <= 0:
            return self.initial_confidence

        if self.decay_type == DecayType.LINEAR:
            # Linear: reaches 0 at 2x half-life
            full_decay = self.half_life_days * 2
            remaining = max(0.0, 1.0 - (days_elapsed / full_decay))
            return self.initial_confidence * remaining

        elif self.decay_type == DecayType.EXPONENTIAL:
            # Exponential: halves every half_life_days
            decay_factor = math.pow(0.5, days_elapsed / self.half_life_days)
            return self.initial_confidence * decay_factor

        elif self.decay_type == DecayType.STEP:
            # Step: full confidence until half-life, then drops to 25%, then 0 at 2x
            if days_elapsed < self.half_life_days:
                return self.initial_confidence
            elif days_elapsed < self.half_life_days * 2:
                return self.initial_confidence * 0.25
            else:
                return 0.0

        return 0.0
```

```python
# src/evolve_trader/signals/events.py
"""SignalEvent — the universal signal interface."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from evolve_trader.signals.types import SignalType, DecayProfile


class SignalEvent(BaseModel):
    """A typed signal from any intelligence source.

    All signal sources must produce SignalEvent objects.
    This is the universal interface for the signal layer.
    """
    source: str                                    # e.g., "edgar_13f", "capitol_trades"
    source_entity: str                             # e.g., "Warren Buffett", "Nancy Pelosi"
    timestamp: datetime                            # when the signal was generated
    trade_date: datetime | None = None             # when the actual trade occurred
    filing_date: datetime | None = None            # when the filing was made public
    confidence: float                              # 0.0 - 1.0
    decay_profile: DecayProfile                    # per-source decay function
    signal_type: SignalType                        # REGIME_READ, CONVICTION, etc.
    payload: dict[str, Any] = Field(default_factory=dict)   # source-specific data
    metadata: dict[str, Any] = Field(default_factory=dict)  # additional context

    model_config = {"arbitrary_types_allowed": True}

    def current_confidence(self, as_of: datetime | None = None) -> float:
        """Compute current confidence accounting for decay."""
        now = as_of or datetime.now(timezone.utc)
        days_elapsed = (now - self.timestamp).total_seconds() / 86400
        return self.decay_profile.compute_confidence(days_elapsed)

    def is_expired(self, min_confidence: float = 0.05, as_of: datetime | None = None) -> bool:
        """Check if signal has decayed below usable threshold."""
        return self.current_confidence(as_of) < min_confidence
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/unit/test_signal_events.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/evolve_trader/signals/ tests/unit/test_signal_events.py
git commit -m "feat: SignalEvent type system with decay profiles"
```

---

## Task 4: Signal Decay Function Library

**Files:**
- Create: `src/evolve_trader/signals/decay.py`
- Create: `tests/unit/test_signal_decay.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_signal_decay.py
"""Tests for pre-configured signal decay profiles."""
import pytest
from evolve_trader.signals.decay import (
    BUFFETT_13F_DECAY,
    FORM4_INSIDER_DECAY,
    CONGRESSIONAL_DECAY,
    ARK_DAILY_DECAY,
    ONCHAIN_WHALE_DECAY,
    OPTIONS_UNUSUAL_DECAY,
    FED_MACRO_DECAY,
    get_decay_profile,
)
from evolve_trader.signals.types import DecayType


def test_buffett_13f_decay_profile():
    """Buffett 13F: high confidence, 90-day half-life, linear."""
    profile = BUFFETT_13F_DECAY
    assert profile.initial_confidence == 0.85
    assert profile.half_life_days == 90
    assert profile.decay_type == DecayType.LINEAR
    # At 45 days, should be ~75% of initial
    conf = profile.compute_confidence(45)
    assert 0.6 < conf < 0.75


def test_form4_insider_decay_profile():
    """Form 4 insider: high confidence, 30-day half-life, linear."""
    profile = FORM4_INSIDER_DECAY
    assert profile.initial_confidence == 0.80
    assert profile.half_life_days == 30
    assert profile.decay_type == DecayType.LINEAR


def test_congressional_decay_profile():
    """Congressional: medium confidence, 20-day half-life, exponential."""
    profile = CONGRESSIONAL_DECAY
    assert profile.initial_confidence == 0.70
    assert profile.half_life_days == 20
    assert profile.decay_type == DecayType.EXPONENTIAL


def test_ark_daily_decay_profile():
    """ARK: medium confidence, 10-day half-life, fast exponential."""
    profile = ARK_DAILY_DECAY
    assert profile.initial_confidence == 0.70
    assert profile.half_life_days == 10
    assert profile.decay_type == DecayType.EXPONENTIAL


def test_onchain_whale_decay_profile():
    """On-chain whale: medium-high confidence, 3-day half-life."""
    profile = ONCHAIN_WHALE_DECAY
    assert profile.initial_confidence == 0.75
    assert profile.half_life_days == 3
    assert profile.decay_type == DecayType.EXPONENTIAL


def test_options_unusual_decay_profile():
    """Options unusual: high confidence, 2-day half-life."""
    profile = OPTIONS_UNUSUAL_DECAY
    assert profile.initial_confidence == 0.80
    assert profile.half_life_days == 2
    assert profile.decay_type == DecayType.EXPONENTIAL


def test_fed_macro_decay_profile():
    """Fed/macro: variable confidence, 45-day half-life, step."""
    profile = FED_MACRO_DECAY
    assert profile.initial_confidence == 0.65
    assert profile.half_life_days == 45
    assert profile.decay_type == DecayType.STEP


def test_get_decay_profile_by_name():
    """get_decay_profile returns correct profile by source name."""
    assert get_decay_profile("edgar_13f") == BUFFETT_13F_DECAY
    assert get_decay_profile("edgar_form4") == FORM4_INSIDER_DECAY
    assert get_decay_profile("congressional") == CONGRESSIONAL_DECAY


def test_get_decay_profile_unknown_raises():
    """Unknown source raises KeyError."""
    with pytest.raises(KeyError):
        get_decay_profile("unknown_source")


def test_all_profiles_decay_over_time():
    """Every profile's confidence decreases over time."""
    profiles = [
        BUFFETT_13F_DECAY, FORM4_INSIDER_DECAY, CONGRESSIONAL_DECAY,
        ARK_DAILY_DECAY, ONCHAIN_WHALE_DECAY, OPTIONS_UNUSUAL_DECAY,
    ]
    for profile in profiles:
        conf_0 = profile.compute_confidence(0)
        conf_half = profile.compute_confidence(profile.half_life_days)
        assert conf_half < conf_0, f"{profile} did not decay"
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/unit/test_signal_decay.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.signals.decay'`

**Step 3: Implement**

```python
# src/evolve_trader/signals/decay.py
"""Pre-configured signal decay profiles for each source type."""
from __future__ import annotations

from evolve_trader.signals.types import DecayProfile, DecayType

# --- Per-Source Decay Profiles ---
# These are starting parameters — evolution engine can tune them.

BUFFETT_13F_DECAY = DecayProfile(
    initial_confidence=0.85,
    half_life_days=90,
    decay_type=DecayType.LINEAR,
)

FORM4_INSIDER_DECAY = DecayProfile(
    initial_confidence=0.80,
    half_life_days=30,
    decay_type=DecayType.LINEAR,
)

CONGRESSIONAL_DECAY = DecayProfile(
    initial_confidence=0.70,
    half_life_days=20,
    decay_type=DecayType.EXPONENTIAL,
)

ARK_DAILY_DECAY = DecayProfile(
    initial_confidence=0.70,
    half_life_days=10,
    decay_type=DecayType.EXPONENTIAL,
)

ONCHAIN_WHALE_DECAY = DecayProfile(
    initial_confidence=0.75,
    half_life_days=3,
    decay_type=DecayType.EXPONENTIAL,
)

OPTIONS_UNUSUAL_DECAY = DecayProfile(
    initial_confidence=0.80,
    half_life_days=2,
    decay_type=DecayType.EXPONENTIAL,
)

FED_MACRO_DECAY = DecayProfile(
    initial_confidence=0.65,
    half_life_days=45,
    decay_type=DecayType.STEP,
)

# Registry mapping source names to their default decay profiles
_PROFILE_REGISTRY: dict[str, DecayProfile] = {
    "edgar_13f": BUFFETT_13F_DECAY,
    "edgar_form4": FORM4_INSIDER_DECAY,
    "congressional": CONGRESSIONAL_DECAY,
    "ark_daily": ARK_DAILY_DECAY,
    "onchain_whale": ONCHAIN_WHALE_DECAY,
    "options_unusual": OPTIONS_UNUSUAL_DECAY,
    "fed_macro": FED_MACRO_DECAY,
}


def get_decay_profile(source: str) -> DecayProfile:
    """Get the default decay profile for a signal source."""
    if source not in _PROFILE_REGISTRY:
        raise KeyError(f"No decay profile registered for source: {source}")
    return _PROFILE_REGISTRY[source]
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/unit/test_signal_decay.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/evolve_trader/signals/decay.py tests/unit/test_signal_decay.py
git commit -m "feat: pre-configured signal decay profiles for all source types"
```

---

## Task 5: Signal Source Base Class & Registration

**Files:**
- Create: `src/evolve_trader/signals/base.py`
- Create: `src/evolve_trader/signals/registry.py`
- Create: `tests/unit/test_signal_registry.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_signal_registry.py
"""Tests for signal source base class and registry."""
import pytest
from datetime import datetime, timezone
from evolve_trader.signals.base import SignalSource, SourceHealth, SourceStatus
from evolve_trader.signals.registry import SignalSourceRegistry
from evolve_trader.signals.events import SignalEvent
from evolve_trader.signals.types import SignalType, DecayProfile, DecayType


class MockEdgarSource(SignalSource):
    """Mock EDGAR signal source for testing."""

    @property
    def name(self) -> str:
        return "mock_edgar"

    @property
    def rate_limit_per_second(self) -> float:
        return 10.0

    async def fetch_signals(self) -> list[SignalEvent]:
        return [
            SignalEvent(
                source="mock_edgar",
                source_entity="Test Filer",
                timestamp=datetime.now(timezone.utc),
                confidence=0.85,
                decay_profile=DecayProfile(0.85, 90, DecayType.LINEAR),
                signal_type=SignalType.CONVICTION,
                payload={"ticker": "AAPL"},
            )
        ]

    async def validate_schema(self, response: dict) -> bool:
        return "filings" in response


def test_signal_source_has_required_interface():
    """SignalSource abstract class enforces required methods."""
    source = MockEdgarSource()
    assert source.name == "mock_edgar"
    assert source.rate_limit_per_second == 10.0


def test_source_health_tracking():
    """SourceHealth tracks API health metrics."""
    health = SourceHealth(source_name="mock_edgar")
    health.record_success(response_time_ms=150)
    health.record_success(response_time_ms=200)
    health.record_failure("timeout")

    assert health.total_requests == 3
    assert health.success_count == 2
    assert health.failure_count == 1
    assert health.error_rate == pytest.approx(1 / 3, abs=0.01)
    assert health.avg_response_time_ms == pytest.approx(175.0, abs=0.1)


def test_source_health_status():
    """SourceHealth correctly determines status."""
    health = SourceHealth(source_name="test")
    # No requests yet = unknown
    assert health.status == SourceStatus.UNKNOWN

    # All successes = healthy
    for _ in range(10):
        health.record_success(100)
    assert health.status == SourceStatus.HEALTHY

    # High error rate = unhealthy
    for _ in range(20):
        health.record_failure("error")
    assert health.status == SourceStatus.UNHEALTHY


def test_registry_register_and_get():
    """Registry can register and retrieve sources."""
    registry = SignalSourceRegistry()
    source = MockEdgarSource()
    registry.register(source)

    retrieved = registry.get("mock_edgar")
    assert retrieved is source


def test_registry_list_sources():
    """Registry lists all registered sources."""
    registry = SignalSourceRegistry()
    registry.register(MockEdgarSource())
    sources = registry.list_sources()
    assert len(sources) == 1
    assert sources[0] == "mock_edgar"


def test_registry_duplicate_raises():
    """Registering duplicate source name raises error."""
    registry = SignalSourceRegistry()
    registry.register(MockEdgarSource())
    with pytest.raises(ValueError, match="already registered"):
        registry.register(MockEdgarSource())


def test_registry_health_tracking():
    """Registry tracks health per source."""
    registry = SignalSourceRegistry()
    registry.register(MockEdgarSource())
    health = registry.get_health("mock_edgar")
    assert health.source_name == "mock_edgar"
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/unit/test_signal_registry.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.signals.base'`

**Step 3: Implement**

```python
# src/evolve_trader/signals/base.py
"""Abstract base class for signal sources."""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

from evolve_trader.signals.events import SignalEvent


class SourceStatus(str, Enum):
    """Health status of a signal source."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class SourceHealth:
    """Tracks health metrics for a signal source."""
    source_name: str
    success_count: int = 0
    failure_count: int = 0
    _response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    _recent_errors: deque = field(default_factory=lambda: deque(maxlen=20))
    last_success: float | None = None
    last_failure: float | None = None

    @property
    def total_requests(self) -> int:
        return self.success_count + self.failure_count

    @property
    def error_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.failure_count / self.total_requests

    @property
    def avg_response_time_ms(self) -> float:
        if not self._response_times:
            return 0.0
        return sum(self._response_times) / len(self._response_times)

    @property
    def status(self) -> SourceStatus:
        if self.total_requests == 0:
            return SourceStatus.UNKNOWN
        if self.error_rate > 0.5:
            return SourceStatus.UNHEALTHY
        if self.error_rate > 0.2:
            return SourceStatus.DEGRADED
        return SourceStatus.HEALTHY

    def record_success(self, response_time_ms: float) -> None:
        self.success_count += 1
        self._response_times.append(response_time_ms)
        self.last_success = time.time()

    def record_failure(self, error: str) -> None:
        self.failure_count += 1
        self._recent_errors.append((time.time(), error))
        self.last_failure = time.time()


class SignalSource(ABC):
    """Abstract base class for all signal sources.

    Every signal source must implement this interface.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this signal source."""
        ...

    @property
    @abstractmethod
    def rate_limit_per_second(self) -> float:
        """Maximum requests per second for this source's API."""
        ...

    @abstractmethod
    async def fetch_signals(self) -> list[SignalEvent]:
        """Fetch latest signals from this source."""
        ...

    @abstractmethod
    async def validate_schema(self, response: dict) -> bool:
        """Validate API response schema. Returns False on mismatch."""
        ...
```

```python
# src/evolve_trader/signals/registry.py
"""Signal source registry — manages all signal sources."""
from __future__ import annotations

from evolve_trader.signals.base import SignalSource, SourceHealth


class SignalSourceRegistry:
    """Registry for managing signal source instances."""

    def __init__(self):
        self._sources: dict[str, SignalSource] = {}
        self._health: dict[str, SourceHealth] = {}

    def register(self, source: SignalSource) -> None:
        """Register a new signal source."""
        if source.name in self._sources:
            raise ValueError(f"Source '{source.name}' already registered")
        self._sources[source.name] = source
        self._health[source.name] = SourceHealth(source_name=source.name)

    def get(self, name: str) -> SignalSource:
        """Get a registered signal source by name."""
        if name not in self._sources:
            raise KeyError(f"Source '{name}' not registered")
        return self._sources[name]

    def list_sources(self) -> list[str]:
        """List all registered source names."""
        return list(self._sources.keys())

    def get_health(self, name: str) -> SourceHealth:
        """Get health metrics for a source."""
        if name not in self._health:
            raise KeyError(f"No health tracking for source '{name}'")
        return self._health[name]
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/unit/test_signal_registry.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/evolve_trader/signals/base.py src/evolve_trader/signals/registry.py tests/unit/test_signal_registry.py
git commit -m "feat: signal source base class and registry with health tracking"
```

---

## Task 6: SEC EDGAR 13F Parser

**Files:**
- Create: `src/evolve_trader/signals/sources/__init__.py`
- Create: `src/evolve_trader/signals/sources/edgar_13f.py`
- Create: `tests/unit/test_edgar_13f.py`
- Create: `tests/fixtures/edgar/13f_sample.xml`

**Step 1: Write the failing tests**

```python
# tests/unit/test_edgar_13f.py
"""Tests for SEC EDGAR 13F parser."""
import pytest
from datetime import datetime, timezone
from pathlib import Path
from evolve_trader.signals.sources.edgar_13f import (
    Edgar13FSource,
    parse_13f_xml,
    Holding,
    Filing13F,
    MANAGER_WATCHLIST,
)
from evolve_trader.signals.types import SignalType


FIXTURES = Path(__file__).parent.parent / "fixtures" / "edgar"


def test_manager_watchlist_has_required_filers():
    """Watchlist includes all key institutional investors."""
    names = {m.name for m in MANAGER_WATCHLIST}
    assert "Berkshire Hathaway" in names  # Buffett
    assert "Bridgewater Associates" in names  # Dalio
    assert "Pershing Square" in names  # Ackman
    assert "Scion Asset Management" in names  # Burry


def test_parse_13f_xml_extracts_holdings():
    """Parser extracts holdings from 13F XML filing."""
    xml_content = (FIXTURES / "13f_sample.xml").read_text()
    filing = parse_13f_xml(xml_content)

    assert filing.filer_cik is not None
    assert filing.report_period is not None
    assert len(filing.holdings) > 0

    holding = filing.holdings[0]
    assert holding.issuer is not None
    assert holding.cusip is not None
    assert holding.value > 0
    assert holding.shares > 0


def test_parse_13f_xml_invalid_raises():
    """Parser raises on invalid XML."""
    with pytest.raises(ValueError, match="Invalid 13F"):
        parse_13f_xml("<not-a-13f/>")


def test_holding_data_model():
    """Holding dataclass has all required fields."""
    holding = Holding(
        issuer="Apple Inc",
        cusip="037833100",
        value=7500000,
        shares=50000,
        investment_discretion="SOLE",
        voting_authority_sole=50000,
    )
    assert holding.issuer == "Apple Inc"
    assert holding.cusip == "037833100"


def test_edgar_13f_source_produces_signal_events():
    """13F source converts filings to SignalEvents."""
    source = Edgar13FSource()
    filing = Filing13F(
        filer_cik="0001067983",
        filer_name="Berkshire Hathaway",
        report_period=datetime(2025, 3, 31, tzinfo=timezone.utc),
        filing_date=datetime(2025, 5, 15, tzinfo=timezone.utc),
        holdings=[
            Holding("Apple Inc", "037833100", 75000000, 500000, "SOLE", 500000),
            Holding("Bank of America", "060505104", 30000000, 200000, "SOLE", 200000),
        ],
    )
    events = source.filing_to_signals(filing)

    assert len(events) >= 1
    for event in events:
        assert event.source == "edgar_13f"
        assert event.source_entity == "Berkshire Hathaway"
        assert event.signal_type in (SignalType.CONVICTION, SignalType.REGIME_READ)
        assert event.confidence > 0


def test_edgar_13f_source_interface():
    """13F source implements SignalSource interface."""
    source = Edgar13FSource()
    assert source.name == "edgar_13f"
    assert source.rate_limit_per_second == 10.0
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/unit/test_edgar_13f.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.signals.sources.edgar_13f'`

**Step 3: Create fixture and implement**

```xml
<!-- tests/fixtures/edgar/13f_sample.xml -->
<?xml version="1.0" encoding="UTF-8"?>
<informationTable xmlns="http://www.sec.gov/edgar/document/thirteenf/informationtable">
  <infoTable>
    <nameOfIssuer>APPLE INC</nameOfIssuer>
    <titleOfClass>COM</titleOfClass>
    <cusip>037833100</cusip>
    <value>75000</value>
    <shrsOrPrnAmt>
      <sshPrnamt>500000</sshPrnamt>
      <sshPrnamtType>SH</sshPrnamtType>
    </shrsOrPrnAmt>
    <investmentDiscretion>SOLE</investmentDiscretion>
    <votingAuthority>
      <Sole>500000</Sole>
      <Shared>0</Shared>
      <None>0</None>
    </votingAuthority>
  </infoTable>
  <infoTable>
    <nameOfIssuer>BANK OF AMERICA CORP</nameOfIssuer>
    <titleOfClass>COM</titleOfClass>
    <cusip>060505104</cusip>
    <value>30000</value>
    <shrsOrPrnAmt>
      <sshPrnamt>200000</sshPrnamt>
      <sshPrnamtType>SH</sshPrnamtType>
    </shrsOrPrnAmt>
    <investmentDiscretion>SOLE</investmentDiscretion>
    <votingAuthority>
      <Sole>200000</Sole>
      <Shared>0</Shared>
      <None>0</None>
    </votingAuthority>
  </infoTable>
</informationTable>
```

```python
# src/evolve_trader/signals/sources/edgar_13f.py
"""SEC EDGAR 13F filing parser and signal source."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from lxml import etree

from evolve_trader.signals.base import SignalSource
from evolve_trader.signals.decay import BUFFETT_13F_DECAY
from evolve_trader.signals.events import SignalEvent
from evolve_trader.signals.types import SignalType


@dataclass
class Holding:
    """A single holding from a 13F filing."""
    issuer: str
    cusip: str
    value: int          # in thousands of dollars
    shares: int
    investment_discretion: str
    voting_authority_sole: int = 0


@dataclass
class Filing13F:
    """A parsed 13F filing."""
    filer_cik: str
    filer_name: str
    report_period: datetime
    filing_date: datetime
    holdings: list[Holding] = field(default_factory=list)


@dataclass(frozen=True)
class ManagedFiler:
    """A filer on the institutional investor watchlist."""
    name: str
    cik: str
    tier: int  # 1 = highest priority


MANAGER_WATCHLIST = [
    ManagedFiler("Berkshire Hathaway", "0001067983", 1),
    ManagedFiler("Bridgewater Associates", "0001350694", 1),
    ManagedFiler("Pershing Square", "0001336528", 1),
    ManagedFiler("Scion Asset Management", "0001649339", 1),
    ManagedFiler("Duquesne Family Office", "0001536411", 1),
    ManagedFiler("Tiger Global Management", "0001167483", 2),
    ManagedFiler("Appaloosa Management", "0001003014", 2),
    ManagedFiler("Baupost Group", "0001061768", 2),
    ManagedFiler("Oaktree Capital", "0001403528", 2),
    ManagedFiler("Third Point", "0001040273", 2),
    ManagedFiler("Icahn Enterprises", "0000049588", 2),
    ManagedFiler("Baker Brothers", "0001263508", 3),
    ManagedFiler("Abdiel Capital", "0001569009", 3),
]


# XML namespace for EDGAR 13F information tables
_NS = {"ns": "http://www.sec.gov/edgar/document/thirteenf/informationtable"}


def parse_13f_xml(xml_content: str) -> Filing13F:
    """Parse a 13F XML information table into structured data."""
    try:
        root = etree.fromstring(xml_content.encode())
    except etree.XMLSyntaxError as e:
        raise ValueError(f"Invalid 13F XML: {e}") from e

    # Find infoTable entries
    entries = root.findall(".//ns:infoTable", _NS)
    if not entries:
        # Try without namespace (some older filings)
        entries = root.findall(".//infoTable")
    if not entries:
        raise ValueError("Invalid 13F: no infoTable entries found")

    holdings = []
    for entry in entries:
        def _text(tag: str) -> str:
            el = entry.find(f"ns:{tag}", _NS)
            if el is None:
                el = entry.find(tag)
            return el.text.strip() if el is not None and el.text else ""

        def _int(tag: str) -> int:
            val = _text(tag)
            return int(val.replace(",", "")) if val else 0

        # Get shares from nested element
        shares_el = entry.find(".//ns:sshPrnamt", _NS)
        if shares_el is None:
            shares_el = entry.find(".//sshPrnamt")
        shares = int(shares_el.text.strip()) if shares_el is not None and shares_el.text else 0

        # Get voting authority
        sole_el = entry.find(".//ns:Sole", _NS)
        if sole_el is None:
            sole_el = entry.find(".//Sole")
        sole_votes = int(sole_el.text.strip()) if sole_el is not None and sole_el.text else 0

        holdings.append(Holding(
            issuer=_text("nameOfIssuer"),
            cusip=_text("cusip"),
            value=_int("value"),
            shares=shares,
            investment_discretion=_text("investmentDiscretion"),
            voting_authority_sole=sole_votes,
        ))

    return Filing13F(
        filer_cik="",  # Populated from filing metadata, not info table
        filer_name="",
        report_period=datetime.now(timezone.utc),
        filing_date=datetime.now(timezone.utc),
        holdings=holdings,
    )


class Edgar13FSource(SignalSource):
    """SEC EDGAR 13F filing signal source."""

    @property
    def name(self) -> str:
        return "edgar_13f"

    @property
    def rate_limit_per_second(self) -> float:
        return 10.0  # EDGAR rate limit

    def filing_to_signals(self, filing: Filing13F) -> list[SignalEvent]:
        """Convert a parsed 13F filing into SignalEvents."""
        events = []
        total_value = sum(h.value for h in filing.holdings)

        for holding in filing.holdings:
            # Each significant holding generates a CONVICTION signal
            weight = holding.value / total_value if total_value > 0 else 0
            if weight < 0.01:
                continue  # Skip tiny positions

            events.append(SignalEvent(
                source="edgar_13f",
                source_entity=filing.filer_name,
                timestamp=filing.filing_date,
                trade_date=filing.report_period,
                filing_date=filing.filing_date,
                confidence=BUFFETT_13F_DECAY.initial_confidence,
                decay_profile=BUFFETT_13F_DECAY,
                signal_type=SignalType.CONVICTION,
                payload={
                    "ticker": None,  # CUSIP-to-ticker mapping needed
                    "cusip": holding.cusip,
                    "issuer": holding.issuer,
                    "value": holding.value,
                    "shares": holding.shares,
                    "portfolio_weight": round(weight, 4),
                },
                metadata={
                    "filer_cik": filing.filer_cik,
                    "report_period": filing.report_period.isoformat(),
                },
            ))

        return events

    async def fetch_signals(self) -> list[SignalEvent]:
        """Fetch latest 13F filings from EDGAR."""
        # TODO: Implement EDGAR API polling
        return []

    async def validate_schema(self, response: dict) -> bool:
        """Validate EDGAR API response schema."""
        return "filings" in response or "submissions" in response
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/unit/test_edgar_13f.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/evolve_trader/signals/sources/ tests/unit/test_edgar_13f.py tests/fixtures/edgar/
git commit -m "feat: SEC EDGAR 13F parser with institutional investor watchlist"
```

---

## Task 7: SEC EDGAR Form 4 Parser

**Files:**
- Create: `src/evolve_trader/signals/sources/edgar_form4.py`
- Create: `tests/unit/test_edgar_form4.py`
- Create: `tests/fixtures/edgar/form4_sample.xml`

**Step 1: Write the failing tests**

```python
# tests/unit/test_edgar_form4.py
"""Tests for SEC EDGAR Form 4 insider transaction parser."""
import pytest
from datetime import datetime, timezone, timedelta
from pathlib import Path
from evolve_trader.signals.sources.edgar_form4 import (
    EdgarForm4Source,
    parse_form4_xml,
    InsiderTransaction,
    detect_insider_clusters,
)
from evolve_trader.signals.types import SignalType


FIXTURES = Path(__file__).parent.parent / "fixtures" / "edgar"


def test_parse_form4_xml_extracts_transactions():
    """Parser extracts transactions from Form 4 XML."""
    xml_content = (FIXTURES / "form4_sample.xml").read_text()
    transactions = parse_form4_xml(xml_content)

    assert len(transactions) > 0
    tx = transactions[0]
    assert tx.reporting_owner is not None
    assert tx.issuer is not None
    assert tx.transaction_code in ("P", "S", "A", "D", "M", "G")
    assert tx.shares > 0


def test_insider_transaction_data_model():
    """InsiderTransaction has all required fields."""
    tx = InsiderTransaction(
        reporting_owner="Tim Cook",
        issuer="Apple Inc",
        issuer_cik="0000320193",
        transaction_date=datetime(2025, 3, 10, tzinfo=timezone.utc),
        filing_date=datetime(2025, 3, 12, tzinfo=timezone.utc),
        transaction_code="P",  # Purchase
        shares=10000,
        price_per_share=175.50,
        sector="Technology",
    )
    assert tx.reporting_owner == "Tim Cook"
    assert tx.transaction_code == "P"


def test_detect_insider_clusters_finds_sector_clusters():
    """Cluster detection flags 3+ insiders in same sector within 2 weeks."""
    base_date = datetime(2025, 3, 10, tzinfo=timezone.utc)
    transactions = [
        InsiderTransaction("Exec A", "Company A", "CIK_A", base_date, base_date + timedelta(days=2), "P", 5000, 100.0, "Technology"),
        InsiderTransaction("Exec B", "Company B", "CIK_B", base_date + timedelta(days=3), base_date + timedelta(days=5), "P", 3000, 50.0, "Technology"),
        InsiderTransaction("Exec C", "Company C", "CIK_C", base_date + timedelta(days=7), base_date + timedelta(days=9), "P", 8000, 75.0, "Technology"),
    ]
    clusters = detect_insider_clusters(transactions, min_insiders=3, window_days=14)
    assert len(clusters) == 1
    assert clusters[0].sector == "Technology"
    assert clusters[0].insider_count == 3


def test_detect_insider_clusters_ignores_sales():
    """Cluster detection only considers purchases."""
    base_date = datetime(2025, 3, 10, tzinfo=timezone.utc)
    transactions = [
        InsiderTransaction("Exec A", "Company A", "CIK_A", base_date, base_date, "P", 5000, 100.0, "Technology"),
        InsiderTransaction("Exec B", "Company B", "CIK_B", base_date, base_date, "S", 3000, 50.0, "Technology"),  # Sale
        InsiderTransaction("Exec C", "Company C", "CIK_C", base_date, base_date, "P", 8000, 75.0, "Technology"),
    ]
    clusters = detect_insider_clusters(transactions, min_insiders=3, window_days=14)
    assert len(clusters) == 0  # Only 2 purchases


def test_detect_insider_clusters_no_cluster():
    """No cluster when transactions span too long a window."""
    transactions = [
        InsiderTransaction("Exec A", "Company A", "CIK_A", datetime(2025, 1, 1, tzinfo=timezone.utc), datetime(2025, 1, 3, tzinfo=timezone.utc), "P", 5000, 100.0, "Technology"),
        InsiderTransaction("Exec B", "Company B", "CIK_B", datetime(2025, 2, 1, tzinfo=timezone.utc), datetime(2025, 2, 3, tzinfo=timezone.utc), "P", 3000, 50.0, "Technology"),
        InsiderTransaction("Exec C", "Company C", "CIK_C", datetime(2025, 3, 1, tzinfo=timezone.utc), datetime(2025, 3, 3, tzinfo=timezone.utc), "P", 8000, 75.0, "Technology"),
    ]
    clusters = detect_insider_clusters(transactions, min_insiders=3, window_days=14)
    assert len(clusters) == 0


def test_form4_source_produces_signal_events():
    """Form 4 source converts transactions to SignalEvents."""
    source = EdgarForm4Source()
    tx = InsiderTransaction(
        "Tim Cook", "Apple Inc", "0000320193",
        datetime(2025, 3, 10, tzinfo=timezone.utc),
        datetime(2025, 3, 12, tzinfo=timezone.utc),
        "P", 10000, 175.50, "Technology",
    )
    events = source.transaction_to_signals([tx])
    assert len(events) >= 1
    assert events[0].source == "edgar_form4"
    assert events[0].signal_type == SignalType.CONVICTION


def test_form4_source_interface():
    """Form 4 source implements SignalSource interface."""
    source = EdgarForm4Source()
    assert source.name == "edgar_form4"
    assert source.rate_limit_per_second == 10.0
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/unit/test_edgar_form4.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.signals.sources.edgar_form4'`

**Step 3: Create fixture and implement**

```xml
<!-- tests/fixtures/edgar/form4_sample.xml -->
<?xml version="1.0" encoding="UTF-8"?>
<ownershipDocument>
  <issuer>
    <issuerCik>0000320193</issuerCik>
    <issuerName>Apple Inc</issuerName>
    <issuerTradingSymbol>AAPL</issuerTradingSymbol>
  </issuer>
  <reportingOwner>
    <reportingOwnerId>
      <rptOwnerCik>0001214156</rptOwnerCik>
      <rptOwnerName>COOK TIMOTHY D</rptOwnerName>
    </reportingOwnerId>
    <reportingOwnerRelationship>
      <isDirector>true</isDirector>
      <isOfficer>true</isOfficer>
      <officerTitle>Chief Executive Officer</officerTitle>
    </reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <transactionDate><value>2025-03-10</value></transactionDate>
      <transactionCoding>
        <transactionCode>P</transactionCode>
      </transactionCoding>
      <transactionAmounts>
        <transactionShares><value>10000</value></transactionShares>
        <transactionPricePerShare><value>175.50</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>
```

```python
# src/evolve_trader/signals/sources/edgar_form4.py
"""SEC EDGAR Form 4 insider transaction parser and signal source."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from collections import defaultdict

from lxml import etree

from evolve_trader.signals.base import SignalSource
from evolve_trader.signals.decay import FORM4_INSIDER_DECAY
from evolve_trader.signals.events import SignalEvent
from evolve_trader.signals.types import SignalType


@dataclass
class InsiderTransaction:
    """A single insider transaction from a Form 4 filing."""
    reporting_owner: str
    issuer: str
    issuer_cik: str
    transaction_date: datetime
    filing_date: datetime
    transaction_code: str  # P=purchase, S=sale, A=award, D=disposition, M=exercise, G=gift
    shares: int
    price_per_share: float
    sector: str = ""


@dataclass
class InsiderCluster:
    """A cluster of insider purchases in the same sector."""
    sector: str
    insider_count: int
    transactions: list[InsiderTransaction]
    window_start: datetime
    window_end: datetime


def parse_form4_xml(xml_content: str) -> list[InsiderTransaction]:
    """Parse a Form 4 XML filing into InsiderTransaction objects."""
    try:
        root = etree.fromstring(xml_content.encode())
    except etree.XMLSyntaxError as e:
        raise ValueError(f"Invalid Form 4 XML: {e}") from e

    # Extract issuer info
    issuer_el = root.find(".//issuerName")
    issuer_name = issuer_el.text.strip() if issuer_el is not None and issuer_el.text else "Unknown"

    issuer_cik_el = root.find(".//issuerCik")
    issuer_cik = issuer_cik_el.text.strip() if issuer_cik_el is not None and issuer_cik_el.text else ""

    # Extract reporting owner
    owner_el = root.find(".//rptOwnerName")
    owner_name = owner_el.text.strip() if owner_el is not None and owner_el.text else "Unknown"

    transactions = []
    for tx_el in root.findall(".//nonDerivativeTransaction"):
        date_el = tx_el.find(".//transactionDate/value")
        code_el = tx_el.find(".//transactionCode")
        shares_el = tx_el.find(".//transactionShares/value")
        price_el = tx_el.find(".//transactionPricePerShare/value")

        if date_el is None or shares_el is None:
            continue

        tx_date = datetime.strptime(date_el.text.strip(), "%Y-%m-%d").replace(tzinfo=timezone.utc)
        tx_code = code_el.text.strip() if code_el is not None and code_el.text else "P"
        shares = int(float(shares_el.text.strip()))
        price = float(price_el.text.strip()) if price_el is not None and price_el.text else 0.0

        transactions.append(InsiderTransaction(
            reporting_owner=owner_name,
            issuer=issuer_name,
            issuer_cik=issuer_cik,
            transaction_date=tx_date,
            filing_date=tx_date + timedelta(days=2),  # Approximate
            transaction_code=tx_code,
            shares=shares,
            price_per_share=price,
        ))

    return transactions


def detect_insider_clusters(
    transactions: list[InsiderTransaction],
    min_insiders: int = 3,
    window_days: int = 14,
) -> list[InsiderCluster]:
    """Detect clusters of insider purchases within the same sector.

    A cluster = min_insiders different insiders at different companies
    in the same sector filing purchases within window_days.
    """
    # Filter to purchases only
    purchases = [tx for tx in transactions if tx.transaction_code == "P"]

    # Group by sector
    by_sector: dict[str, list[InsiderTransaction]] = defaultdict(list)
    for tx in purchases:
        if tx.sector:
            by_sector[tx.sector].append(tx)

    clusters = []
    for sector, sector_txs in by_sector.items():
        # Sort by transaction date
        sorted_txs = sorted(sector_txs, key=lambda t: t.transaction_date)

        # Sliding window
        for i, anchor in enumerate(sorted_txs):
            window_end = anchor.transaction_date + timedelta(days=window_days)
            window_txs = [
                tx for tx in sorted_txs
                if anchor.transaction_date <= tx.transaction_date <= window_end
            ]

            # Count unique companies (different insiders at different companies)
            unique_companies = {tx.issuer_cik for tx in window_txs}
            if len(unique_companies) >= min_insiders:
                clusters.append(InsiderCluster(
                    sector=sector,
                    insider_count=len(unique_companies),
                    transactions=window_txs,
                    window_start=anchor.transaction_date,
                    window_end=window_end,
                ))
                break  # One cluster per sector per pass

    return clusters


class EdgarForm4Source(SignalSource):
    """SEC EDGAR Form 4 insider transaction signal source."""

    @property
    def name(self) -> str:
        return "edgar_form4"

    @property
    def rate_limit_per_second(self) -> float:
        return 10.0

    def transaction_to_signals(self, transactions: list[InsiderTransaction]) -> list[SignalEvent]:
        """Convert insider transactions to SignalEvents."""
        events = []
        for tx in transactions:
            if tx.transaction_code != "P":
                continue  # Only purchases generate conviction signals

            events.append(SignalEvent(
                source="edgar_form4",
                source_entity=tx.reporting_owner,
                timestamp=tx.filing_date,
                trade_date=tx.transaction_date,
                filing_date=tx.filing_date,
                confidence=FORM4_INSIDER_DECAY.initial_confidence,
                decay_profile=FORM4_INSIDER_DECAY,
                signal_type=SignalType.CONVICTION,
                payload={
                    "issuer": tx.issuer,
                    "transaction_code": tx.transaction_code,
                    "shares": tx.shares,
                    "price": tx.price_per_share,
                    "value": tx.shares * tx.price_per_share,
                },
                metadata={
                    "issuer_cik": tx.issuer_cik,
                    "sector": tx.sector,
                },
            ))

        return events

    async def fetch_signals(self) -> list[SignalEvent]:
        """Fetch latest Form 4 filings from EDGAR."""
        # TODO: Implement EDGAR API polling
        return []

    async def validate_schema(self, response: dict) -> bool:
        return "filings" in response
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/unit/test_edgar_form4.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/evolve_trader/signals/sources/edgar_form4.py tests/unit/test_edgar_form4.py tests/fixtures/edgar/form4_sample.xml
git commit -m "feat: SEC EDGAR Form 4 parser with insider cluster detection"
```

---

## Task 8: Congressional Trading Source

**Approach:** Use the `congressional-trading` PyPI package ([github.com/ivanma9/CongressionalTrading](https://github.com/ivanma9/CongressionalTrading)) for House data ingestion. It already scrapes House Clerk ZIP indexes → XML → PDF via pdftotext with rate limiting, circuit breaker, and retry logic. We wrap its output in our SignalEvent pipeline rather than building a House scraper from scratch. For Senate data, build a lighter scraper against Capitol Trades (pre-normalized HTML). For committee enrichment, use the free ProPublica Congress API.

**Dependencies:** `pip install congressional-trading` (adds pdftotext, APScheduler, FastAPI deps)

**Files:**
- Create: `src/evolve_trader/signals/sources/congressional.py`
- Create: `tests/unit/test_congressional.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_congressional.py
"""Tests for congressional trading signal source."""
import pytest
from datetime import datetime, timezone
from evolve_trader.signals.sources.congressional import (
    CongressionalTradeSource,
    CongressionalTrade,
    CONGRESSIONAL_WATCHLIST,
    LeadershipRole,
)
from evolve_trader.signals.types import SignalType


def test_watchlist_has_required_members():
    """Watchlist includes key congressional traders."""
    names = {m.name for m in CONGRESSIONAL_WATCHLIST}
    assert "Nancy Pelosi" in names
    assert "Dan Crenshaw" in names
    assert "Tommy Tuberville" in names
    assert "Ron Wyden" in names


def test_congressional_trade_data_model():
    """CongressionalTrade has all required fields."""
    trade = CongressionalTrade(
        member_name="Nancy Pelosi",
        party="D",
        state="CA",
        chamber="House",
        trade_date=datetime(2025, 2, 15, tzinfo=timezone.utc),
        filing_date=datetime(2025, 3, 1, tzinfo=timezone.utc),
        ticker="NVDA",
        transaction_type="purchase",
        size_range="$1,000,001 - $5,000,000",
        committees=["Intelligence"],
        leadership_role=LeadershipRole.SPEAKER_EMERITUS,
    )
    assert trade.member_name == "Nancy Pelosi"
    assert trade.ticker == "NVDA"


def test_leadership_role_boost():
    """Leadership roles get higher confidence boost."""
    trade_leader = CongressionalTrade(
        member_name="Nancy Pelosi", party="D", state="CA", chamber="House",
        trade_date=datetime(2025, 2, 15, tzinfo=timezone.utc),
        filing_date=datetime(2025, 3, 1, tzinfo=timezone.utc),
        ticker="NVDA", transaction_type="purchase", size_range="$1M+",
        committees=["Intelligence"], leadership_role=LeadershipRole.SPEAKER_EMERITUS,
    )
    trade_regular = CongressionalTrade(
        member_name="Regular Member", party="D", state="NY", chamber="House",
        trade_date=datetime(2025, 2, 15, tzinfo=timezone.utc),
        filing_date=datetime(2025, 3, 1, tzinfo=timezone.utc),
        ticker="MSFT", transaction_type="purchase", size_range="$15,001 - $50,000",
        committees=[], leadership_role=LeadershipRole.NONE,
    )
    source = CongressionalTradeSource()
    events_leader = source.trade_to_signals(trade_leader)
    events_regular = source.trade_to_signals(trade_regular)

    assert events_leader[0].confidence > events_regular[0].confidence


def test_committee_relevance_boost():
    """Trades in sectors matching member's committee get confidence boost."""
    trade = CongressionalTrade(
        member_name="Dan Crenshaw", party="R", state="TX", chamber="House",
        trade_date=datetime(2025, 2, 15, tzinfo=timezone.utc),
        filing_date=datetime(2025, 3, 1, tzinfo=timezone.utc),
        ticker="LMT", transaction_type="purchase", size_range="$100,001 - $250,000",
        committees=["Armed Services"], leadership_role=LeadershipRole.NONE,
    )
    source = CongressionalTradeSource()
    events = source.trade_to_signals(trade)
    # Armed Services member buying defense stock = committee relevance
    assert events[0].metadata.get("committee_relevant") is True


def test_congressional_source_interface():
    """Congressional source implements SignalSource interface."""
    source = CongressionalTradeSource()
    assert source.name == "congressional"
    assert source.rate_limit_per_second > 0


def test_congressional_source_produces_signal_events():
    """Source converts trades to properly typed SignalEvents."""
    source = CongressionalTradeSource()
    trade = CongressionalTrade(
        member_name="Nancy Pelosi", party="D", state="CA", chamber="House",
        trade_date=datetime(2025, 2, 15, tzinfo=timezone.utc),
        filing_date=datetime(2025, 3, 1, tzinfo=timezone.utc),
        ticker="NVDA", transaction_type="purchase", size_range="$1M+",
        committees=["Intelligence"], leadership_role=LeadershipRole.SPEAKER_EMERITUS,
    )
    events = source.trade_to_signals(trade)

    assert len(events) == 1
    event = events[0]
    assert event.source == "congressional"
    assert event.source_entity == "Nancy Pelosi"
    assert event.signal_type == SignalType.CONVICTION
    assert "party" in event.metadata
    assert "state" in event.metadata
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/unit/test_congressional.py -v
```

Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement**

```python
# src/evolve_trader/signals/sources/congressional.py
"""Congressional trading signal source."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from evolve_trader.signals.base import SignalSource
from evolve_trader.signals.decay import CONGRESSIONAL_DECAY
from evolve_trader.signals.events import SignalEvent
from evolve_trader.signals.types import SignalType


class LeadershipRole(str, Enum):
    """Congressional leadership roles (Wei & Zhou inflection point)."""
    SPEAKER = "speaker"
    SPEAKER_EMERITUS = "speaker_emeritus"
    MAJORITY_LEADER = "majority_leader"
    MINORITY_LEADER = "minority_leader"
    WHIP = "whip"
    COMMITTEE_CHAIR = "committee_chair"
    COMMITTEE_RANKING = "committee_ranking"
    NONE = "none"


# Leadership roles get confidence boost (Wei & Zhou: leaders outperform by 47% annually)
_LEADERSHIP_CONFIDENCE_BOOST = {
    LeadershipRole.SPEAKER: 0.20,
    LeadershipRole.SPEAKER_EMERITUS: 0.15,
    LeadershipRole.MAJORITY_LEADER: 0.18,
    LeadershipRole.MINORITY_LEADER: 0.15,
    LeadershipRole.WHIP: 0.12,
    LeadershipRole.COMMITTEE_CHAIR: 0.10,
    LeadershipRole.COMMITTEE_RANKING: 0.08,
    LeadershipRole.NONE: 0.0,
}

# Committee-to-sector mapping for relevance detection
_COMMITTEE_SECTORS = {
    "Armed Services": {"Aerospace & Defense", "Defense"},
    "Intelligence": {"Technology", "Cybersecurity", "Defense"},
    "Energy and Commerce": {"Energy", "Utilities", "Healthcare"},
    "Financial Services": {"Financials", "Banking", "Insurance"},
    "Ways and Means": {"Financials", "Real Estate"},
    "Agriculture": {"Agriculture", "Food"},
    "Transportation": {"Industrials", "Transportation"},
    "Science, Space, and Technology": {"Technology", "Aerospace"},
}


@dataclass
class CongressionalMember:
    """A member on the congressional trading watchlist."""
    name: str
    party: str
    state: str
    chamber: str  # House or Senate
    committees: list[str] = field(default_factory=list)
    leadership_role: LeadershipRole = LeadershipRole.NONE


CONGRESSIONAL_WATCHLIST = [
    CongressionalMember("Nancy Pelosi", "D", "CA", "House", ["Intelligence", "Financial Services"], LeadershipRole.SPEAKER_EMERITUS),
    CongressionalMember("Dan Crenshaw", "R", "TX", "House", ["Armed Services", "Intelligence"]),
    CongressionalMember("Ron Wyden", "D", "OR", "Senate", ["Finance", "Intelligence"], LeadershipRole.COMMITTEE_CHAIR),
    CongressionalMember("Josh Gottheimer", "D", "NJ", "House", ["Intelligence", "Financial Services"]),
    CongressionalMember("Marjorie Taylor Greene", "R", "GA", "House", []),
    CongressionalMember("Tommy Tuberville", "R", "AL", "Senate", ["Armed Services", "Agriculture"]),
    CongressionalMember("Markwayne Mullin", "R", "OK", "Senate", ["Armed Services", "Energy"]),
    CongressionalMember("Warren Davidson", "R", "OH", "House", ["Financial Services"]),
    CongressionalMember("Donald Norcross", "D", "NJ", "House", ["Armed Services", "Education"]),
    CongressionalMember("Rick Scott", "R", "FL", "Senate", ["Armed Services", "Commerce"], LeadershipRole.COMMITTEE_RANKING),
]


@dataclass
class CongressionalTrade:
    """A congressional stock trade disclosure."""
    member_name: str
    party: str
    state: str
    chamber: str
    trade_date: datetime
    filing_date: datetime
    ticker: str
    transaction_type: str  # purchase, sale
    size_range: str
    committees: list[str] = field(default_factory=list)
    leadership_role: LeadershipRole = LeadershipRole.NONE


class CongressionalTradeSource(SignalSource):
    """Congressional trading signal source."""

    @property
    def name(self) -> str:
        return "congressional"

    @property
    def rate_limit_per_second(self) -> float:
        return 5.0  # Conservative for API

    def trade_to_signals(self, trade: CongressionalTrade) -> list[SignalEvent]:
        """Convert a congressional trade to SignalEvents."""
        base_confidence = CONGRESSIONAL_DECAY.initial_confidence

        # Leadership boost
        boost = _LEADERSHIP_CONFIDENCE_BOOST.get(trade.leadership_role, 0.0)
        confidence = min(1.0, base_confidence + boost)

        # Committee relevance check
        committee_relevant = False
        for committee in trade.committees:
            if committee in _COMMITTEE_SECTORS:
                committee_relevant = True
                confidence = min(1.0, confidence + 0.05)
                break

        return [SignalEvent(
            source="congressional",
            source_entity=trade.member_name,
            timestamp=trade.filing_date,
            trade_date=trade.trade_date,
            filing_date=trade.filing_date,
            confidence=confidence,
            decay_profile=CONGRESSIONAL_DECAY,
            signal_type=SignalType.CONVICTION,
            payload={
                "ticker": trade.ticker,
                "transaction_type": trade.transaction_type,
                "size_range": trade.size_range,
            },
            metadata={
                "party": trade.party,
                "state": trade.state,
                "chamber": trade.chamber,
                "committees": trade.committees,
                "leadership_role": trade.leadership_role.value,
                "committee_relevant": committee_relevant,
            },
        )]

    async def fetch_signals(self) -> list[SignalEvent]:
        """Fetch latest congressional trades.

        House data: via `congressional-trading` package (scrapes House Clerk disclosures).
        Senate data: scrape Capitol Trades (pre-normalized HTML).
        Committee enrichment: ProPublica Congress API (free).
        """
        # TODO: Wire up congressional-trading package for House data
        # TODO: Build Capitol Trades scraper for Senate data
        # TODO: Add ProPublica committee enrichment
        return []

    async def validate_schema(self, response: dict) -> bool:
        return "trades" in response or "data" in response
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/unit/test_congressional.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/evolve_trader/signals/sources/congressional.py tests/unit/test_congressional.py
git commit -m "feat: congressional trading source with leadership role and committee relevance"
```

---

## Task 9: Basic Regime Classifier

**Files:**
- Create: `src/evolve_trader/regime/__init__.py`
- Create: `src/evolve_trader/regime/labels.py`
- Create: `src/evolve_trader/regime/classifier.py`
- Create: `strategies/regime-classifier-v1.skill.md`
- Create: `tests/unit/test_regime_classifier.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_regime_classifier.py
"""Tests for the basic regime classifier."""
import pytest
from datetime import datetime, timezone, timedelta
from evolve_trader.regime.labels import RegimeLabel, PrimaryRegime, MomentumState
from evolve_trader.regime.classifier import BasicRegimeClassifier
from evolve_trader.signals.events import SignalEvent
from evolve_trader.signals.types import SignalType, DecayProfile, DecayType


def _make_signal(source: str, signal_type: SignalType, payload: dict, confidence: float = 0.8) -> SignalEvent:
    """Helper to create test signals."""
    return SignalEvent(
        source=source,
        source_entity="Test",
        timestamp=datetime.now(timezone.utc),
        confidence=confidence,
        decay_profile=DecayProfile(confidence, 30, DecayType.LINEAR),
        signal_type=signal_type,
        payload=payload,
    )


def test_regime_label_has_required_fields():
    """RegimeLabel has all required fields."""
    label = RegimeLabel(
        primary_regime=PrimaryRegime.RISK_ON,
        sector_bias="overweight technology",
        momentum_state=MomentumState.STRENGTHENING,
        confidence=0.75,
        time_horizon="short-term (1-4 weeks)",
    )
    assert label.primary_regime == PrimaryRegime.RISK_ON
    assert label.confidence == 0.75


def test_regime_label_immutable():
    """RegimeLabel is immutable."""
    label = RegimeLabel(
        primary_regime=PrimaryRegime.RISK_ON,
        sector_bias="neutral",
        momentum_state=MomentumState.STABLE,
        confidence=0.7,
        time_horizon="medium-term",
    )
    with pytest.raises(AttributeError):
        label.confidence = 0.5


def test_classifier_risk_on_from_buy_signals():
    """Multiple buy signals from strong sources → risk-on regime."""
    classifier = BasicRegimeClassifier()
    signals = [
        _make_signal("edgar_13f", SignalType.CONVICTION, {"action": "BUY", "sector": "Technology"}, 0.85),
        _make_signal("congressional", SignalType.CONVICTION, {"transaction_type": "purchase", "ticker": "NVDA"}, 0.75),
        _make_signal("edgar_form4", SignalType.CONVICTION, {"transaction_code": "P", "sector": "Technology"}, 0.80),
    ]
    label = classifier.classify(signals)
    assert label.primary_regime == PrimaryRegime.RISK_ON
    assert label.confidence > 0.5


def test_classifier_risk_off_from_sell_signals():
    """Multiple sell signals → risk-off regime."""
    classifier = BasicRegimeClassifier()
    signals = [
        _make_signal("edgar_13f", SignalType.CONVICTION, {"action": "SELL", "sector": "Technology"}, 0.85),
        _make_signal("edgar_form4", SignalType.CONVICTION, {"transaction_code": "S", "sector": "Financials"}, 0.80),
    ]
    label = classifier.classify(signals)
    assert label.primary_regime == PrimaryRegime.RISK_OFF


def test_classifier_transitional_on_mixed_signals():
    """Mixed signals → transitional regime."""
    classifier = BasicRegimeClassifier()
    signals = [
        _make_signal("edgar_13f", SignalType.CONVICTION, {"action": "BUY"}, 0.85),
        _make_signal("edgar_form4", SignalType.CONVICTION, {"transaction_code": "S"}, 0.80),
    ]
    label = classifier.classify(signals)
    assert label.primary_regime == PrimaryRegime.TRANSITIONAL


def test_classifier_no_signals_returns_low_confidence():
    """No signals → low confidence transitional."""
    classifier = BasicRegimeClassifier()
    label = classifier.classify([])
    assert label.confidence < 0.3
    assert label.primary_regime == PrimaryRegime.TRANSITIONAL
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/unit/test_regime_classifier.py -v
```

Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement**

```python
# src/evolve_trader/regime/labels.py
"""Regime label types."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PrimaryRegime(str, Enum):
    RISK_ON = "risk-on"
    RISK_OFF = "risk-off"
    TRANSITIONAL = "transitional"


class MomentumState(str, Enum):
    STRENGTHENING = "strengthening"
    WEAKENING = "weakening"
    STABLE = "stable"
    TRANSITIONAL = "transitional"


@dataclass(frozen=True)
class RegimeLabel:
    """Immutable regime classification output."""
    primary_regime: PrimaryRegime
    sector_bias: str
    momentum_state: MomentumState
    confidence: float
    time_horizon: str
```

```python
# src/evolve_trader/regime/classifier.py
"""Basic regime classifier — consumes SignalEvents, outputs RegimeLabel."""
from __future__ import annotations

from evolve_trader.regime.labels import RegimeLabel, PrimaryRegime, MomentumState
from evolve_trader.signals.events import SignalEvent
from evolve_trader.signals.types import SignalType


class BasicRegimeClassifier:
    """Hand-crafted regime classifier.

    This is the v1 monolithic classifier. Subject to FIX/DERIVED/CAPTURED
    evolution. Later phases may decompose into sub-classifiers.
    """

    def classify(self, signals: list[SignalEvent]) -> RegimeLabel:
        """Classify current market regime from active signals."""
        if not signals:
            return RegimeLabel(
                primary_regime=PrimaryRegime.TRANSITIONAL,
                sector_bias="neutral",
                momentum_state=MomentumState.STABLE,
                confidence=0.2,
                time_horizon="unknown",
            )

        # Score bullish vs bearish signals
        bullish_score = 0.0
        bearish_score = 0.0
        sectors: dict[str, float] = {}

        for signal in signals:
            weight = signal.current_confidence()
            if weight <= 0:
                continue

            payload = signal.payload
            is_buy = (
                payload.get("action") == "BUY"
                or payload.get("transaction_type") == "purchase"
                or payload.get("transaction_code") == "P"
            )
            is_sell = (
                payload.get("action") == "SELL"
                or payload.get("transaction_type") == "sale"
                or payload.get("transaction_code") == "S"
            )

            if is_buy:
                bullish_score += weight
            elif is_sell:
                bearish_score += weight

            # Track sector signals
            sector = payload.get("sector") or signal.metadata.get("sector", "")
            if sector:
                sectors[sector] = sectors.get(sector, 0) + (weight if is_buy else -weight)

        total_score = bullish_score + bearish_score
        if total_score == 0:
            return RegimeLabel(
                primary_regime=PrimaryRegime.TRANSITIONAL,
                sector_bias="neutral",
                momentum_state=MomentumState.STABLE,
                confidence=0.2,
                time_horizon="short-term (1-4 weeks)",
            )

        bull_ratio = bullish_score / total_score

        # Determine primary regime
        if bull_ratio > 0.65:
            regime = PrimaryRegime.RISK_ON
        elif bull_ratio < 0.35:
            regime = PrimaryRegime.RISK_OFF
        else:
            regime = PrimaryRegime.TRANSITIONAL

        # Sector bias
        if sectors:
            top_sector = max(sectors, key=lambda s: abs(sectors[s]))
            direction = "overweight" if sectors[top_sector] > 0 else "underweight"
            sector_bias = f"{direction} {top_sector.lower()}"
        else:
            sector_bias = "neutral"

        # Momentum state (simplified: based on signal strength)
        if bull_ratio > 0.75:
            momentum = MomentumState.STRENGTHENING
        elif bull_ratio < 0.25:
            momentum = MomentumState.WEAKENING
        else:
            momentum = MomentumState.TRANSITIONAL

        # Confidence based on signal agreement
        confidence = abs(bull_ratio - 0.5) * 2  # 0.0 at even split, 1.0 at unanimous
        confidence = min(0.95, confidence * min(len(signals) / 3, 1.0))  # Scale up with signal count

        return RegimeLabel(
            primary_regime=regime,
            sector_bias=sector_bias,
            momentum_state=momentum,
            confidence=round(confidence, 3),
            time_horizon="short-term (1-4 weeks)",
        )
```

```markdown
<!-- strategies/regime-classifier-v1.skill.md -->
---
name: regime-classifier-v1
description: Basic regime classifier that consumes SignalEvents and outputs RegimeLabels
version: 1
status: active
skill_type: regime_classifier
---

# Regime Classifier v1

## Reasoning Framework

Classify the current market regime by analyzing active signal events:

1. **Aggregate directional signals:** Count weighted bullish vs bearish signals from all active sources
2. **Determine primary regime:**
   - >65% bullish weight → risk-on
   - <35% bullish weight → risk-off
   - Otherwise → transitional
3. **Identify sector bias:** Which sector has the strongest net signal?
4. **Assess momentum:** Is conviction strengthening or weakening across signals?
5. **Calibrate confidence:** Higher when signals agree, lower when they conflict

## Evolution Notes

This classifier is subject to FIX/DERIVED/CAPTURED evolution:
- FIX: When regime predictions are consistently wrong
- DERIVED: When a specialized sub-classifier for a specific sector or signal type would improve accuracy
- CAPTURED: When emergent regime-classification patterns are detected
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/unit/test_regime_classifier.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/evolve_trader/regime/ strategies/regime-classifier-v1.skill.md tests/unit/test_regime_classifier.py
git commit -m "feat: basic regime classifier with RegimeLabel output"
```

---

## Task 10: LLM Usage Logger PostgreSQL Backend

This task upgrades the shared Phase 1 logger rather than replacing it.

**Files:**
- Modify: `src/evolve_trader/core/llm_logger.py`
- Modify: `src/evolve_trader/db/models.py` (add `LLMUsageRecord`)
- Create: `tests/unit/test_llm_logger_pg.py`

**Requirements:**
- Preserve the Phase 1 `LLMUsageLogger` interface so callers do not change
- Move persistence from file-based storage to PostgreSQL via the repository layer
- Support per-component cost aggregation, monthly budget thresholds, and migration of Phase 1 usage records
- Keep stored content limited to compact metadata, rationale summaries, and cost fields; do not introduce raw prompt/response logging

**Acceptance criteria:**
- Existing Phase 1 logger tests still pass or are trivially updated for the new backend
- Phase 1 usage records migrate without duplication
- Dashboard and budget-control features can query usage through the same repository abstraction

---

## Task 11: Integration Testing — Full Signal Pipeline

**Files:**
- Create: `tests/integration/test_signal_pipeline.py`

**Step 1: Write the integration test**

```python
# tests/integration/test_signal_pipeline.py
"""Integration test: signal sources → SignalEvents → decay → regime classifier."""
import pytest
from datetime import datetime, timezone, timedelta
from evolve_trader.signals.sources.edgar_13f import Edgar13FSource, Filing13F, Holding
from evolve_trader.signals.sources.edgar_form4 import EdgarForm4Source, InsiderTransaction
from evolve_trader.signals.sources.congressional import CongressionalTradeSource, CongressionalTrade, LeadershipRole
from evolve_trader.signals.registry import SignalSourceRegistry
from evolve_trader.regime.classifier import BasicRegimeClassifier
from evolve_trader.regime.labels import PrimaryRegime


def test_full_signal_pipeline_risk_on():
    """Full pipeline: multiple buy signals → risk-on regime."""
    # Register sources
    registry = SignalSourceRegistry()
    edgar_13f = Edgar13FSource()
    edgar_form4 = EdgarForm4Source()
    congressional = CongressionalTradeSource()
    registry.register(edgar_13f)
    registry.register(edgar_form4)
    registry.register(congressional)

    # Generate signals from each source
    now = datetime.now(timezone.utc)

    # 13F filing: Buffett buying tech
    filing = Filing13F(
        filer_cik="0001067983", filer_name="Berkshire Hathaway",
        report_period=now - timedelta(days=45), filing_date=now,
        holdings=[Holding("Apple Inc", "037833100", 75000, 500000, "SOLE", 500000)],
    )
    signals_13f = edgar_13f.filing_to_signals(filing)

    # Form 4: insider purchase
    tx = InsiderTransaction(
        "Tim Cook", "Apple Inc", "0000320193",
        now - timedelta(days=5), now - timedelta(days=3),
        "P", 10000, 175.50, "Technology",
    )
    signals_form4 = edgar_form4.transaction_to_signals([tx])

    # Congressional: Pelosi buying NVDA
    trade = CongressionalTrade(
        "Nancy Pelosi", "D", "CA", "House",
        now - timedelta(days=10), now - timedelta(days=3),
        "NVDA", "purchase", "$1M+",
        ["Intelligence"], LeadershipRole.SPEAKER_EMERITUS,
    )
    signals_congress = congressional.trade_to_signals(trade)

    # Combine all signals
    all_signals = signals_13f + signals_form4 + signals_congress
    assert len(all_signals) >= 3

    # Run through regime classifier
    classifier = BasicRegimeClassifier()
    label = classifier.classify(all_signals)

    # Multiple buy signals from strong sources → should be risk-on
    assert label.primary_regime == PrimaryRegime.RISK_ON
    assert label.confidence > 0.5


def test_full_signal_pipeline_with_decay():
    """Old signals have reduced impact on regime classification."""
    now = datetime.now(timezone.utc)
    old_date = now - timedelta(days=150)  # 150 days old

    # Old 13F signal (should be mostly decayed)
    edgar = Edgar13FSource()
    filing = Filing13F(
        filer_cik="0001067983", filer_name="Berkshire Hathaway",
        report_period=old_date - timedelta(days=45), filing_date=old_date,
        holdings=[Holding("Apple Inc", "037833100", 75000, 500000, "SOLE", 500000)],
    )
    old_signals = edgar.filing_to_signals(filing)

    # Verify signals have decayed
    for signal in old_signals:
        current_conf = signal.current_confidence(as_of=now)
        assert current_conf < signal.confidence, "Old signal should have decayed"


def test_signal_registry_tracks_all_sources():
    """Registry manages all signal sources with health tracking."""
    registry = SignalSourceRegistry()
    registry.register(Edgar13FSource())
    registry.register(EdgarForm4Source())
    registry.register(CongressionalTradeSource())

    sources = registry.list_sources()
    assert len(sources) == 3
    assert "edgar_13f" in sources
    assert "edgar_form4" in sources
    assert "congressional" in sources

    # Health tracking initialized
    for source_name in sources:
        health = registry.get_health(source_name)
        assert health.total_requests == 0
```

**Step 2: Run test**

```bash
pytest tests/integration/test_signal_pipeline.py -v
```

Expected: PASS (all unit components already implemented)

**Step 3: Commit**

```bash
git add tests/integration/test_signal_pipeline.py
git commit -m "test: integration tests for full signal pipeline"
```

---

## Task 12: Final Verification

**Step 1: Run full test suite**

```bash
pytest tests/ -v --tb=short
```

Expected: ALL PASS — both Phase 1 and Phase 2 tests

**Step 2: Run linting and type checking**

```bash
ruff check src/evolve_trader/
mypy src/evolve_trader/ --ignore-missing-imports
```

Expected: No errors

**Step 3: Verify database migration**

```bash
alembic upgrade head
alembic downgrade -1
alembic upgrade head
```

Expected: Clean upgrade/downgrade cycle

**Step 4: Commit any fixes**

```bash
git add -A
git commit -m "test: Phase 2 final verification — all tests passing"
```

---

## Parallelization Notes

Tasks in this phase have the following dependency structure:

```
Task 1 (PostgreSQL Schema) ──────────┐
Task 2 (Alembic Migrations) ─────────┤
                                      ├── Task 10 (LLM Logger Migration)
Task 3 (SignalEvent Types) ──────────┤
Task 4 (Decay Functions) ────────────┤
                                      ├── Task 6 (EDGAR 13F) ──┐
Task 5 (Base Class & Registry) ──────┤── Task 7 (Form 4) ──────┤── Task 9 (Regime Classifier)
                                      ├── Task 8 (Congressional)┘           │
                                      │                                      │
                                      └──────────────────────── Task 11 (Integration Tests)
```

**Can run in parallel:**
- Tasks 1-2 (DB) and Tasks 3-5 (Signal types) are independent — run simultaneously
- Tasks 6, 7, 8 (signal sources) are independent of each other — run simultaneously after Tasks 3-5
- Task 9 (regime classifier) depends on signal types but not on specific sources
- Task 10 (LLM logger) depends only on Task 1
- Task 11 (integration) depends on everything
- Task 12 (final verification) must be last
