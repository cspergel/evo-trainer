"""FastAPI dashboard API — read-heavy endpoints for system visibility.

Serves data to the Next.js frontend. No destructive actions in Phase 5.
Approval and kill-switch actions are deferred to Phases 6 and 11.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from evolve_trader.db.engine import create_db_engine, create_tables, get_session_factory
from evolve_trader.db.models import (
    EvolutionEventRecord,
    SignalEventRecord,
    TradeLog,
)
from evolve_trader.db.repositories import (
    EvolutionEventRepository,
    LLMUsageRepository,
    PortfolioSnapshotRepository,
    SignalEventRepository,
    TradeLogRepository,
)

app = FastAPI(
    title="Evolve-Trader Dashboard API",
    version="0.1.0",
    description="Read-heavy API for system visibility. Phase 5.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Database setup (lazy init) ---

_engine = None
_SessionFactory = None


def _init_db() -> None:
    global _engine, _SessionFactory  # noqa: PLW0603
    if _engine is None:
        db_url = os.environ.get("DATABASE_URL", "sqlite:///data/evolve_trader.db")
        _engine = create_db_engine(db_url)
        create_tables(_engine)
        _SessionFactory = get_session_factory(_engine)


def get_db() -> Session:  # type: ignore[misc]
    """Dependency that provides a database session."""
    _init_db()
    assert _SessionFactory is not None
    session = _SessionFactory()
    try:
        yield session
    finally:
        session.close()


# --- Health ---


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "phase": "5"}


# --- Portfolio ---


@app.get("/api/portfolio/latest")
def get_portfolio_latest(db: Session = Depends(get_db)) -> dict[str, Any]:
    """Latest portfolio snapshot."""
    repo = PortfolioSnapshotRepository(db)
    snapshot = repo.get_latest()
    if not snapshot:
        return {
            "total_value": 100_000,
            "cash": 100_000,
            "positions": {},
            "drawdown": 0.0,
            "sharpe_ratio": None,
            "recorded_at": datetime.now(UTC).isoformat(),
        }
    return {
        "total_value": snapshot.total_value,
        "cash": snapshot.cash,
        "positions": snapshot.positions,
        "drawdown": snapshot.drawdown,
        "sharpe_ratio": snapshot.sharpe_ratio,
        "recorded_at": snapshot.recorded_at.isoformat(),
    }


# --- Trades ---


@app.get("/api/trades")
def get_trades(
    strategy: str | None = None,
    ticker: str | None = None,
    limit: int = 50,
    db: Session = Depends(get_db),
) -> list[dict[str, Any]]:
    """Trade history, optionally filtered by strategy or ticker."""
    repo = TradeLogRepository(db)
    if strategy:
        trades = repo.get_by_strategy(strategy)
    elif ticker:
        trades = repo.get_by_ticker(ticker)
    else:
        from sqlalchemy import select

        stmt = select(TradeLog).order_by(TradeLog.created_at.desc()).limit(limit)
        trades = list(db.scalars(stmt).all())

    return [
        {
            "id": t.id,
            "strategy_skill": t.strategy_skill,
            "ticker": t.ticker,
            "direction": t.direction,
            "quantity": t.quantity,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "pnl": t.pnl,
            "return_pct": t.return_pct,
            "regime_label": t.regime_label,
            "signal_sources": t.signal_sources,
            "entry_date": t.entry_date.isoformat() if t.entry_date else None,
            "exit_date": t.exit_date.isoformat() if t.exit_date else None,
        }
        for t in trades
    ]


# --- Signals ---


@app.get("/api/signals")
def get_signals(
    source: str | None = None,
    limit: int = 50,
    db: Session = Depends(get_db),
) -> list[dict[str, Any]]:
    """Recent signal events, optionally filtered by source."""
    repo = SignalEventRepository(db)
    if source:
        signals = repo.get_by_source(source, limit=limit)
    else:
        from sqlalchemy import select

        stmt = select(SignalEventRecord).order_by(SignalEventRecord.created_at.desc()).limit(limit)
        signals = list(db.scalars(stmt).all())

    return [
        {
            "id": s.id,
            "source": s.source,
            "source_entity": s.source_entity,
            "confidence": s.confidence,
            "signal_type": s.signal_type,
            "decay_type": s.decay_type,
            "half_life_days": s.half_life_days,
            "payload": s.payload,
            "timestamp": s.timestamp.isoformat(),
        }
        for s in signals
    ]


# --- Evolution ---


@app.get("/api/evolution")
def get_evolution_events(
    limit: int = 50,
    db: Session = Depends(get_db),
) -> list[dict[str, Any]]:
    """Recent evolution events (FIX/DERIVED/CAPTURED)."""
    from sqlalchemy import select

    stmt = (
        select(EvolutionEventRecord).order_by(EvolutionEventRecord.created_at.desc()).limit(limit)
    )
    events = list(db.scalars(stmt).all())

    return [
        {
            "id": e.id,
            "event_type": e.event_type,
            "parent_skill": e.parent_skill,
            "child_skill": e.child_skill,
            "trigger_reason": e.trigger_reason,
            "performance_before": e.performance_before,
            "performance_after": e.performance_after,
            "created_at": e.created_at.isoformat(),
        }
        for e in events
    ]


@app.get("/api/evolution/lineage/{skill_name}")
def get_skill_lineage(
    skill_name: str,
    db: Session = Depends(get_db),
) -> list[dict[str, Any]]:
    """Get evolution lineage for a specific skill."""
    repo = EvolutionEventRepository(db)
    events = repo.get_lineage(skill_name)
    return [
        {
            "event_type": e.event_type,
            "parent_skill": e.parent_skill,
            "child_skill": e.child_skill,
            "trigger_reason": e.trigger_reason,
            "created_at": e.created_at.isoformat(),
        }
        for e in events
    ]


# --- LLM Costs ---


@app.get("/api/costs")
def get_llm_costs(db: Session = Depends(get_db)) -> dict[str, Any]:
    """LLM cost summary."""
    repo = LLMUsageRepository(db)
    return {
        "total_cost_usd": repo.total_cost(),
        "cost_by_component": repo.cost_by_component(),
    }


# --- System Status ---


@app.get("/api/status")
def get_system_status(db: Session = Depends(get_db)) -> dict[str, Any]:
    """Overall system status for operator visibility."""
    from sqlalchemy import func, select

    strategy_count = db.scalar(select(func.count()).select_from(TradeLog)) or 0
    signal_count = db.scalar(select(func.count()).select_from(SignalEventRecord)) or 0
    evolution_count = db.scalar(select(func.count()).select_from(EvolutionEventRecord)) or 0

    return {
        "mode": "paper-training",  # Phase 6 will make this dynamic
        "total_trades": strategy_count,
        "total_signals": signal_count,
        "total_evolution_events": evolution_count,
        "kill_switch_status": "not_implemented",  # Phase 11
        "approval_queue_size": 0,  # Phase 6
        "evolution_paused": False,
    }
