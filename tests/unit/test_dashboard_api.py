"""Tests for the FastAPI dashboard API."""

import os
from collections.abc import Generator
from datetime import UTC, datetime

from fastapi.testclient import TestClient
from sqlalchemy import StaticPool
from sqlalchemy.orm import Session

# Must be set before api import
os.environ["DATABASE_URL"] = "sqlite://"

from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from evolve_trader.dashboard.api import app, get_db  # noqa: E402
from evolve_trader.db.models import (  # noqa: E402
    Base,
    EvolutionEventRecord,
    LLMUsageRecord,
    SignalEventRecord,
    TradeLog,
)

# Single shared in-memory SQLite (StaticPool keeps one connection)
_test_engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
Base.metadata.create_all(_test_engine)
_TestSession = sessionmaker(bind=_test_engine)


def _get_test_db() -> Generator[Session, None, None]:
    session = _TestSession()
    try:
        yield session
    finally:
        session.close()


app.dependency_overrides[get_db] = _get_test_db

client = TestClient(app)


def _seed_data() -> None:
    """Insert test data into the shared test DB."""
    session = _TestSession()
    session.add(
        TradeLog(
            strategy_skill="momentum-v1",
            ticker="AAPL",
            direction="BUY",
            quantity=10,
            entry_price=150.0,
            exit_price=160.0,
            entry_date=datetime(2025, 1, 1, tzinfo=UTC),
            exit_date=datetime(2025, 1, 15, tzinfo=UTC),
            pnl=100.0,
            return_pct=0.0667,
            regime_label="risk-on",
            signal_sources=["edgar_13f"],
        )
    )
    session.add(
        SignalEventRecord(
            source="congressional",
            source_entity="Pelosi",
            timestamp=datetime(2025, 3, 15, tzinfo=UTC),
            confidence=0.85,
            decay_type="exponential",
            half_life_days=20,
            signal_type="conviction",
            payload={"ticker": "NVDA", "action": "BUY"},
        )
    )
    session.add(
        EvolutionEventRecord(
            event_type="FIX",
            parent_skill="momentum-v1",
            child_skill="momentum-v2",
            trigger_reason="Poor Sharpe",
            performance_before={"sharpe": 0.3},
        )
    )
    session.add(
        LLMUsageRecord(
            model="claude-sonnet-4",
            component="evolution",
            input_tokens=1000,
            output_tokens=500,
            cost_usd=0.01,
        )
    )
    session.commit()
    session.close()


_seed_data()


def test_health() -> None:
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_portfolio_latest_default() -> None:
    resp = client.get("/api/portfolio/latest")
    assert resp.status_code == 200
    assert resp.json()["total_value"] == 100_000


def test_trades() -> None:
    resp = client.get("/api/trades")
    assert resp.status_code == 200
    assert len(resp.json()) >= 1
    assert resp.json()[0]["ticker"] == "AAPL"


def test_trades_filter_by_strategy() -> None:
    resp = client.get("/api/trades?strategy=momentum-v1")
    assert resp.status_code == 200
    assert len(resp.json()) >= 1


def test_signals() -> None:
    resp = client.get("/api/signals")
    assert resp.status_code == 200
    assert len(resp.json()) >= 1
    assert resp.json()[0]["source"] == "congressional"


def test_signals_filter_by_source() -> None:
    resp = client.get("/api/signals?source=congressional")
    assert resp.status_code == 200
    assert len(resp.json()) >= 1


def test_evolution_events() -> None:
    resp = client.get("/api/evolution")
    assert resp.status_code == 200
    assert len(resp.json()) >= 1
    assert resp.json()[0]["event_type"] == "FIX"


def test_evolution_lineage() -> None:
    resp = client.get("/api/evolution/lineage/momentum-v2")
    assert resp.status_code == 200
    assert len(resp.json()) >= 1


def test_llm_costs() -> None:
    resp = client.get("/api/costs")
    assert resp.status_code == 200
    assert resp.json()["total_cost_usd"] >= 0.01
    assert "evolution" in resp.json()["cost_by_component"]


def test_system_status() -> None:
    resp = client.get("/api/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["mode"] == "paper-training"
    assert data["total_trades"] >= 1
    assert data["total_signals"] >= 1
