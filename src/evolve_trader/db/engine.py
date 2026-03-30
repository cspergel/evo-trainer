"""Database engine configuration.

Supports SQLite (dev/test) and PostgreSQL (production).
"""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from evolve_trader.db.models import Base


def create_db_engine(url: str = "sqlite:///:memory:") -> Engine:
    """Create a SQLAlchemy engine.

    Args:
        url: Database URL. Defaults to in-memory SQLite for testing.
             Use 'postgresql://...' for production.
    """
    connect_args = {}
    if url.startswith("sqlite"):
        connect_args["check_same_thread"] = False

    return create_engine(url, connect_args=connect_args)


def create_tables(engine: Engine) -> None:
    """Create all tables from ORM models."""
    Base.metadata.create_all(engine)


def get_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Create a session factory bound to the given engine."""
    return sessionmaker(bind=engine)
