"""Database engine/session helpers for the optional analytical layer."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.engine.url import make_url
from sqlalchemy.orm import Session, sessionmaker

from ew_housing_energy_impact.settings import get_settings


def get_engine(database_url: str | None = None) -> Engine:
    """Create a SQLAlchemy engine from DATABASE_URL or the safe local default."""

    url = database_url or get_settings().database_url
    parsed = make_url(url)
    if parsed.drivername.startswith("sqlite") and parsed.database not in {None, "", ":memory:"}:
        from pathlib import Path

        Path(parsed.database).parent.mkdir(parents=True, exist_ok=True)
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    return create_engine(url, future=True, connect_args=connect_args)


def create_tables(engine: Engine | None = None) -> None:
    """Create analytical serving tables using SQLAlchemy metadata."""

    from ew_housing_energy_impact.db.models import Base

    target_engine = engine or get_engine()
    Base.metadata.create_all(target_engine)


@contextmanager
def session_scope(engine: Engine | None = None) -> Iterator[Session]:
    """Provide a transactional SQLAlchemy session."""

    target_engine = engine or get_engine()
    session_factory = sessionmaker(bind=target_engine, future=True)
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
