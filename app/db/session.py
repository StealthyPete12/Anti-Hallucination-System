"""SQLAlchemy engine/session wiring for the Phase 5 observability database.

Mirrors ``app/dependencies.py``'s pattern: a pure factory (``build_engine``)
plus a cached singleton (``get_engine``) that FastAPI's ``Depends`` can
override in tests - no real Postgres needed to unit test anything that
depends on ``get_db``.
"""

from __future__ import annotations

from collections.abc import Iterator
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from config import Settings, settings


def build_engine(settings: Settings) -> Engine:
    """A short ``connect_timeout`` matters here: if Postgres is unreachable
    (e.g. local dev without ``docker compose``, or a unit test importing
    ``app.main``), failing fast keeps the app's startup/request path from
    hanging - callers are expected to catch connection errors, not rely on
    this never failing."""
    connect_args = {}
    if settings.postgres_dsn.startswith("postgresql"):
        connect_args = {"connect_timeout": 3}
    return create_engine(settings.postgres_dsn, connect_args=connect_args, pool_pre_ping=True)


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    return build_engine(settings)


def init_db(engine: Engine) -> None:
    """Create every table that doesn't exist yet.

    This project's migration strategy: additive, idempotent
    ``create_all`` on startup, matching its lightweight-by-design approach
    elsewhere (no ORM migrations framework for a single-table schema). If
    the schema ever needs a destructive change (rename/drop a column), that
    is the point to introduce Alembic - ``create_all`` only ever adds
    missing tables, never alters existing ones.
    """
    from app.db.models import Base

    Base.metadata.create_all(engine)


def get_db() -> Iterator[Session]:
    """FastAPI dependency: yields a Session, closing it after the request."""
    session_factory = sessionmaker(bind=get_engine())
    db = session_factory()
    try:
        yield db
    finally:
        db.close()
