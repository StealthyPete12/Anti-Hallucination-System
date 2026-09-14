"""Unit tests for app/db/repository.py's FeedbackRepository.

Runs against an in-memory SQLite engine rather than a real Postgres - the
schema and queries here are simple enough that SQLite's behavior matches
Postgres exactly for what's being tested, and it keeps these tests fast and
network-free (the same "fake/in-memory engine, no external dependency"
convention this project already uses for Qdrant's ``:memory:`` mode). A real
Postgres is exercised in tests/integration/test_postgres_feedback.py.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Base
from app.db.repository import FeedbackRepository


@pytest.fixture
def session() -> Iterator[Session]:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    db = factory()
    try:
        yield db
    finally:
        db.close()


def test_create_persists_feedback(session: Session) -> None:
    repo = FeedbackRepository(session)
    record = repo.create(trace_id="trace-1", rating=1, comment="great answer")

    assert record.id is not None
    assert record.trace_id == "trace-1"
    assert record.rating == 1
    assert record.comment == "great answer"
    assert record.created_at is not None


def test_create_without_comment(session: Session) -> None:
    repo = FeedbackRepository(session)
    record = repo.create(trace_id="trace-2", rating=-1)
    assert record.comment is None


def test_get_by_trace_id_returns_only_matching_records(session: Session) -> None:
    repo = FeedbackRepository(session)
    repo.create(trace_id="trace-a", rating=1)
    repo.create(trace_id="trace-a", rating=-1, comment="changed my mind")
    repo.create(trace_id="trace-b", rating=1)

    results = repo.get_by_trace_id("trace-a")
    assert len(results) == 2
    assert all(r.trace_id == "trace-a" for r in results)


def test_get_by_trace_id_returns_empty_for_unknown_trace(session: Session) -> None:
    repo = FeedbackRepository(session)
    assert repo.get_by_trace_id("nonexistent") == []


def test_list_recent_orders_newest_first(session: Session) -> None:
    repo = FeedbackRepository(session)
    first = repo.create(trace_id="t1", rating=1)
    second = repo.create(trace_id="t2", rating=-1)

    recent = repo.list_recent(limit=10)
    assert [r.id for r in recent] == [second.id, first.id]


def test_list_recent_respects_limit(session: Session) -> None:
    repo = FeedbackRepository(session)
    for i in range(5):
        repo.create(trace_id=f"t{i}", rating=1)
    assert len(repo.list_recent(limit=2)) == 2
