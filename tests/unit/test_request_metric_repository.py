"""Unit tests for app/db/repository.py's RequestMetricRepository.

Same in-memory SQLite convention as test_feedback_repository.py - see that
file's docstring for why SQLite is an acceptable stand-in here.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Base
from app.db.repository import RequestMetricRepository, feedback_latency_correlation_query


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


def test_create_persists_request_metric(session: Session) -> None:
    repo = RequestMetricRepository(session)
    record = repo.create(
        trace_id="trace-1",
        cost_usd=0.0012,
        latency_ms=845.3,
        retrieval_mode="hybrid",
        citation_validation_passed=True,
    )

    assert record.id is not None
    assert record.trace_id == "trace-1"
    assert record.cost_usd == 0.0012
    assert record.latency_ms == 845.3
    assert record.retrieval_mode == "hybrid"
    assert record.citation_validation_passed is True
    assert record.created_at is not None


def test_create_records_citation_validation_failure(session: Session) -> None:
    repo = RequestMetricRepository(session)
    record = repo.create(
        trace_id="trace-2",
        cost_usd=0.002,
        latency_ms=1200.0,
        retrieval_mode="dense",
        citation_validation_passed=False,
    )
    assert record.citation_validation_passed is False


def test_feedback_latency_correlation_query_joins_on_trace_id() -> None:
    sql = feedback_latency_correlation_query()
    assert "feedback" in sql
    assert "request_metrics" in sql
    assert "trace_id" in sql
