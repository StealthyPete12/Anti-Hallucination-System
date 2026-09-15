"""Integration test: FeedbackRepository against a real Postgres engine.

Exercises app/db/session.py's actual engine-building code path (unlike
tests/unit/test_feedback_repository.py, which substitutes SQLite for speed)
so the Postgres-specific SQL SQLAlchemy generates - reserved-word handling,
timezone-aware timestamps, autoincrement - is checked against the real
dialect at least once.

Skipped automatically if RAG_POSTGRES_DSN isn't reachable (e.g. no
`docker compose up postgres` running locally, or CI without a Postgres
service container) - same convention as test_pipeline_e2e.py skipping on a
missing embedding model.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker

from app.db.models import Base
from app.db.repository import (
    FeedbackRepository,
    RequestMetricRepository,
    feedback_latency_correlation_query,
)
from app.db.session import build_engine
from config import settings

pytestmark = pytest.mark.integration


@pytest.fixture
def session():  # type: ignore[no-untyped-def]
    engine = build_engine(settings)
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except OperationalError as exc:
        pytest.skip(f"Postgres not reachable at {settings.postgres_dsn}: {exc}")

    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    db = factory()
    try:
        yield db
    finally:
        db.rollback()
        db.execute(text("DELETE FROM feedback"))
        db.execute(text("DELETE FROM request_metrics"))
        db.commit()
        db.close()


def test_create_and_retrieve_feedback_against_real_postgres(session) -> None:  # type: ignore[no-untyped-def]
    repo = FeedbackRepository(session)
    repo.create(trace_id="pg-trace-1", rating=1, comment="accurate and fast")

    results = repo.get_by_trace_id("pg-trace-1")
    assert len(results) == 1
    assert results[0].rating == 1
    assert results[0].comment == "accurate and fast"
    assert results[0].created_at is not None


def test_list_recent_against_real_postgres(session) -> None:  # type: ignore[no-untyped-def]
    repo = FeedbackRepository(session)
    repo.create(trace_id="pg-trace-a", rating=1)
    repo.create(trace_id="pg-trace-b", rating=-1)

    recent = repo.list_recent(limit=10)
    assert len(recent) == 2


def test_feedback_correlates_with_request_metrics_via_real_postgres(session) -> None:  # type: ignore[no-untyped-def]
    """Proves the actual analysis question section 6 exists to answer:
    joining feedback.rating against request_metrics.latency_ms by trace_id."""
    feedback_repo = FeedbackRepository(session)
    metric_repo = RequestMetricRepository(session)

    metric_repo.create(
        trace_id="slow-trace",
        cost_usd=0.01,
        latency_ms=5000.0,
        retrieval_mode="dense",
        citation_validation_passed=True,
    )
    feedback_repo.create(trace_id="slow-trace", rating=-1, comment="too slow")

    metric_repo.create(
        trace_id="fast-trace",
        cost_usd=0.001,
        latency_ms=200.0,
        retrieval_mode="dense",
        citation_validation_passed=True,
    )
    feedback_repo.create(trace_id="fast-trace", rating=1)

    rows = session.execute(text(feedback_latency_correlation_query())).mappings().all()
    by_rating = {row["rating"]: row for row in rows}

    assert by_rating[-1]["avg_latency_ms"] == pytest.approx(5000.0)
    assert by_rating[1]["avg_latency_ms"] == pytest.approx(200.0)
