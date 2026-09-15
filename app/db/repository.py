"""Repository for the ``feedback`` table (Phase 5).

Keeps every SQL/ORM detail behind a small, typed interface so
``app/main.py``'s ``/feedback`` handler (and future analysis code) never
touches SQLAlchemy sessions or query construction directly - the same
"protocols/pure functions in, framework details behind a seam" shape the
rest of this codebase uses (e.g. ``rag/protocols.py`` for pipeline stages).
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Feedback, RequestMetric


class FeedbackRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, *, trace_id: str, rating: int, comment: str | None = None) -> Feedback:
        record = Feedback(trace_id=trace_id, rating=rating, comment=comment)
        self.session.add(record)
        self.session.commit()
        self.session.refresh(record)
        return record

    def get_by_trace_id(self, trace_id: str) -> list[Feedback]:
        stmt = select(Feedback).where(Feedback.trace_id == trace_id)
        return list(self.session.scalars(stmt))

    def list_recent(self, limit: int = 100) -> list[Feedback]:
        stmt = select(Feedback).order_by(Feedback.created_at.desc()).limit(limit)
        return list(self.session.scalars(stmt))


class RequestMetricRepository:
    """Persists per-request cost/latency, correlated by ``trace_id`` with
    ``FeedbackRepository`` - see ``app.db.models.RequestMetric`` for why this
    table exists alongside Phoenix and ``rag/metrics.py``'s rolling window."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def create(
        self,
        *,
        trace_id: str,
        cost_usd: float,
        latency_ms: float,
        retrieval_mode: str,
        citation_validation_passed: bool,
    ) -> RequestMetric:
        record = RequestMetric(
            trace_id=trace_id,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            retrieval_mode=retrieval_mode,
            citation_validation_passed=citation_validation_passed,
        )
        self.session.add(record)
        self.session.commit()
        self.session.refresh(record)
        return record


def feedback_latency_correlation_query() -> str:
    """Raw SQL answering "do negative ratings correlate with higher latency?"
    - the exact analysis question this table exists to make possible (see the
    README's Phase 5 "Trace correlation" section). Returned as a string
    rather than executed here: it's meant for ad hoc use (a notebook, `psql`,
    Grafana's Postgres data source), not a typed repository return value."""
    return """
        SELECT
            f.rating,
            COUNT(*) AS num_requests,
            AVG(m.latency_ms) AS avg_latency_ms,
            AVG(m.cost_usd) AS avg_cost_usd,
            SUM(CASE WHEN NOT m.citation_validation_passed THEN 1 ELSE 0 END) AS hallucination_flags
        FROM feedback f
        JOIN request_metrics m ON m.trace_id = f.trace_id
        GROUP BY f.rating
        ORDER BY f.rating;
    """
