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

from app.db.models import Feedback


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
