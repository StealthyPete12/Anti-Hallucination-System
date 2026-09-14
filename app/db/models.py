"""SQLAlchemy ORM models for the Phase 5 observability database.

Just one table for now (``feedback``). Kept as its own module - separate
from ``session.py`` (engine wiring) and ``repository.py`` (queries) - so
each has one job, matching this project's existing separation of concerns
(e.g. ``rag/models.py`` vs. ``rag/pipeline.py``).
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy import DateTime, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Feedback(Base):
    """One user rating for one traced request.

    ``trace_id`` is the same UUID string threaded through
    ``rag.pipeline.RagPipeline.query()``, the structured logs, and every
    OpenTelemetry span for that request - this column is what lets a
    thumbs-down be joined back to exactly the trace that produced it (Phoenix
    trace search, or a future analysis query against this table) without any
    additional bookkeeping.
    """

    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trace_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    rating: Mapped[int] = mapped_column(Integer, nullable=False)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: dt.datetime.now(dt.UTC)
    )
