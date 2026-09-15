"""SQLAlchemy ORM models for the Phase 5 observability database.

Two tables. Kept as its own module - separate from ``session.py`` (engine
wiring) and ``repository.py`` (queries) - so each has one job, matching this
project's existing separation of concerns (e.g. ``rag/models.py`` vs.
``rag/pipeline.py``).
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text
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


class RequestMetric(Base):
    """Cost/latency for one traced ``/query`` request, keyed by the same
    ``trace_id`` as ``Feedback``.

    Phoenix already carries this per-span, and ``rag/metrics.py`` carries an
    in-process rolling window of it - neither is enough on its own to answer
    "do thumbs-down responses correlate with higher latency?", which needs
    feedback and request cost/latency joined in one query. This table exists
    for exactly that join (see the README's Phase 5 "Trace correlation"
    section and the optional Grafana dashboard) - it is not a replacement
    for either of the other two, which remain the richer, per-span/real-time
    views.
    """

    __tablename__ = "request_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trace_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    cost_usd: Mapped[float] = mapped_column(Float, nullable=False)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    retrieval_mode: Mapped[str] = mapped_column(String(16), nullable=False)
    citation_validation_passed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: dt.datetime.now(dt.UTC)
    )
