"""Create the Phase 5 observability database schema.

This project's migration strategy is deliberately lightweight: an additive,
idempotent ``CREATE TABLE IF NOT EXISTS`` via SQLAlchemy's
``Base.metadata.create_all`` (see ``app/db/session.py::init_db``), matching
the single-table schema's actual complexity. It also runs automatically on
API startup (``app/main.py``'s lifespan handler) so a fresh Postgres is
always usable without a manual step - this script exists for explicit,
scriptable use (CI, a one-off ops task) and to document the strategy in one
place. If the schema ever needs a destructive change (rename/drop a
column), that's the point to introduce Alembic; `create_all` only adds
missing tables, it never alters existing ones.

    uv run python -m scripts.migrate
"""

from __future__ import annotations

import logging

from app.db.session import build_engine, init_db
from config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
logger = logging.getLogger("scripts.migrate")


def main() -> None:
    engine = build_engine(settings)
    init_db(engine)
    logger.info("Schema is up to date at %s", settings.postgres_dsn)


if __name__ == "__main__":
    main()
