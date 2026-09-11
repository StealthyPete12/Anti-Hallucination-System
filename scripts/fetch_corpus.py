"""One-time script: download the PostgreSQL 17 documentation pages that make
up this project's corpus into data/raw_html/.

This is a corpus-*acquisition* step, not part of the reusable ingestion
pipeline (rag/ingest.py). It is deliberately simple and re-runnable: given
the same PAGE_SLUGS list, it always fetches the same version-pinned URLs
(https://www.postgresql.org/docs/17/<slug>.html), so the raw corpus source
is reproducible. See docs/corpus_selection.md for why these pages and this
version were chosen.

Usage:
    python scripts/fetch_corpus.py
"""

from __future__ import annotations

import logging
import time
import urllib.error
import urllib.request
from pathlib import Path

logger = logging.getLogger("scripts.fetch_corpus")

POSTGRES_VERSION = "17"
BASE_URL = f"https://www.postgresql.org/docs/{POSTGRES_VERSION}"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw_html"

PAGE_SLUGS = [
    "tutorial-start",
    "tutorial-arch",
    "tutorial-createdb",
    "tutorial-accessdb",
    "tutorial-table",
    "tutorial-populate",
    "tutorial-select",
    "tutorial-join",
    "tutorial-agg",
    "tutorial-update",
    "tutorial-delete",
    "tutorial-window",
    "ddl-basics",
    "ddl-default",
    "ddl-constraints",
    "ddl-system-columns",
    "ddl-alter",
    "ddl-schemas",
    "ddl-inherit",
    "ddl-partitioning",
    "datatype-numeric",
    "datatype-character",
    "datatype-datetime",
    "datatype-boolean",
    "datatype-enum",
    "datatype-json",
    "datatype-uuid",
    "datatype-net-types",
    "dml-insert",
    "dml-update",
    "dml-delete",
    "dml-returning",
    "queries-overview",
    "queries-table-expressions",
    "queries-select-lists",
    "queries-union",
    "queries-order",
    "queries-limit",
    "queries-with",
    "indexes-intro",
    "indexes-types",
    "indexes-multicolumn",
    "indexes-unique",
    "indexes-expressional",
    "indexes-partial",
    "mvcc-intro",
    "mvcc-caveats",
    "transaction-iso",
    "explicit-locking",
    "backup-dump",
    "backup-file",
    "high-availability",
    "client-authentication",
    "role-attributes",
    "role-membership",
    "user-manag",
    "textsearch-intro",
    "functions-json",
    "functions-window",
    "runtime-config-connection",
]


def fetch_page(slug: str, out_dir: Path, timeout: float = 10.0) -> bool:
    url = f"{BASE_URL}/{slug}.html"
    dest = out_dir / f"{slug}.html"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
            dest.write_bytes(resp.read())
        logger.info("Fetched %s -> %s", url, dest)
        return True
    except urllib.error.HTTPError as exc:
        logger.warning("HTTP %s for %s (skipping)", exc.code, url)
        return False
    except OSError as exc:
        logger.error("Failed to fetch %s: %s", url, exc)
        return False


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ok, missing = 0, 0
    for slug in PAGE_SLUGS:
        if fetch_page(slug, OUT_DIR):
            ok += 1
        else:
            missing += 1
        time.sleep(0.1)  # be a polite client

    logger.info("Done: %d fetched, %d missing (of %d candidates)", ok, missing, len(PAGE_SLUGS))


if __name__ == "__main__":
    main()
