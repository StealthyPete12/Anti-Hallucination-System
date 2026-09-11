"""One-time script: convert data/raw_html/*.html (see fetch_corpus.py) into
clean plain-text corpus files under data/corpus/.

Not part of the reusable ingestion pipeline (rag/ingest.py) -- this is a
corpus-acquisition step. Deterministic given the same raw HTML inputs:
each output file is named postgres_<slug_with_underscores>.txt and contains
the page's headings, paragraphs, code blocks, and list items with
navigation/table-of-contents chrome stripped out.

Requires beautifulsoup4 (a dev dependency; see pyproject.toml).

Usage:
    python scripts/extract_corpus.py
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from bs4 import BeautifulSoup
from bs4.element import Tag

logger = logging.getLogger("scripts.extract_corpus")

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw_html"
OUT_DIR = REPO_ROOT / "data" / "corpus"

MIN_WORDS = 40


def clean_text(content: Tag) -> str:
    for cls in ("navheader", "navfooter", "toc"):
        for tag in content.find_all(class_=cls):
            tag.decompose()

    lines: list[str] = []
    for el in content.descendants:
        if not isinstance(el, Tag):
            continue
        if el.name in ("h1", "h2", "h3", "h4"):
            text = el.get_text(" ", strip=True)
            if text:
                lines.append("\n" + text.upper() + "\n")
        elif el.name == "p":
            text = el.get_text(" ", strip=True)
            if text:
                lines.append(text)
        elif el.name in ("pre", "programlisting"):
            text = el.get_text("\n", strip=True)
            if text:
                lines.append("\n```\n" + text + "\n```\n")
        elif el.name == "li":
            text = el.get_text(" ", strip=True)
            if text:
                lines.append("  - " + text)

    joined = "\n".join(lines)
    joined = re.sub(r"[ \t]+", " ", joined)
    joined = re.sub(r"\n{3,}", "\n\n", joined)
    return joined.strip() + "\n"


def extract_one(html_path: Path, out_dir: Path) -> tuple[str, int] | None:
    raw = html_path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(raw, "html.parser")
    content = soup.find("div", id="docContent")
    if content is None:
        logger.warning("Skipping %s: no docContent div found", html_path.name)
        return None

    text = clean_text(content)
    words = len(text.split())
    if words < MIN_WORDS:
        logger.warning("Skipping %s: only %d words (likely a stub/TOC page)", html_path.name, words)
        return None

    doc_id = "postgres_" + html_path.stem.replace("-", "_")
    out_path = out_dir / f"{doc_id}.txt"
    out_path.write_text(text, encoding="utf-8")
    return doc_id, words


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    html_files = sorted(RAW_DIR.glob("*.html"))
    if not html_files:
        logger.warning("No HTML files found in %s -- run scripts/fetch_corpus.py first", RAW_DIR)
        return

    total_words = 0
    written = 0
    for html_path in html_files:
        result = extract_one(html_path, OUT_DIR)
        if result is None:
            continue
        doc_id, words = result
        total_words += words
        written += 1
        logger.info("%-45s %6d words", doc_id, words)

    logger.info(
        "Wrote %d corpus files, %d words total (~%.0f 'pages' at 500 words/page)",
        written,
        total_words,
        total_words / 500,
    )


if __name__ == "__main__":
    main()
