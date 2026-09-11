# Corpus Selection

## What was selected

The corpus is a curated subset of the **official PostgreSQL 17 documentation**
(https://www.postgresql.org/docs/17/), fetched directly from
`https://www.postgresql.org/docs/17/<page>.html` for 59 individual reference
pages (a 60th and 61st candidate page, `tutorial-where` and
`backup-continuous`, returned HTTP 404 under those slugs in version 17 and
were dropped).

The pages cover:

- **Tutorial**: architecture, creating/accessing a database, tables,
  populating, `SELECT`, joins, aggregates, `UPDATE`/`DELETE`, window
  functions.
- **Data Definition (DDL)**: table basics, defaults, constraints, system
  columns, `ALTER TABLE`, schemas, inheritance, partitioning.
- **Data types**: numeric, character, date/time, boolean, enum, JSON, UUID,
  network address types.
- **Data Manipulation (DML)**: `INSERT`, `UPDATE`, `DELETE`, `RETURNING`.
- **Queries**: table expressions, select lists, `UNION`, `ORDER BY`,
  `LIMIT`, `WITH` (CTEs).
- **Indexes**: introduction, index types (B-tree, Hash, GiST, SP-GiST, GIN,
  BRIN), multicolumn, unique, expressional, partial indexes.
- **Concurrency**: MVCC introduction and caveats, transaction isolation
  levels, explicit locking (table-level, row-level, deadlocks, advisory
  locks).
- **Administration**: backup (`pg_dump`, file-system level), high
  availability/replication overview, client authentication, role attributes
  and membership, `runtime-config-connection` (`listen_addresses`, `port`,
  `max_connections`, etc.).
- **Text search and JSON functions**.

Each page was downloaded to `data/raw_html/`, then converted to plain text
with a one-time extraction script (`BeautifulSoup`-based, not part of the
reusable pipeline) that keeps the page's headings, paragraphs, code blocks,
and list items while stripping navigation chrome. The resulting files live
in `data/corpus/` as one `.txt` file per source page, named
`postgres_<page_slug_with_underscores>.txt`.

## Corpus size

- **59 documents**, **89,806 words** total, roughly **~180 "pages"** at an
  approximate 500 words/page density — within the 100-500 page target for
  this phase.
- After chunking (220-word chunks, 40-word overlap, see
  [rag/ingest.py](../rag/ingest.py)), the corpus yields **515 chunks** in
  `data/index/chunks.jsonl`.

## Why this corpus

- **Publicly redistributable.** PostgreSQL's documentation is distributed
  under the [PostgreSQL License](https://www.postgresql.org/about/licence/),
  a liberal, OSI-approved license that permits use, copying, modification,
  and redistribution (including for commercial purposes) with attribution
  and without warranty, as long as the copyright notice and permission
  notice are preserved. There is no paywall and no authentication barrier
  to accessing it.
- **Fact-dense.** Reference documentation for a mature relational database
  is dense with concrete, checkable facts: default parameter values, exact
  behaviors of specific SQL commands, precise numeric limits, and
  well-defined semantics (e.g., isolation-level guarantees, lock-mode
  conflict tables). This makes it well suited to writing verifiable
  question/answer pairs — the opposite of prose that is mostly opinion or
  narrative.
- **Right-sized.** The full PostgreSQL manual runs to thousands of pages
  (the complete PDF is enormous); this curated subset was deliberately
  bounded to ~180 "pages" so it can be read, chunked, and — critically —
  manually verified end to end for Phase 0, while still spanning enough
  distinct topics (tutorial, DDL, DML, data types, indexing, concurrency,
  administration) to support a genuinely varied golden set, including
  multi-hop questions that combine facts from different chapters.
- **Version-pinned for reproducibility.** All pages were fetched from
  `/docs/17/` (not `/docs/current/`, which silently moves to a newer major
  version over time), so re-fetching the same URLs later returns the same
  content. As of this writing, `/docs/17/` resolves to PostgreSQL 17.11
  documentation.

## Expected use cases

This corpus is the foundation for evaluating a RAG pipeline's:

- **Retrieval quality** — can the retriever find the right chunk(s) for a
  question given specific, factual, technical prose?
- **Multi-hop reasoning** — can the pipeline assemble evidence spread
  across multiple chunks/documents (e.g., combining a locking rule from
  `explicit_locking` with an MVCC caveat from `mvcc_caveats`)?
- **Hallucination resistance / grounding** — does the system correctly
  refuse to answer when the corpus genuinely doesn't cover a topic
  (`unanswerable` questions), rather than fabricating a plausible-sounding
  PostgreSQL fact?
- **Robustness to false premises** — does the system correct a user's
  incorrect assumption (e.g., "CREATE ROLE grants LOGIN by default") instead
  of either fabricating an answer that accepts the premise, or refusing
  outright when it actually has the corrected fact on hand?

## Source URLs

All pages were fetched from `https://www.postgresql.org/docs/17/<slug>.html`.
See `data/raw_html/` for the raw snapshots and
[../data/corpus/](../data/corpus/) for the extracted text. The full list of
59 slugs used is recorded in the ingestion history; representative examples
include `tutorial-start`, `ddl-constraints`, `datatype-json`, `dml-insert`,
`indexes-partial`, `mvcc-intro`, `transaction-iso`, `explicit-locking`,
`backup-dump`, `high-availability`, `role-attributes`, and
`runtime-config-connection`.

## Licensing considerations

> PostgreSQL is released under the PostgreSQL License, a liberal
> Open Source license, similar to the BSD or MIT licenses.
>
> PostgreSQL Database Management System (formerly known as Postgres,
> then as Postgres95)
>
> Portions Copyright © 1996-2025, PostgreSQL Global Development Group
>
> Portions Copyright © 1994, The Regents of the University of California
>
> Permission to use, copy, modify, and distribute this software and its
> documentation for any purpose, without fee, and without a written
> agreement is hereby granted, provided that the above copyright notice
> and this paragraph and the following two paragraphs appear in all
> copies.

Because this project redistributes verbatim excerpts of that documentation
as the evaluation corpus, this file and the corpus itself carry that
attribution forward. The corpus in `data/corpus/` is derived, machine-text
extractions of copyrighted-but-liberally-licensed PostgreSQL documentation;
it is used here strictly for non-commercial, educational RAG-evaluation
research and retains the original copyright notice above. See
https://www.postgresql.org/about/licence/ for the full license text.

## What was avoided

- No paywalled documentation (e.g., vendor manuals requiring a support
  contract).
- No scraped content that sits behind a login or authentication wall.
- No content with redistribution restrictions incompatible with an open
  research/evaluation project.
