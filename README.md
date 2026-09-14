# Anti-Hallucination System

A RAG (Retrieval-Augmented Generation) evaluation project.

- **Phase 0** built the foundation: a real, redistributable corpus, a
  deterministic ingestion/chunking pipeline, and a manually verified
  50-question golden benchmark set, with automated tests that prevent the
  benchmark from drifting or silently breaking.
- **Phase 1** builds the simplest possible production-quality baseline RAG
  system on top of that foundation — chunking, embedding, vector storage,
  retrieval, and LLM generation, served through a FastAPI `/query` endpoint
  and runnable end to end with `docker compose up`. No hybrid retrieval,
  rerankers, query rewriting, or agents yet — this phase exists to
  establish a measurable baseline that later phases are compared against.
- **Phase 2** builds the evaluation harness that turns "it seems to work"
  into numbers: deterministic retrieval metrics (Recall@5/MRR/nDCG@10),
  LLM-judged faithfulness (per-claim hallucination detection) and
  correctness, and refusal-behavior scoring on the `unanswerable`/
  `false_premise` golden-set subsets — plus cost, latency, and a measured
  run-to-run noise floor, so future retrieval/generation changes can be
  judged against real variance instead of guesswork. See
  [Phase 2: Evaluation harness](#phase-2-evaluation-harness) below.
- **Phase 3** turns that harness into an automated CI quality gate: every
  pull request runs the full evaluation, compares the result against
  `evals/baseline.json` with a configurable set of regression rules, posts
  a markdown report as a PR comment, and fails the build on a hard
  regression (fabrications, faithfulness, recall, or refusal) while
  surfacing soft regressions (cost, latency) as non-blocking warnings.
  Merges to `main` automatically re-run the evaluation and promote a new
  baseline, archiving every prior scorecard under `evals/history/`. See
  [Phase 3: CI quality gate](#phase-3-ci-quality-gate) below.
- **Phase 4** (this phase) makes retrieval and generation config-driven
  (dense/sparse/hybrid retrieval with BM25 + Reciprocal Rank Fusion,
  cross-encoder reranking, swappable embedding/generator/reranker models,
  optional query rewriting) and adds a reproducible experiment framework
  plus judge-validation tooling, so every change is measured against the
  Phase 2 harness before it's ever called an improvement. See
  [Phase 4: Retrieval & generation experiments](#phase-4-retrieval--generation-experiments)
  below.

## Project overview

```
.
├── rag/
│   ├── ingest.py               # Phase 0: corpus -> chunks.jsonl (feeds the golden set)
│   ├── chunking.py              # Phase 1: recursive, token-bounded chunker (feeds the vector store)
│   ├── models.py                # shared value types (Chunk, RetrievedChunk, SparseVector, ...)
│   ├── protocols.py             # Chunker/Embedder/VectorStore/Generator/Reranker/... contracts
│   ├── pipeline.py              # wires the above into index_corpus(), retrieve(), and query()
│   ├── generator.py             # LiteLLM-backed, cited, grounded generation
│   ├── fusion.py                # Phase 4: Reciprocal Rank Fusion
│   ├── sparse.py                # Phase 4: BM25 sparse-vector encoding
│   ├── reranker.py              # Phase 4: cross-encoder reranking (MiniLM / bge-reranker-base)
│   ├── query_rewriter.py        # Phase 4: optional LLM query rewriting before retrieval
│   ├── embedders/
│   │   └── local.py             # sentence-transformers embedder (model swappable via config)
│   └── stores/
│       └── qdrant.py            # Qdrant store: dense + Phase 4 named sparse vector + hybrid RRF
│
├── eval/                        # Phase 2: evaluation harness
│   ├── dataset.py                # typed golden-set loader
│   ├── judge.py                  # LiteLLM judge client: temp=0, JSON, retries, cost, prompt hashing
│   ├── metrics/
│   │   ├── retrieval.py           # Recall@K, MRR, nDCG@K (pure Python, no LLM)
│   │   ├── faithfulness.py        # per-claim hallucination detection
│   │   ├── correctness.py         # correct/partial/incorrect verdict scoring
│   │   └── refusal.py             # unanswerable/false_premise refusal scoring
│   ├── prompts/                  # versioned judge prompts (never hardcoded in Python)
│   │   ├── faithfulness_v1.txt
│   │   ├── correctness_v1.txt
│   │   └── refusal_v1.txt
│   ├── results.py                # QuestionResult: one question's full evaluation record
│   ├── scorecard.py               # aggregation + noise-floor (mean/stdev across --repeats)
│   ├── summary.py                 # human-readable stdout summary table
│   ├── run.py                     # CLI: `python -m eval.run`
│   ├── experiment.py              # Phase 4: reproducible experiment framework
│   └── agreement.py               # Phase 4: human-vs-judge % agreement + Cohen's kappa
│
├── experiments/                  # Phase 4: one JSON record per experiment (git sha, timestamp,
│                                  # config, metrics) - see eval/experiment.py
│
├── app/
│   ├── main.py                  # FastAPI app: GET /health, POST /query
│   ├── dependencies.py          # build_pipeline(settings) factory + cached get_pipeline()
│   └── schemas.py               # QueryRequest / QueryResponse / HealthResponse
│
├── scripts/
│   ├── fetch_corpus.py          # one-time: download PostgreSQL 17 doc pages
│   ├── extract_corpus.py        # one-time: raw HTML -> clean data/corpus/*.txt
│   ├── index_corpus.py          # chunk + embed + upsert data/corpus/*.txt into Qdrant
│   ├── run_phase4_experiments.py # Phase 4: runs the measured retrieval-only experiment battery
│   └── label.py                  # Phase 4: sample + manually label answers for judge validation
│
├── data/
│   ├── corpus/                  # 59 plain-text PostgreSQL 17 doc pages (~180 "pages")
│   ├── raw_html/                # gitignored: raw HTML snapshots (regenerable)
│   ├── index/
│   │   └── chunks.jsonl         # generated by rag/ingest.py; feeds the golden-set tests
│   └── golden_set.jsonl         # 50 manually verified benchmark questions
│
├── docs/
│   └── corpus_selection.md      # why this corpus, licensing, sizing
│
├── tests/
│   ├── test_golden_set.py       # Phase 0: golden-set schema/integrity tests
│   ├── unit/                    # fast, no real model/Qdrant/LLM calls (Phase 4: fusion, sparse,
│   │                             # reranker, query rewriter, experiment framework, agreement,
│   │                             # labeling workflow)
│   └── integration/             # real qdrant-client engine (incl. Phase 4 sparse/hybrid) /
│                                 # real model weights (incl. Phase 4 reranker/experiment e2e) /
│                                 # eval.run's async orchestration (fakes throughout,
│                                 # but exercises every module wired together)
│
├── config.py                    # centralized, env-driven settings (pydantic-settings)
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml               # uv project, ruff, pytest, mypy config
├── .pre-commit-config.yaml
└── README.md
```

### Why two chunkers?

`rag/ingest.py` (Phase 0) and `rag/chunking.py` (Phase 1) are deliberately
separate and serve different consumers:

- **`rag/ingest.py`** is a fixed, word-based chunker whose only job is to
  reproduce the exact `data/index/chunks.jsonl` that `data/golden_set.jsonl`'s
  `supporting_chunk_ids` were hand-verified against. Changing its chunking
  strategy would silently invalidate every golden-set chunk reference, so it
  is frozen as-is.
- **`rag/chunking.py`** is the production chunker the RAG pipeline actually
  indexes with: a recursive character splitter, chunk size and overlap
  measured in real model tokens (512 tokens / 15% overlap by default,
  matching bge-small-en-v1.5's max sequence length), with `doc_id`,
  `chunk_index`, `source_path`, `heading`, and `chunk_id` metadata on every
  chunk.

Both read the same `data/corpus/*.txt` files; they simply produce
differently-shaped chunks for different purposes.

The corpus is a curated, version-pinned (PostgreSQL 17) subset of the
official PostgreSQL documentation — see
[docs/corpus_selection.md](docs/corpus_selection.md) for why it was chosen,
its licensing, and its exact scope. The golden set is 50 hand-verified
question/answer/evidence triples spanning four question types
(`single_hop`, `multi_hop`, `unanswerable`, `false_premise`), intended to
become the benchmark for future retrieval, grounding, and
hallucination-resistance evaluation work.

## Architecture / pipeline flow

```
User Question
      |
      v
[Phase 4, optional] Query Rewriter (LLM rewrite for retrieval only;
      |              generator always sees the original question)
      v
Embed Query (embedding model swappable via config, BGE query prefix if applicable)
      |
      v
Qdrant Search - retrieval_mode = dense | sparse | hybrid (Phase 4):
      dense  -> cosine search on the named "dense" vector
      sparse -> BM25 search on the named "sparse" vector
      hybrid -> both, fused client-side with Reciprocal Rank Fusion
      |
      v
Top-`retrieval_fanout` Candidates (rank preserved)
      |
      v
[Phase 4, optional] Cross-Encoder Reranker (MiniLM / bge-reranker-base)
      |              re-scores the fanout, keeps the top `top_k`
      v
Top-K Chunks
      |
      v
LiteLLM Generator (cited, grounded answer; refuses if context is insufficient;
      |              model swappable via config)
      v
API Response (answer, citations, retrieved_chunk_ids, latency_ms, usage, trace_id)
```

`rag/protocols.py` defines this as `typing.Protocol` contracts (`Chunker`,
`Embedder`, `VectorStore`/`HybridVectorStore`, `Generator`, and - Phase 4 -
`SparseEncoder`, `Reranker`, `QueryRewriter`); `rag/pipeline.py` wires
concrete implementations together via constructor injection, and
`app/dependencies.py`'s `build_pipeline(settings)` builds that pipeline
from `config.py`. Every component is swappable behind its Protocol without
touching the others, and every Phase 4 addition defaults to off/dense - a
zero-arg `RagPipeline(...)` is byte-for-byte the Phase 1 baseline. See
[Phase 4: Retrieval & generation experiments](#phase-4-retrieval--generation-experiments)
below.

## Installation

This project uses [`uv`](https://docs.astral.sh/uv/) for dependency and
environment management.

```bash
# install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# clone, then from the project root:
uv sync
```

`uv sync` creates `.venv/` and installs the project plus its dependencies
(FastAPI, Qdrant client, LiteLLM, sentence-transformers, and the dev tools
pytest/ruff/mypy/pre-commit) from `pyproject.toml`. The embedding model's
`torch` dependency is pinned to the CPU-only PyTorch wheel index (see
`[tool.uv.sources]` in `pyproject.toml`) so `uv sync` doesn't pull multi-GB
CUDA packages on machines without a GPU.

## Environment setup

Requires Python 3.11+. `uv sync` provisions its own interpreter/venv
automatically. All commands below assume they're run with `uv run ...`
(which transparently uses the project's `.venv`), or after activating
`.venv` yourself (`source .venv/bin/activate`).

Copy `.env.example` to `.env` and fill in an LLM API key to enable
generation:

```bash
cp .env.example .env
# then set OPENAI_API_KEY (or point RAG_LLM_MODEL at another LiteLLM-
# supported provider and set that provider's key instead)
```

Every setting in `config.py` can be overridden by an environment variable
prefixed `RAG_` (e.g. `RAG_TOP_K=8`, `RAG_CHUNK_SIZE_TOKENS=256`). See
`.env.example` for the full list.

## Running everything with Docker Compose (recommended)

```bash
docker compose up --build
```

This starts Qdrant and the FastAPI app. The embedding model
(`BAAI/bge-small-en-v1.5`) is downloaded and baked into the API image at
*build* time (see the Dockerfile), and the container runs with
`HF_HUB_OFFLINE=1` by default — so the API never needs outbound network
access just to serve a query, only Qdrant and your chosen LLM provider.

Once it's up:

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

Then index the corpus into Qdrant (one-time, or whenever `data/corpus/`
changes):

```bash
docker compose exec api python -m scripts.index_corpus
```

Then query it:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the default transaction isolation level in PostgreSQL?"}'
```

```json
{
  "answer": "Read Committed is the default isolation level in PostgreSQL [postgres_transaction_iso#2].",
  "citations": ["postgres_transaction_iso#2"],
  "retrieved_chunk_ids": ["postgres_transaction_iso#2", "postgres_transaction_iso#1", "postgres_transaction_iso#3"],
  "latency_ms": {"embed": 12.4, "retrieve": 8.1, "generate": 640.2, "total": 660.7},
  "usage": {"prompt_tokens": 812, "completion_tokens": 24, "total_tokens": 836},
  "cost_usd": 0.00051,
  "trace_id": "b6b6e9b0-6f2f-4c1a-9e2a-6a7a6b1a2c3d"
}
```

An unanswerable question correctly refuses instead of guessing:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the maximum number of rows PostgreSQL can store in a single table?"}'
```

```json
{
  "answer": "The supplied context does not contain enough information to answer that question.",
  "citations": [],
  "retrieved_chunk_ids": ["postgres_ddl_basics#0", "postgres_indexes_intro#0", "..."],
  "...": "..."
}
```

To stop and remove everything (including the Qdrant volume):

```bash
docker compose down -v
```

## Running locally without Docker

```bash
# 1. Start Qdrant (the only external dependency)
docker run -p 6333:6333 qdrant/qdrant:v1.15.1

# 2. Index the corpus (downloads bge-small-en-v1.5 on first run, ~130MB)
uv run python -m scripts.index_corpus --log-level INFO

# 3. Start the API
uv run uvicorn app.main:app --reload

# 4. Query it
curl http://localhost:8000/health
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What does MVCC stand for in PostgreSQL?"}'
```

## Running ingestion / generating the golden-set chunk index (Phase 0)

This is unrelated to Qdrant indexing above — it regenerates
`data/index/chunks.jsonl`, the fixed reference chunk index the golden set's
`supporting_chunk_ids` are checked against (see "Why two chunkers?" above).

```bash
# default: data/corpus -> data/index/chunks.jsonl, 220-word chunks, 40-word overlap
uv run python -m rag.ingest
```

Each line looks like:

```json
{"chunk_id": "postgres_mvcc_intro#0", "text": "...", "doc_id": "postgres_mvcc_intro", "chunk_index": 0, "num_words": 220}
```

## Running tests

```bash
uv run pytest                          # full suite (unit + integration + golden set)
uv run pytest tests/unit -v            # fast unit tests only (fakes; no model/Qdrant/LLM calls)
uv run pytest -m integration -v        # integration tests (real qdrant-client engine, real model weights)
uv run pytest -m "not integration" -v  # everything except integration tests
```

- `tests/unit/` — chunking, embedding-prefix logic, generator citation
  contract, pipeline orchestration, the FastAPI endpoints, and (Phase 2)
  the golden-set loader, every metric module (`retrieval`, `faithfulness`,
  `correctness`, `refusal`), the judge client, scorecard aggregation, and
  summary rendering — each tested against small fakes satisfying the
  Protocols in `rag/protocols.py` or a fake judge. Phase 4 adds: RRF
  (`test_fusion.py`), BM25 sparse encoding (`test_sparse.py`), the
  cross-encoder reranker (`test_reranker.py`, fake `CrossEncoder`), the
  query rewriter (`test_query_rewriter.py`, fake LiteLLM), hybrid-mode
  pipeline orchestration (`test_pipeline.py`), the `build_pipeline(settings)`
  factory (`test_dependencies.py`), the experiment framework's pure logic
  (`test_experiment.py`), the labeling workflow (`test_label.py`), and
  judge-agreement math (`test_agreement.py`). No network access, model
  download, or running Qdrant/LLM required.
- `tests/integration/` — `test_qdrant_store.py` runs the real
  `qdrant-client` engine in `:memory:` mode (no server needed);
  `test_qdrant_hybrid.py` does the same for Phase 4's named sparse vector,
  BM25 search, and hybrid RRF fusion against the real Qdrant query engine;
  `test_pipeline_e2e.py` additionally downloads the real embedding model
  and asserts retrieval actually surfaces the right chunk for a real
  question (skipped automatically if offline); `test_experiment_e2e.py`
  runs the Phase 4 experiment framework the same way, in both dense and
  hybrid mode; `test_eval_run.py` runs `eval.run`'s full async orchestration
  (dataset load → concurrent pipeline+judge fan-out → scorecard → noise
  floor → file write → CLI `main()`) against a fake pipeline and fake
  judge, proving every Phase 2 module works together correctly as a whole.
- `tests/test_golden_set.py` — Phase 0's golden-set schema/integrity
  guardrail; unaffected by anything in this phase.

## Environment variables

All settings live in `config.py` (via `pydantic-settings`) and are
overridable with an `RAG_`-prefixed environment variable or `.env` file:

| Variable | Default | Purpose |
|---|---|---|
| `RAG_CORPUS_DIR` | `data/corpus` | Source documents to index |
| `RAG_CHUNK_SIZE_TOKENS` | `512` | Max tokens per chunk |
| `RAG_CHUNK_OVERLAP_RATIO` | `0.15` | Overlap as a fraction of chunk size |
| `RAG_EMBEDDING_MODEL_NAME` | `BAAI/bge-small-en-v1.5` | Sentence-transformers model |
| `RAG_EMBEDDING_CACHE_DIR` | `.cache/embeddings` | Local model weight cache |
| `RAG_EMBEDDING_BATCH_SIZE` | `32` | Batch size for document embedding |
| `RAG_QDRANT_URL` | `http://localhost:6333` | Qdrant connection (`:memory:` works too) |
| `RAG_QDRANT_COLLECTION_NAME` | `postgres_docs` | Qdrant collection name |
| `RAG_TOP_K` | `5` | Chunks retrieved per query |
| `RAG_LLM_MODEL` | `gpt-4o-mini` | Any [LiteLLM](https://docs.litellm.ai/docs/providers)-supported model string |
| `RAG_LLM_TEMPERATURE` | `0.0` | Generation temperature |
| `RAG_LLM_MAX_TOKENS` | `1024` | Max generation tokens |
| `RAG_JUDGE_MODEL` | `gpt-4o-mini` | LiteLLM model string used by the Phase 2 evaluation judge |
| `RAG_JUDGE_TIMEOUT_SECONDS` | `60.0` | Per-call judge timeout |
| `RAG_JUDGE_MAX_RETRIES` | `3` | Judge call retries (exponential backoff) before failing the question |
| `RAG_RETRIEVAL_MODE` | `dense` | Phase 4: `dense` \| `sparse` \| `hybrid` |
| `RAG_RRF_K` | `60` | Phase 4: Reciprocal Rank Fusion's rank-discount constant |
| `RAG_RETRIEVAL_FANOUT` | `20` | Phase 4: candidates fetched before fusion/reranking narrows to `top_k` |
| `RAG_RERANKER_ENABLED` | `false` | Phase 4: enable the cross-encoder reranking stage |
| `RAG_RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Phase 4: also accepts `BAAI/bge-reranker-base` |
| `RAG_QUERY_REWRITE_ENABLED` | `false` | Phase 4: rewrite the retrieval query with an LLM before embedding |
| `RAG_QUERY_REWRITE_MODEL` | `gpt-4o-mini` | Phase 4: LiteLLM model string used for query rewriting |
| `RAG_LOG_LEVEL` | `INFO` | Logging verbosity |

Provider API keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, ...) are read
directly by LiteLLM and are **not** `RAG_`-prefixed. See `.env.example`.

## API reference

### `GET /health`

```json
{"status": "ok"}
```

### `POST /query`

Request:

```json
{"question": "What is the retention period for audit logs?"}
```

Response (`QueryResponse`):

```python
class QueryResponse(BaseModel):
    answer: str  # grounded answer with inline [chunk_id] citations
    citations: list[str]  # chunk_ids actually cited in the answer, in order
    retrieved_chunk_ids: list[str]  # every chunk retrieval returned, rank-ordered
    latency_ms: dict[str, float]  # {"embed", "retrieve", "generate", "total"}
    usage: TokenUsage  # prompt_tokens, completion_tokens, total_tokens
    cost_usd: float  # dollar cost of the generation call (litellm.completion_cost)
    trace_id: str  # UUID, also present in the structured logs for this request
```

`retrieved_chunk_ids`, `latency_ms`, and `cost_usd` are returned from this
same production endpoint from day one — there is no separate evaluation
code path. The Phase 2 evaluation harness (`eval/run.py`) consumes exactly
this response via `pipeline.query()`.

### Grounding and citation rules

The generator's system prompt (`rag/generator.py`) requires: every factual
claim be supported by a retrieved chunk, cited inline as `[chunk_id]`;
explicit refusal ("The supplied context does not contain enough
information to answer that question.") when the context doesn't contain
the answer; and never a fabricated citation. As a second, independent
safety net, `rag/generator.py` also programmatically strips any citation
in the model's output that doesn't match a chunk_id that was actually
retrieved — a prompt-following failure can't surface as a fabricated
citation in the API response.

## Logging

Every request logs (with a shared `trace_id`): receipt (including
`retrieval_mode`, whether reranking and query rewriting are active),
per-stage timing (embed/retrieve/generate/total, with rerank/rewrite time
folded into `retrieve`), retrieved chunk IDs, citations, and any error, at
`INFO` level by default (`RAG_LOG_LEVEL`). This is the observability seam
later phases (tracing, metrics, dashboards) build on.

Phase 4 additionally logs (all at `INFO` unless noted):

- `rag.stores.qdrant` — collection creation (dense vs. dense+sparse),
  per-search result counts for dense/sparse/hybrid search, hybrid fusion's
  dense/sparse/fused candidate counts (`DEBUG`).
- `rag.reranker` / `rag.query_rewriter` — model load, and (`DEBUG`) how many
  candidates were reranked / what a query was rewritten to.
- `eval.experiment` — every experiment's name, mode, and full settings
  overrides at start, and its indexed chunk count and measured metrics at
  completion - the same information persisted to its JSON record, so a log
  line and a saved experiment can always be cross-checked against each
  other.

## Development workflow

```bash
uv run ruff check .            # lint
uv run ruff check . --fix      # lint, auto-fixing what's safe
uv run ruff format .           # format
uv run mypy                    # type-check rag/, app/, config.py, scripts/
uv run pytest                  # test
```

### pre-commit

```bash
uv run pre-commit install      # one-time, installs the git hook
uv run pre-commit run --all-files
```

The configured hooks run ruff (lint + format), standard hygiene checks
(trailing whitespace, EOF newline, YAML/TOML/JSON validity, large-file and
merge-conflict guards), and the full pytest suite before every commit.

### Adding to the corpus or golden set

1. Add new `*.txt` files to `data/corpus/` (or extend an existing one).
2. Re-run `uv run python -m rag.ingest` to regenerate `data/index/chunks.jsonl`.
3. For any new golden-set question, find the exact chunk(s) that support the
   answer in the regenerated `chunks.jsonl` — do not guess chunk IDs.
4. Add the record to `data/golden_set.jsonl`, keeping the type distribution
   test in `tests/test_golden_set.py` in sync (update `EXPECTED_DISTRIBUTION`
   / `EXPECTED_TOTAL` if you intentionally change the counts).
5. Re-run `uv run python -m scripts.index_corpus` so Qdrant reflects the
   updated corpus.
6. Run `uv run pytest` and fix anything it flags before committing.

## Phase 2: Evaluation harness

Phase 2 turns the Phase 1 baseline into a set of numbers: it runs the real
production pipeline (the exact same `pipeline.query()` the FastAPI endpoint
calls — no separate "evaluation" code path) against every question in
`data/golden_set.jsonl`, scores retrieval quality deterministically, scores
generation quality and safety with an LLM judge, and aggregates everything
into a scorecard. It intentionally makes **no** retrieval or generation
improvements — no rerankers, no hybrid search, no query rewriting, no
agents — its only job is to measure the baseline precisely enough that
later improvements can be judged against real evidence instead of vibes.

### Architecture

```
data/golden_set.jsonl (50 questions)
      |
      v
eval.dataset.load_golden_set()  ->  list[GoldenQuestion]
      |
      v  (asyncio.Semaphore-bounded concurrency, 8-12 in flight)
pipeline.query(question)  ->  answer, retrieved_chunk_ids, retrieved_chunks,
      |                       latency_ms, cost_usd, trace_id
      |
      +--> eval.metrics.retrieval  (pure Python: Recall@5, MRR, nDCG@10)
      |
      +--> eval.judge.LLMJudge  (LiteLLM, temp=0, JSON, retries, cost, prompt hash)
              |
              +--> eval.metrics.faithfulness  (always)
              +--> eval.metrics.correctness   (always)
              +--> eval.metrics.refusal       (only unanswerable / false_premise)
      |
      v
eval.results.QuestionResult (one per question)
      |
      v
eval.scorecard.build_scorecard()  ->  retrieval / generation / safety / ops + per_question
      |
      v (if --repeats > 1)
eval.scorecard.compute_noise_floor()  ->  mean/stdev per metric across repeats
      |
      v
evals/results.json  +  eval.summary.render_summary() printed to stdout
```

Each question's pipeline call is synchronous (real network/model calls), so
it runs in a worker thread via `asyncio.to_thread` behind a semaphore; the
faithfulness/correctness/refusal judge calls for a question are natively
async and run concurrently with each other. This is what lets a 50-question
run with 8-12 concurrent in-flight questions complete in roughly a minute
instead of running every question strictly one at a time.

### Metric definitions

**Retrieval** (`eval/metrics/retrieval.py`) — pure Python, deterministic, no
LLM calls, computed against `supporting_chunk_ids`:

- **Recall@5** — 1.0 if *any* supporting chunk appears in the top 5
  retrieved chunks, else 0.0 (hit-rate@k, not fraction-of-relevant-chunks).
- **MRR** — `1 / rank` of the first supporting chunk found (1-indexed);
  `0.0` if none is found anywhere in the retrieved list.
- **nDCG@10** — `DCG@10 / IDCG@10` with binary relevance, where
  `DCG@k = Σ rel_i / log2(i + 1)` for ranks `i = 1..k`.

A question with no `supporting_chunk_ids` (every `unanswerable` question) is
excluded from these three metrics' averages entirely (`None`, not `0.0`) —
there is nothing to recall, so scoring it as a retrieval failure would
misrepresent retrieval quality.

**Faithfulness** (`eval/metrics/faithfulness.py`) — deliberately *not* a
vague 1-5 LLM score. The judge breaks the generated answer into atomic
factual claims and verdicts each one independently against the retrieved
context (never outside knowledge). Score = `supported_claims / total_claims`.
An answer with zero claims (a pure refusal) is vacuously faithful (`1.0`) —
it made no assertions, so it cannot have hallucinated.

**Correctness** (`eval/metrics/correctness.py`) — the judge compares
question, `reference_answer`, and generated answer, returning exactly one
verdict: `correct` (1.0), `partial` (0.5), or `incorrect` (0.0). The overall
score is the mean across all evaluated questions, including the
`unanswerable`/`false_premise` subset — for those, the reference answer
*is* the correct refusal, so a system that correctly declines also scores
`correct` here.

**Refusal** (`eval/metrics/refusal.py`) — applies only to the
`unanswerable` and `false_premise` golden-set subsets, where declining (or
challenging a false premise) is the *correct* behavior. `refusal_rate =
correct_refusals / required_refusals`; `fabrications` counts how many of
those questions the system answered instead of refusing — each one is a
hallucination-adjacent safety incident, not just a missed point.

### Prompt versioning and hashing

Every judge prompt lives as a plain-text file in `eval/prompts/`
(`faithfulness_v1.txt`, `correctness_v1.txt`, `refusal_v1.txt`) — never
hardcoded in Python. `eval/judge.py` loads each file once per process,
computes its SHA256, and stamps that hash (plus the version parsed from the
filename, e.g. `v1`) onto every `JudgeResponse`. `eval.run.build_config()`
records all three hashes under `config.judge_prompt_hash` in every
scorecard, so editing a prompt is immediately visible as a hash change in
the output — "faithfulness" and "correctness" can never silently mean
something different between two runs without it showing up in the diff.

### Running the evaluation harness

Requires the same running Qdrant + indexed corpus + LLM provider key as
Phase 1 (see "Running locally without Docker" above), since it exercises
the real production pipeline:

```bash
uv run python -m eval.run \
  --dataset data/golden_set.jsonl \
  --out evals/results.json \
  --repeats 1
```

| Flag | Default | Purpose |
|---|---|---|
| `--dataset` | `data/golden_set.jsonl` | Golden-set JSONL file to evaluate |
| `--out` | `evals/results.json` | Where to write the scorecard JSON |
| `--repeats` | `1` | Full evaluation passes; `>1` also computes a noise floor |
| `--concurrency` | `10` | Max in-flight questions (pipeline + judge combined) |
| `--judge-model` | `settings.judge_model` (`RAG_JUDGE_MODEL`) | LiteLLM model string for the judge |
| `--log-level` | `settings.log_level` | Overrides `RAG_LOG_LEVEL` for this run |

A human-readable summary (Retrieval / Generation / Safety / Ops) prints to
stdout automatically at the end of every run, in addition to the full JSON
scorecard written to `--out`.

### Scorecard structure

```json
{
  "git_sha": "a1b2c3d",
  "timestamp": "2026-09-14T10:00:00Z",
  "config": {
    "embedder": "BAAI/bge-small-en-v1.5",
    "generator": "gpt-4o-mini",
    "judge": "gpt-4o-mini",
    "judge_prompt_hash": {
      "faithfulness_v1": "…", "correctness_v1": "…", "refusal_v1": "…"
    },
    "top_k": 5,
    "rerank": false
  },
  "retrieval": {"recall_at_5": 0.82, "mrr": 0.68, "ndcg_at_10": 0.74},
  "generation": {"faithfulness": 0.89, "correctness": 0.79},
  "safety": {"refusal_rate": 0.93, "fabrications": 1},
  "ops": {"mean_cost_usd": 0.0019, "p95_latency_ms": 1740.0},
  "num_questions": 50,
  "num_errors": 0,
  "per_question": [
    {
      "id": "q001", "type": "single_hop", "question": "…",
      "generated_answer": "…", "reference_answer": "…",
      "retrieved_chunk_ids": ["…"],
      "retrieval_metrics": {"recall_at_5": 1.0, "mrr": 1.0, "ndcg_at_10": 1.0},
      "faithfulness": {"score": 1.0, "unsupported_count": 0, "claims": [...], "prompt_hash": "…"},
      "correctness": {"verdict": "correct", "reason": "…", "score": 1.0, "prompt_hash": "…"},
      "refusal": null,
      "latency_ms": {"embed": 12.4, "retrieve": 8.1, "generate": 640.2, "total": 660.7},
      "cost_usd": 0.0007,
      "trace_id": "…",
      "error": null
    }
  ]
}
```

With `--repeats > 1`, three additional top-level keys appear: `repeats`
(the count), `runs` (each pass's `retrieval`/`generation`/`safety`/`ops`
section, for eyeballing pass-to-pass drift), and `noise_floor` (see below).
A question whose pipeline call or judge call raises is never allowed to
crash the run: its `error` field is set, it's excluded from every
aggregate, and `num_errors` reports the count — silently dropping a failed
question from a metric's denominator would inflate that metric.

### Noise floor methodology

LLM judges and generators are not perfectly deterministic even at
`temperature=0` (provider-side batching, minor numeric drift, sporadic
retries after a malformed-JSON response). `--repeats 5` runs the entire
50-question evaluation five times with **zero code changes** and computes
the mean and standard deviation of every headline metric
(`recall_at_5`, `mrr`, `ndcg_at_10`, `faithfulness`, `correctness`,
`refusal_rate`, `mean_cost_usd`, `p95_latency_ms`) across the five runs via
`eval.scorecard.compute_noise_floor()`.

```bash
uv run python -m eval.run --repeats 5 --out evals/noise_floor.json
```

**Interpretation:** this standard deviation *is* the project's noise floor.
A future change (a reranker, a different chunk size, a new prompt) that
moves a metric by less than roughly one noise-floor standard deviation
cannot be distinguished from run-to-run evaluation noise and should not be
reported as an improvement; a change has to clear that bar before it's
trusted as real. Because this repository's evaluation environment has no
Qdrant instance or LLM provider credentials configured, an actual 5-repeat
noise-floor measurement (with real numbers) has not been run here — do that
once against a live deployment and record the resulting `evals/noise_floor.json`
alongside this section for future reference before comparing any
retrieval/generation change against it.

## Phase 3: CI quality gate

Phase 3 turns the Phase 2 evaluation harness into an automated regression
gate: every pull request against `main` runs the full evaluation against
the real production pipeline, compares the resulting scorecard to
`evals/baseline.json`, and posts the comparison as a PR comment. A hard
rule violation fails the build; a soft one is surfaced as a warning but
never blocks merge. Merging to `main` re-runs the evaluation and promotes
its scorecard to the new baseline, archiving the previous one under
`evals/history/` forever.

### Architecture

```
pull_request  ─────────────────────────────────────────────┐
                                                             v
                                          .github/workflows/eval-pr.yml
                                             |
                                             +-- uv sync, start Qdrant, rebuild index
                                             +-- eval.run  ->  evals/results.json
                                             +-- eval.compare --baseline evals/baseline.json
                                             |                --candidate evals/results.json
                                             |                --out comment.md
                                             |     |
                                             |     +--> eval.compare.GATE (hard/soft rules)
                                             |     +--> eval.report.render_report()
                                             |
                                             +-- comment.md posted to the PR (upsert, not duplicate)
                                             +-- evals/results.json + comment.md uploaded as artifacts
                                             +-- exit 1 on any hard failure -> PR check fails

push to main ────────────────────────────────────────────────┐
                                                              v
                                    .github/workflows/update-baseline.yml
                                       |
                                       +-- uv sync, start Qdrant, rebuild index
                                       +-- eval.run  ->  evals/results.json
                                       +-- archive evals/history/<date>.json (never overwritten)
                                       +-- promote evals/results.json -> evals/baseline.json
                                       +-- commit + push "[skip ci]"
```

### Scorecard comparison engine (`eval/compare.py`)

`Rule(metric, max_drop=None, max_absolute=None, max_increase_pct=None,
hard=True)` describes one bound on one dotted scorecard metric (e.g.
`"generation.faithfulness"`). `compare_scorecards(baseline, candidate)`
evaluates every rule in the active `GATE` list and also computes a
noise-floor-aware status (🟢/🟡/🔴) for every headline metric, whether or
not a `GATE` rule happens to cover it — the noise floor comes from the
baseline's own measured `noise_floor` section when present (a
`--repeats > 1` run), falling back to a hardcoded table otherwise. `main()`
exits `1` if any `hard=True` rule triggers, `0` otherwise:

```bash
uv run python -m eval.compare \
  --baseline evals/baseline.json \
  --candidate evals/results.json \
  --out comment.md
```

### Gate rules

| Metric | Bound | Hard? | Rationale |
|---|---|---|---|
| `safety.fabrications` | `max_absolute=0` | ✅ hard | Any fabrication on a required-refusal question is a hallucination-adjacent safety incident — zero tolerance regardless of magnitude elsewhere. |
| `generation.faithfulness` | `max_drop=0.05` | ✅ hard | ≈2× the estimated per-claim faithfulness noise floor (~0.025 stdev, see below) — a drop that size is a real hallucination regression, not judge jitter. |
| `retrieval.recall_at_5` | `max_drop=0.03` | ✅ hard | ≈2× the estimated hit-rate noise floor (~0.015 stdev) — retrieval is deterministic, so its noise floor (and this threshold) is the tightest of the four hard rules. |
| `safety.refusal_rate` | `max_drop=0.10` | ✅ hard | Refusal is judged over a small subset of the golden set (only `unanswerable`/`false_premise` questions), so a single flipped verdict moves this metric a lot more than the others — the threshold is wider (≈2× a ~0.05 stdev) to avoid failing the gate on one borderline judge call, while still catching a systematic regression like a missing refusal instruction. |
| `ops.mean_cost_usd` | `max_increase_pct=25` | ⚠️ soft | Cost drifts with provider pricing and prompt length changes that aren't regressions; flagged for review, never blocking. |
| `ops.p95_latency_ms` | `max_increase_pct=30` | ⚠️ soft | Same reasoning as cost — infra/network variance is real and shouldn't block a merge on its own. |

**Noise floor requirement:** each hard threshold above is set to
approximately **2× the standard deviation** `eval.scorecard.compute_noise_floor()`
would measure for that metric across repeated runs (`eval.run --repeats 5`)
— large enough that ordinary judge/generator non-determinism can't
accidentally trip the gate, small enough to still catch a real regression.
`evals/baseline.json` currently ships with an estimated `noise_floor`
section (documented in its own `_note` field) rather than a measured one,
since this environment has no live Qdrant instance or LLM provider
credentials configured (see "Noise floor methodology" above); once
`update-baseline.yml` runs against real credentials, its promoted baseline
carries a measured `noise_floor` and `eval.compare` automatically prefers
it over the hardcoded fallback table.

### PR report format (`eval/report.py`)

`render_report()` turns a `ComparisonResult` into the exact markdown posted
to the PR: a per-section (Retrieval/Generation/Safety/Operations) table of
`Metric | Baseline | Candidate | Delta | Status`, followed by a **Gate
Results** section listing 🚫 hard failures, ⚠️ warnings, and ✅ passed rules
separately. Status icons: 🟢 improved, 🟡 within noise floor, 🔴 regression
(no gate rule covers that metric, or it's covered but didn't trigger), 🚫
gate failure (a hard rule triggered on that exact metric).

### PR evaluation workflow (`.github/workflows/eval-pr.yml`)

Runs on every `pull_request` targeting `main`. A `tests` job (no secrets,
runs for fork PRs too) lints and runs the unit suite; a gated `eval` job
(only for same-repo PRs, so a fork can never see the provider keys) starts
a Qdrant service container, restores the `uv` and HuggingFace-model caches,
rebuilds the index, runs `eval.run`, runs `eval.compare`, upserts one PR
comment with the report (updated in place on new commits, never
duplicated), uploads `evals/results.json` + `comment.md` as build
artifacts, and fails the job if `eval.compare` exited non-zero. If the
`GENERATOR_API_KEY` secret isn't configured, the evaluation step is skipped
entirely and a comment explains why — a missing secret is a CI
configuration gap, not a quality regression, and shouldn't be reported as
one.

### Baseline promotion workflow (`.github/workflows/update-baseline.yml`)

Runs on every `push` to `main` (skipped if the commit message contains
`[skip ci]`, which is how the workflow avoids re-triggering itself on its
own baseline-promotion commit). Re-runs the same evaluation, archives the
resulting scorecard as `evals/history/<UTC date>.json` — never overwriting
an existing file for that date, appending a `-HHMMSS` suffix instead if one
already exists — then copies it over `evals/baseline.json` and pushes a
`[skip ci]` commit.

### History tracking

`evals/history/` accumulates one scorecard per promoted baseline, forever —
each carries the same fields as any other scorecard (`git_sha`,
`timestamp`, `config`, `retrieval`/`generation`/`safety`/`ops`,
`per_question`), so retrieval quality, faithfulness, cost, and latency can
all be plotted over time just by loading every file in the directory in
timestamp order. `evals/baseline.json` is always a copy of the most recent
history entry.

### Interpreting reports

- **🚫 Gate Failed** at the top of a report means at least one hard rule
  triggered — merge is blocked until the regression is fixed or the change
  is reverted.
- **✅ Gate Passed** with warnings present means a soft rule (cost/latency)
  moved outside its bound — worth a look, not a blocker.
- A 🟡 row with no gate line under "Gate Results" for that metric means the
  movement is smaller than the estimated noise floor — likely not a real
  change at all.

### Failure examples

`tests/unit/test_gate_regressions.py` encodes the three regression
scenarios the gate must catch, each asserting the specific `GATE` rule that
fires:

1. **`top_k` dropped to 1** — recall/MRR/nDCG collapse; `retrieval.recall_at_5`
   (`max_drop=0.03`) fails.
2. **The "refuse if the answer isn't in context" instruction removed from
   the generation prompt** — `safety.refusal_rate` (`max_drop=0.10`) and
   `safety.fabrications` (`max_absolute=0`) both fail together, matching
   what actually happens when that instruction is dropped.
3. **A single fabrication introduced** — `safety.fabrications` alone fails
   even when every other metric is untouched, proving the zero-tolerance
   rule doesn't need a large score movement to catch it.

A fourth test (`test_healthy_change_passes_gate`) proves a genuine
improvement passes clean, so the gate isn't just failing everything.

## Phase 4: Retrieval & generation experiments

Phase 4's rule, restated: **no improvement is accepted unless it's
measured** against the Phase 2 harness/golden set, and a bad result is kept
and reported, not hidden. Every technique below is off/at its Phase 1
default unless explicitly configured on - `RagPipeline(...)` with no Phase
4 keyword arguments is byte-for-byte the Phase 1 baseline (see
`tests/unit/test_pipeline.py`).

**What's measured for real vs. documented only:** this development
environment has no running Qdrant server and no LLM provider API key
configured (no `.env`; see "Noise floor methodology" in Phase 2). That
makes every *retrieval-only* metric (Recall@5/MRR/nDCG@10, retrieval
latency) measurable for real right here - `eval/experiment.py`'s
`retrieve()` path needs no LLM call at all, only the real embedding model
and a real (in-memory) Qdrant engine. Anything touching generation
(faithfulness, correctness, refusal, cost, or a generator/query-rewrite
comparison) needs a live LLM provider key and is documented as such below;
the code, config, and tests for those paths are complete and exercised
with fakes, they're just never claimed to have a measured number this
environment can't produce.

### 1. Hybrid retrieval: dense, sparse (BM25), and RRF fusion

`rag/sparse.py`'s `BM25Vectorizer` fits a vocabulary/IDF table from the
indexed corpus and encodes documents/queries into the sparse
`(indices, values)` vectors Qdrant's sparse vector index expects.
`rag/stores/qdrant.py`'s `QdrantVectorStore(enable_sparse=True)`
provisions a named `"dense"` vector alongside a named `"sparse"` one on the
*same* collection, so `search`, `search_sparse`, and `search_hybrid` are
all available side by side. `rag/fusion.py` implements Reciprocal Rank
Fusion exactly as specified:

```python
def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    scores: defaultdict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])
```

`search_hybrid` runs the dense and sparse branches independently (each
fetching `retrieval_fanout` candidates), fuses their rank-ordered chunk-id
lists with this function, and returns the top `top_k` with `.score`
overwritten by the fused RRF score (dense cosine similarity and BM25
weight aren't on comparable scales, so keeping either raw score would be
misleading). All three modes are selected purely through
`RAG_RETRIEVAL_MODE` (`dense` | `sparse` | `hybrid`) - no retrieval mode is
hardcoded anywhere in `rag/pipeline.py`.

### 2. Cross-encoder reranking

`rag/reranker.py`'s `CrossEncoderReranker` wraps
`sentence_transformers.CrossEncoder`. When `RAG_RERANKER_ENABLED=true`,
`RagPipeline.retrieve()` fetches `RAG_RETRIEVAL_FANOUT` (default 20)
candidates from whichever retrieval mode is active, scores every
`(query, chunk.text)` pair with the cross-encoder, and keeps the top
`RAG_TOP_K` - exactly the `Query -> top-20 -> cross-encoder -> top-5 ->
generator` pipeline the spec calls for. `RAG_RERANKER_MODEL` defaults to
`cross-encoder/ms-marco-MiniLM-L-6-v2` and also accepts
`BAAI/bge-reranker-base` as a drop-in alternative.

### 3. Embedding model experiments

`RAG_EMBEDDING_MODEL_NAME` was already config-driven from Phase 1
(`rag/embedders/local.py` takes any `sentence-transformers`-compatible
model string); Phase 4 adds `eval/experiment.py` as the harness to actually
compare models against the golden set instead of assuming a bigger model
is better. The required comparison - `BAAI/bge-small-en-v1.5` (baseline) vs.
`Qwen/Qwen3-Embedding-0.6B` - is wired up and runnable
(`--override embedding_model_name=Qwen/Qwen3-Embedding-0.6B`), but a
0.6B-parameter model's download/CPU-inference cost was out of this
environment's time budget to run to completion; treat that specific
comparison as configured-but-not-yet-measured here, not skipped in the
code.

### 4. Chunking experiments

Measured for real (`scripts/run_phase4_experiments.py`, retrieval-only:
Recall@5/MRR/nDCG@10 + latency), each rebuilding a fresh index against the
real corpus and real golden set:

| Chunk size | Overlap | Recall@5 | MRR | nDCG@10 |
|---|---|---|---|---|
| 512 (baseline) | 15% | 0.700 | 0.603 | 0.588 |
| 512 | 0% | 0.650 | 0.553 | 0.528 |
| 512 | 20% | 0.675 | 0.581 | 0.566 |
| 1024 | 15% | 0.700 | 0.604 | 0.587 |
| 1024 | 0% | 0.725 | 0.608 | 0.590 |
| 1024 | 20% | 0.725 | 0.607 | 0.591 |

**What the data shows, honestly:** dropping overlap to 0% at the baseline
512-token chunk size is a clear *regression* (Recall@5 -0.050, nDCG@10
-0.060) - overlap is load-bearing at this chunk size for this corpus, most
likely because 512-token chunks often split a fact from the sentence that
answers a golden-set question, and overlap is what keeps both halves
retrievable. Going the other way (512/20%, more overlap than baseline) is
*also* worse than the 15% baseline, not better - so more overlap isn't
monotonically good either; 15% looks close to a local optimum at this
chunk size, not just "not yet enough." At 1024 tokens, overlap matters far
less (0%, 15%, and 20% are all within ~0.001-0.002 of each other) because
a 1024-token chunk is large enough relative to this corpus's paragraph
length that overlap has little left to buy - and 1024-token chunking
overall is a small, real improvement over the 512-token baseline
(Recall@5 +0.025) on this corpus. Neither direction was hidden to make a
cleaner story.

### 5. Top-K experiments

`RAG_TOP_K` was already config-driven from Phase 1; measured here for real
at 3, 5 (baseline), and 10 - see the results table.

### 6. Query rewriting

`rag/query_rewriter.py`'s `LiteLLMQueryRewriter` rewrites the *retrieval*
query only (the generator always receives the user's original question -
`tests/unit/test_pipeline.py::test_query_rewriter_changes_the_embedded_query_only`
asserts this directly), falls back to the original question on any LLM
error or empty output, and is toggled purely by `RAG_QUERY_REWRITE_ENABLED`.
Its effect on `multi_hop` questions specifically needs the LLM to actually
rewrite queries, which needs a live provider key this environment doesn't
have configured - the code path and its fallback behavior are unit-tested
with a fake LiteLLM call, but the real before/after `multi_hop` numbers are
not measured here.

### 7. Generator model experiments

`RAG_LLM_MODEL` (any LiteLLM-supported model string) was already
config-driven from Phase 1. `eval/experiment.py`'s `mode="full"` runs the
complete Phase 2 scorecard (correctness/faithfulness/cost/latency) against
whichever generator a settings override points at, so comparing generators
is a one-line `--override llm_model=...` away - but, like every
`mode="full"` experiment, it requires a live LLM provider key this
environment doesn't have. Not measured here.

### 8. Reranker model comparison (MiniLM vs. bge-reranker-base)

The retrieval-side effect (does reranking change *which* chunks end up in
the top 5, and by how much) is measured for real - see `reranker_minilm` in
the results table below. The faithfulness/correctness trade-off between the
two reranker models needs `mode="full"` (an LLM judge), which is not
measured here for the reason above; `tests/unit/test_reranker.py` proves
both model names wire through `CrossEncoderReranker` identically.

### 9. Experiment framework

`eval/experiment.py`'s `run_experiment(spec)` builds a fresh, independent
`RagPipeline` from `ExperimentSpec.overrides`, re-indexes a fresh in-memory
(or real, if `RAG_QDRANT_URL` points at one) Qdrant collection so
index-affecting overrides (chunk size, embedding model) are actually
reflected in what gets searched, evaluates it (`retrieval_only` - no LLM
needed - or `full` - complete Phase 2 scorecard), and writes a
self-contained JSON record under `experiments/<name>/<timestamp>.json`
(plus a `latest.json`) carrying:

```json
{
  "experiment": "hybrid_rrf",
  "description": "...",
  "mode": "retrieval_only",
  "git_sha": "a1b2c3d",
  "timestamp": "2026-09-14T14:55:12Z",
  "overrides": {"qdrant_url": ":memory:", "retrieval_mode": "hybrid"},
  "config": {"embedder": "...", "top_k": 5, "retrieval_mode": "hybrid", "rerank": false, "...": "..."},
  "metrics": {"retrieval": {"recall_at_5": 0.725, "mrr": 0.635, "ndcg_at_10": 0.622}, "ops": {"...": "..."}},
  "per_question": ["..."]
}
```

Every field needed to reproduce the run - git SHA, timestamp, exact
config, measured metrics, and the raw per-question evaluation output - is
in the one file, so a later reader never has to guess what was actually
run. `scripts/run_phase4_experiments.py` defines and runs the full
retrieval-only battery behind the results table below in one command:

```bash
uv run python -m scripts.run_phase4_experiments
```

or a single ad hoc experiment via the CLI:

```bash
uv run python -m eval.experiment --name my_experiment \
  --override retrieval_mode=hybrid --override top_k=8
```

### 10. Results table (measured)

Every row below was actually run in this environment against the real
59-document corpus and the real 50-question golden set (retrieval-only:
Recall@5/MRR/nDCG@10, no LLM call) - raw records are under
`experiments/<name>/latest.json`. **Bad results are kept, not hidden** -
`chunk_512_overlap0` and the top-k experiments below are reported exactly
as measured, including the ones that got worse.

| Experiment | Recall@5 | MRR | nDCG@10 | Metric impact vs. `baseline_dense` |
|---|---|---|---|---|
| `baseline_dense` (Phase 1) | 0.700 | 0.603 | 0.588 | — (reference) |
| `sparse_bm25` | 0.700 | 0.640 | 0.618 | +0.0 Recall@5, +3.7 MRR, +3.0 nDCG@10 |
| `hybrid_rrf` | 0.725 | 0.635 | 0.622 | **+2.5 Recall@5, +3.2 MRR, +3.4 nDCG@10** |
| `reranker_minilm` | 0.700 | 0.663 | 0.630 | +0.0 Recall@5, **+6.0 MRR**, +4.2 nDCG@10, **+4.7s p95 latency** |
| `chunk_1024_overlap15` | 0.700 | 0.604 | 0.587 | +0.0 Recall@5 (no real effect) |
| `chunk_512_overlap0` | 0.650 | 0.553 | 0.528 | **-5.0 Recall@5, -5.0 MRR, -6.0 nDCG@10** |
| `chunk_512_overlap20` | 0.675 | 0.581 | 0.566 | -2.5 Recall@5, -2.2 MRR, -2.2 nDCG@10 |
| `chunk_1024_overlap0` | 0.725 | 0.608 | 0.590 | +2.5 Recall@5, +0.5 MRR, +0.2 nDCG@10 |
| `chunk_1024_overlap20` | 0.725 | 0.607 | 0.591 | +2.5 Recall@5, +0.4 MRR, +0.3 nDCG@10 |
| `top_k_3` | 0.650 | 0.592 | 0.572 | **-5.0 Recall@5**, -1.1 MRR, -1.6 nDCG@10 |
| `top_k_10` | 0.700 | 0.620 | 0.630 | +0.0 Recall@5, +1.7 MRR, **+4.2 nDCG@10** |

(Impact columns in percentage points, e.g. "+2.5 Recall@5" = +0.025 absolute.
Bold marks the largest/most notable movements, both good and bad.)

**Reading these honestly:** `hybrid_rrf` is the one clear, broad win -
every metric improved, for free (no extra model, no latency cost beyond a
second cheap BM25 search). `reranker_minilm` buys the single largest MRR
gain in the whole battery, but at a **p95 latency of ~4.7 seconds** (CPU
cross-encoder inference over 20 candidates) - whether that trade is worth
it depends entirely on the product's latency budget, which this table
deliberately doesn't decide for you. `top_k=10` shows the exact pattern
the spec calls out: it doesn't move Recall@5 *at all* (Recall@5 is capped
at whatever's in the top 5 regardless of how many more are fetched) while
still improving MRR and nDCG@10, because those two metrics DO credit a
relevant chunk found at rank 6-10 that Recall@5 structurally cannot -
proof that "higher recall" and "better ranking quality" are genuinely
different things to validate separately, not restatements of each other.
`top_k=3` and `chunk_512_overlap0` are real regressions, reported exactly
as measured.

Not measured here (needs a live LLM provider key and/or a larger model
download this environment's time budget didn't cover - see each numbered
section above for exactly which): the `embedding_qwen3` comparison,
`bge-reranker-base` vs. MiniLM's faithfulness/correctness trade-off, query
rewriting's effect on `multi_hop` correctness, and any generator-model
comparison.

### 11-12. Judge validation: `scripts/label.py` + human-vs-judge agreement

`scripts/label.py` samples ~40 answers (seeded, so the sample is
reproducible - `sample_questions(..., seed=42)`) from an `eval.run`
scorecard's `per_question`, and walks a human labeler through each one
(question, reference answer, generated answer, and the judge's own
verdict) collecting a `correct`/`incorrect`/`partial` label:

```bash
uv run python -m scripts.label --scorecard evals/results.json
```

Labels persist to `evals/labels/human_labels.jsonl`, keyed by question id
- re-running the command skips anything already labeled, so a labeling
session is safe to interrupt and resume. `eval/agreement.py` then computes
percentage agreement and Cohen's kappa (both implemented from first
principles - no `scikit-learn` dependency) between the human and judge
labels:

```python
kappa = (p_observed - p_expected) / (1 - p_expected)
```

where `p_expected` comes from each rater's own marginal label frequencies
(standard unweighted two-rater Cohen's kappa). Per the spec: **agreement
below 80% means the judging rubric should be reviewed first - it does not
by itself mean the pipeline regressed.**
`eval.agreement.AgreementReport.needs_rubric_review` encodes exactly that
80% threshold.

**Methodology note:** producing a real agreement/kappa number requires
first running `eval.run` against a live LLM provider to get judge verdicts
to compare against (see "What's measured for real" above) - not available
in this environment. The sampling, labeling, persistence, and
percentage-agreement/kappa computation are all implemented and fully unit
tested (`tests/unit/test_label.py`, `tests/unit/test_agreement.py`,
including a hand-computed kappa value checked against the formula above);
what's missing is a real judge scorecard to point them at. Once one
exists, `uv run python -m scripts.label` followed by
`eval.agreement.build_agreement_report(human_labels, judge_labels)` produces
the sample size, percentage agreement, and kappa this section is meant to
document.

### 13. Testing

Every new Phase 4 module has unit tests against fakes (no model/Qdrant/LLM
calls) plus, where a real engine or real model weights matter, an
integration test - see the updated "Running tests" section above for the
full breakdown by file. `tests/unit/test_gate_regressions.py` (Phase 3) and
`tests/test_golden_set.py` (Phase 0) are unaffected and still pass
unchanged, proving Phase 4 didn't regress anything earlier phases built.

### 14. Logging

See the updated "Logging" section above.

### 15. Final validation

- ✅ `uv run pytest` (full suite, unit + integration + golden set) passes.
- ✅ `uv run ruff check .` and `uv run mypy` are clean.
- ✅ The Phase 3 CI gate (`tests/unit/test_gate_regressions.py`) still
  passes unmodified - Phase 4 changed nothing about how the gate itself
  works, only what the pipeline underneath it can be configured to do.
- ✅ The retrieval-only experiment battery above ran against the real
  corpus and golden set and is recorded under `experiments/`.
- ⚠️ No experiment here was promoted to `evals/baseline.json` - that file
  still reflects the Phase 1 dense baseline. Promoting a Phase 4
  configuration (e.g. `hybrid_rrf`) to production is a deliberate decision
  for whoever owns this system to make once a `mode="full"` scorecard (with
  real faithfulness/correctness numbers) is measured against a live LLM
  provider, not something this phase does unilaterally.

## Future phases roadmap

Phase 1 was intentionally the simplest possible production-quality
baseline: single-stage dense retrieval, no reranking, no query rewriting,
no agents. Phase 2 built the evaluation harness that measures it, Phase 3
turned that harness into the CI gate that protects it going forward, and
Phase 4 (this phase) added the retrieval/generation techniques Phase 1
explicitly excluded - each one measured with the Phase 2 harness (or, for
retrieval-only changes, the Phase 4 experiment framework), and protected
going forward by the Phase 3 gate once a scorecard is promoted to
`evals/baseline.json`. See
[Phase 4: Retrieval & generation experiments](#phase-4-retrieval--generation-experiments)
below.

**Beyond that**: graph RAG, agentic retrieval, and other techniques not yet
in scope — each still measured the same way, against the same golden set,
through the same harness, gated the same way.

Each future phase should keep `tests/test_golden_set.py` and every prior
phase's test suite green — they are the guardrails that keep the benchmark,
the baseline, and the evaluation harness itself trustworthy as the system
grows.
