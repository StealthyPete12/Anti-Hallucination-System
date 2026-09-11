FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.12.13 /uv /uvx /usr/local/bin/

WORKDIR /app

# Install dependencies first (separate layer from app code for build caching)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# Pre-download the embedding model at build time (network is available during
# `docker build` even in network-restricted runtime environments), so the
# container never needs internet access to serve a query. Baked into its own
# cached layer, keyed only on pyproject.toml/uv.lock, so app-code changes
# don't force a re-download.
ENV RAG_MODEL_CACHE_DIR=/app/.cache/embeddings
RUN .venv/bin/python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('BAAI/bge-small-en-v1.5', cache_folder='${RAG_MODEL_CACHE_DIR}')"

# Now copy the rest of the application and install the project itself
COPY . .
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:${PATH}"

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
