"""Local sentence-embedding model: BAAI/bge-small-en-v1.5 via sentence-transformers.

BGE models require a distinct instruction prefix on the *query* side only;
documents are embedded as-is. Getting this backwards silently degrades
retrieval quality without raising an error, so query vs. document embedding
is centralized here as the one place that divergence happens.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

logger = logging.getLogger("rag.embedders.local")

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class EmbeddingError(RuntimeError):
    """Raised when the embedding model fails to load or embed input."""


class LocalBGEEmbedder:
    """Local, disk-cached sentence-transformers embedder for BAAI/bge-small-en-v1.5."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        cache_dir: str | None = None,
        batch_size: int = 32,
        device: str = "cpu",
        normalize: bool = True,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.device = device
        self.normalize = normalize
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> Any:
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _load_model(self) -> SentenceTransformer:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingError(
                "sentence-transformers is required for LocalBGEEmbedder; install it via "
                "`uv add sentence-transformers`"
            ) from exc

        logger.info("Loading embedding model %s (cache_dir=%s)", self.model_name, self.cache_dir)
        try:
            model = SentenceTransformer(
                self.model_name, cache_folder=self.cache_dir, device=self.device
            )
        except Exception as exc:  # noqa: BLE001 - surfaced as a typed error for callers
            raise EmbeddingError(
                f"Failed to load embedding model {self.model_name!r}: {exc}"
            ) from exc
        logger.info(
            "Embedding model %s loaded (dim=%s)",
            self.model_name,
            model.get_embedding_dimension(),
        )
        return model

    @property
    def dimension(self) -> int:
        dim = self.model.get_embedding_dimension()
        if dim is None:
            raise EmbeddingError(f"Could not determine embedding dimension for {self.model_name}")
        return int(dim)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document chunks. No BGE prefix is applied."""
        if not texts:
            return []
        logger.debug("Embedding %d document(s) in batches of %d", len(texts), self.batch_size)
        try:
            vectors = self.model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingError(f"Failed to embed {len(texts)} document(s): {exc}") from exc
        return [vector.tolist() for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single search query, applying the required BGE query prefix."""
        prefixed = f"{BGE_QUERY_PREFIX}{text}"
        try:
            vector = self.model.encode(
                prefixed,
                normalize_embeddings=self.normalize,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingError(f"Failed to embed query: {exc}") from exc
        return list(vector.tolist())

    def count_tokens(self, text: str) -> int:
        """Count tokens using this model's own tokenizer.

        Used to drive ``rag.chunking`` so chunk boundaries respect this
        model's real max-sequence-length (512 tokens for bge-small-en-v1.5).
        """
        tokenizer = self.model.tokenizer
        encoded: list[int] = tokenizer.encode(text, add_special_tokens=False)
        return len(encoded)
