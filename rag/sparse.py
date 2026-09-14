"""BM25 sparse-vector encoding for hybrid retrieval.

Produces the same kind of sparse (term_id -> weight) vectors that Qdrant's
sparse vector index expects, computed with the standard Okapi BM25 formula.
Deliberately dependency-free (no ``rank_bm25``/``fastembed``): a fixed
vocabulary is fit once from the corpus at index time and persisted
(:meth:`BM25Vectorizer.save` / :meth:`BM25Vectorizer.load`) so query-time
encoding always maps tokens to the same term ids used at indexing time.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rag.models import SparseVector

logger = logging.getLogger("rag.sparse")

_TOKEN_RE = re.compile(r"[a-z0-9]+")

DEFAULT_K1 = 1.5
DEFAULT_B = 0.75


def tokenize(text: str) -> list[str]:
    """Lowercase, alphanumeric-only tokenization. Fast and dependency-free."""
    return _TOKEN_RE.findall(text.lower())


class SparseEncodingError(RuntimeError):
    """Raised when BM25 encoding is attempted before the vectorizer is fit."""


@dataclass
class BM25Vectorizer:
    """Fits a BM25 vocabulary/IDF table from a corpus and encodes documents/queries."""

    k1: float = DEFAULT_K1
    b: float = DEFAULT_B
    vocabulary: dict[str, int] = field(default_factory=dict)
    idf: dict[int, float] = field(default_factory=dict)
    avg_doc_len: float = 0.0
    num_docs: int = 0

    @property
    def is_fit(self) -> bool:
        return bool(self.vocabulary)

    def fit(self, documents: list[str]) -> BM25Vectorizer:
        """Build the vocabulary, document frequencies, and IDF table from a corpus."""
        if not documents:
            raise ValueError("Cannot fit BM25Vectorizer on an empty document list")

        doc_token_lists = [tokenize(doc) for doc in documents]
        self.num_docs = len(doc_token_lists)
        self.avg_doc_len = sum(len(toks) for toks in doc_token_lists) / self.num_docs

        vocabulary: dict[str, int] = {}
        doc_freq: Counter[int] = Counter()
        for tokens in doc_token_lists:
            term_ids: set[int] = set()
            for token in tokens:
                term_id = vocabulary.setdefault(token, len(vocabulary))
                term_ids.add(term_id)
            doc_freq.update(term_ids)

        self.vocabulary = vocabulary
        # BM25 IDF (Robertson/Sparck-Jones), floored at a small positive value
        # so a term appearing in every document never gets a negative weight.
        self.idf = {
            term_id: max(
                math.log(1 + (self.num_docs - df + 0.5) / (df + 0.5)),
                1e-6,
            )
            for term_id, df in doc_freq.items()
        }
        logger.info(
            "Fit BM25Vectorizer: vocab_size=%d num_docs=%d avg_doc_len=%.1f",
            len(self.vocabulary),
            self.num_docs,
            self.avg_doc_len,
        )
        return self

    def _require_fit(self) -> None:
        if not self.is_fit:
            raise SparseEncodingError(
                "BM25Vectorizer has not been fit; call .fit(documents) or .load(path) first"
            )

    def encode_document(self, text: str) -> SparseVector:
        """BM25 term weights for one document, using the fitted IDF/avg-doc-len."""
        self._require_fit()
        tokens = tokenize(text)
        doc_len = len(tokens)
        term_counts = Counter(tokens)

        indices: list[int] = []
        values: list[float] = []
        for token, tf in term_counts.items():
            term_id = self.vocabulary.get(token)
            if term_id is None:
                continue  # out-of-vocabulary term: no signal to contribute
            idf = self.idf.get(term_id, 0.0)
            denom = tf + self.k1 * (1 - self.b + self.b * doc_len / max(self.avg_doc_len, 1e-6))
            weight = idf * (tf * (self.k1 + 1)) / denom
            if weight > 0:
                indices.append(term_id)
                values.append(weight)
        return SparseVector(indices=indices, values=values)

    def encode_query(self, text: str) -> SparseVector:
        """Query-side weights: plain IDF per unique query term (standard BM25 query weighting)."""
        self._require_fit()
        indices: list[int] = []
        values: list[float] = []
        for token in set(tokenize(text)):
            term_id = self.vocabulary.get(token)
            if term_id is None:
                continue
            idf = self.idf.get(term_id, 0.0)
            if idf > 0:
                indices.append(term_id)
                values.append(idf)
        return SparseVector(indices=indices, values=values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "k1": self.k1,
            "b": self.b,
            "vocabulary": self.vocabulary,
            "idf": {str(k): v for k, v in self.idf.items()},
            "avg_doc_len": self.avg_doc_len,
            "num_docs": self.num_docs,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BM25Vectorizer:
        return cls(
            k1=float(data["k1"]),
            b=float(data["b"]),
            vocabulary=dict(data["vocabulary"]),
            idf={int(k): float(v) for k, v in data["idf"].items()},
            avg_doc_len=float(data["avg_doc_len"]),
            num_docs=int(data["num_docs"]),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict()), encoding="utf-8")
        logger.info("Saved BM25Vectorizer to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> BM25Vectorizer:
        path = Path(path)
        if not path.exists():
            raise SparseEncodingError(f"No BM25 vocabulary file at {path}")
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
