"""Optional query rewriting before retrieval (Phase 4).

Rewrites the user's question into a retrieval-optimized standalone query -
particularly aimed at ``multi_hop`` golden-set questions, where the
original phrasing often buries the actual entities/relations a retriever
needs. Toggled entirely through config (``RAG_QUERY_REWRITE_ENABLED``); the
*generator* still always sees the user's original question, only the text
sent to the embedder/retriever is rewritten - a rewriting failure or a bad
rewrite should never change what the system tells the user it's answering.
"""

from __future__ import annotations

import logging

import litellm

logger = logging.getLogger("rag.query_rewriter")

SYSTEM_PROMPT = """You rewrite a user's question into a single, self-contained search query \
optimized for retrieving relevant passages from a document corpus.

Rules:
1. Preserve the original meaning and every entity/constraint in the question - never add facts.
2. If the question has multiple parts or hops, make each part's subject explicit rather than \
using pronouns or ellipsis.
3. Output ONLY the rewritten query, with no explanation, quotes, or preamble.
"""


class QueryRewriteError(RuntimeError):
    """Raised when the rewriting LLM call fails."""


class LiteLLMQueryRewriter:
    """LiteLLM-backed query rewriter. Falls back to the original question on any error."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 128,
        timeout: float = 30.0,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def rewrite(self, question: str) -> str:
        try:
            response = litellm.completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout,
            )
        except Exception as exc:  # noqa: BLE001 - a rewrite failure must not break retrieval
            logger.warning("Query rewrite failed, falling back to original question: %s", exc)
            return question

        rewritten = (response.choices[0].message.content or "").strip()
        if not rewritten:
            logger.warning("Query rewrite returned empty output, falling back to original")
            return question
        logger.debug("Rewrote query %r -> %r", question, rewritten)
        return rewritten
