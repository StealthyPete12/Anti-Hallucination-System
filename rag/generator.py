"""LiteLLM-backed generator: turns retrieved chunks into a cited, grounded answer.

Citation behavior is enforced twice, deliberately: the system prompt instructs
the model to cite only supplied chunk_ids and to refuse when the context is
insufficient, and ``_extract_valid_citations`` independently filters the
model's output against the set of chunk_ids that were actually retrieved.
Anything the model cites that wasn't retrieved is dropped and logged rather
than trusted, so a prompt-following failure can't surface as a fabricated
citation in the API response.
"""

from __future__ import annotations

import logging

import litellm

from rag.citation_validator import CITATION_RE
from rag.cost import estimate_cost_usd
from rag.models import GenerationResult, RetrievedChunk, TokenUsage

logger = logging.getLogger("rag.generator")

SYSTEM_PROMPT = """You are a careful, precise assistant that answers questions using ONLY the \
supplied context passages.

Rules you MUST follow:
1. Every factual claim in your answer must be directly supported by the supplied context.
2. Cite the chunk(s) that support each claim inline, immediately after the claim, using the \
exact format [chunk_id], e.g. "The retention period is 90 days [ops-guide#14]." Use only the \
chunk_ids given to you below - never invent a chunk_id, and never cite a chunk that was not \
supplied.
3. If the supplied context does not contain enough information to answer the question, you \
MUST refuse explicitly. Reply with exactly: "The supplied context does not contain enough \
information to answer that question." Do not guess, speculate, or use outside knowledge.
4. Never fabricate facts, sources, or citations. If in doubt, refuse.
"""


class GenerationError(RuntimeError):
    """Raised when the LLM call fails or returns an unusable response."""


class LiteLLMGenerator:
    """Generator backed by LiteLLM, supporting any LiteLLM-compatible model string."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: float = 60.0,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def generate(self, query: str, chunks: list[RetrievedChunk]) -> GenerationResult:
        retrieved_ids = {chunk.chunk_id for chunk in chunks}
        user_prompt = _build_user_prompt(query, chunks)

        logger.info("Calling LLM model=%s with %d retrieved chunk(s)", self.model, len(chunks))
        try:
            response = litellm.completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout,
            )
        except Exception as exc:  # noqa: BLE001
            raise GenerationError(f"LLM call failed (model={self.model}): {exc}") from exc

        answer = response.choices[0].message.content or ""
        usage = _extract_usage(response)
        cost_usd = _extract_cost(response, self.model, usage)
        citations = _extract_valid_citations(answer, retrieved_ids)

        logger.info("Generated answer with %d valid citation(s)", len(citations))
        return GenerationResult(answer=answer, citations=citations, usage=usage, cost_usd=cost_usd)


def _build_user_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        context = "(no context retrieved)"
    else:
        context = "\n\n".join(f"[{chunk.chunk_id}]\n{chunk.text}" for chunk in chunks)
    return (
        f"Context passages:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer the question using only the context passages above, citing chunk_ids inline "
        "as instructed."
    )


def _extract_valid_citations(answer: str, retrieved_ids: set[str]) -> list[str]:
    seen: list[str] = []
    for match in CITATION_RE.findall(answer):
        if match in retrieved_ids:
            if match not in seen:
                seen.append(match)
        else:
            logger.warning("Model cited chunk_id %r that was not retrieved; dropping", match)
    return seen


def _extract_usage(response: object) -> TokenUsage:
    usage = getattr(response, "usage", None)
    if usage is None:
        return TokenUsage()
    return TokenUsage(
        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        total_tokens=getattr(usage, "total_tokens", 0) or 0,
    )


def _extract_cost(response: object, model: str, usage: TokenUsage) -> float:
    """Cost for one generation call.

    Prefers ``config.MODEL_RATES`` (Phase 5's deterministic, config-driven
    source of truth) when the model is listed there; falls back to
    litellm's own pricing table (unchanged from Phase 1) for every other
    model, so cost is still reported for a model this project's rate table
    hasn't been updated for yet.
    """
    from_config = estimate_cost_usd(model, usage.prompt_tokens, usage.completion_tokens)
    if from_config is not None:
        return from_config
    try:
        return float(litellm.completion_cost(completion_response=response, model=model))
    except Exception as exc:  # noqa: BLE001 - cost lookup can fail for unpriced/unknown models
        logger.debug("Could not compute cost for model=%s: %s", model, exc)
        return 0.0
