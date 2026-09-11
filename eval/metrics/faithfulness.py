"""Faithfulness metric: per-claim hallucination detection against retrieved context.

Per the design brief, this deliberately does NOT do vague 1-5 LLM scoring.
The judge breaks a generated answer into atomic factual claims and verdicts
each one independently against the retrieved context; the faithfulness score
is the fraction of claims that were supported. Per-claim verdicts are more
reproducible than a single holistic score, and they're directly useful for
error analysis (a low score comes with the exact unsupported claims, not
just a number).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from eval.judge import JudgeResponse, LLMJudge

PROMPT_NAME = "faithfulness_v1"


@dataclass(frozen=True)
class Claim:
    text: str
    supported: bool
    chunk_id: str | None


@dataclass(frozen=True)
class FaithfulnessEvaluation:
    claims: list[Claim]
    unsupported_count: int
    score: float | None
    judge_response: JudgeResponse


def format_context(chunks: list[tuple[str, str]]) -> str:
    """Render ``(chunk_id, text)`` pairs as the bracketed-id context block the prompt expects."""
    if not chunks:
        return "(no context retrieved)"
    return "\n\n".join(f"[{chunk_id}]\n{text}" for chunk_id, text in chunks)


def parse_claims(data: dict[str, Any]) -> tuple[list[Claim], int]:
    """Validate and parse the judge's raw JSON into typed claims.

    Raises ``ValueError`` on a malformed response so the caller can decide
    whether to retry or fail the question outright - a silently-empty claim
    list would make a hallucinating answer look perfectly faithful.
    """
    raw_claims = data.get("claims")
    if not isinstance(raw_claims, list):
        raise ValueError(f"Faithfulness judge response missing 'claims' list: {data!r}")

    claims: list[Claim] = []
    for raw in raw_claims:
        if not isinstance(raw, dict) or "text" not in raw or "supported" not in raw:
            raise ValueError(f"Malformed claim in faithfulness judge response: {raw!r}")
        claims.append(
            Claim(
                text=str(raw["text"]),
                supported=bool(raw["supported"]),
                chunk_id=raw.get("chunk_id"),
            )
        )

    unsupported_count = data.get("unsupported_count")
    if not isinstance(unsupported_count, int):
        unsupported_count = sum(1 for c in claims if not c.supported)

    return claims, unsupported_count


def score_claims(claims: list[Claim]) -> float | None:
    """supported_claims / total_claims.

    An answer with zero claims (e.g. a pure refusal) makes no assertions at
    all, so it cannot hallucinate - it is vacuously faithful, scored 1.0
    rather than left undefined. This keeps refusals from being silently
    dropped out of the faithfulness average.
    """
    if not claims:
        return 1.0
    supported = sum(1 for c in claims if c.supported)
    return supported / len(claims)


async def evaluate_faithfulness(
    judge: LLMJudge, *, answer: str, retrieved_chunks: list[tuple[str, str]]
) -> FaithfulnessEvaluation:
    """Judge whether ``answer``'s claims are supported by ``retrieved_chunks``.

    ``retrieved_chunks`` is a list of ``(chunk_id, text)`` pairs - exactly
    what the generator itself saw, so faithfulness is graded against the
    real evidence rather than a re-derived approximation of it.
    """
    context = format_context(retrieved_chunks)
    response = await judge.evaluate(PROMPT_NAME, {"context": context, "answer": answer})
    claims, unsupported_count = parse_claims(response.data)
    return FaithfulnessEvaluation(
        claims=claims,
        unsupported_count=unsupported_count,
        score=score_claims(claims),
        judge_response=response,
    )


def mean_faithfulness(evaluations: list[FaithfulnessEvaluation]) -> float | None:
    scores = [e.score for e in evaluations if e.score is not None]
    if not scores:
        return None
    return sum(scores) / len(scores)
