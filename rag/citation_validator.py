"""Runtime (serving-time) citation validation - Phase 5.

Independent of ``rag/generator.py``'s own citation filtering (which silently
drops an invalid citation from the answer's ``citations`` list so the API
never surfaces a fabricated source as if it were real). This module instead
*flags* the same condition as an explicit, deterministic hallucination
indicator: it re-parses the raw generated answer text for ``[chunk_id]``
citations and checks each one against what was actually retrieved for that
query, regardless of what the generator already sanitized.

Near-zero cost (one regex pass, one set lookup per citation), fully
deterministic, no LLM call - safe to run on every request.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

#: Matches the same inline citation format the generator's system prompt
#: requires: "[chunk_id]". Kept here as the single source of truth;
#: rag/generator.py imports it rather than defining its own copy.
CITATION_RE = re.compile(r"\[([^\[\]\s]+)\]")


def extract_citations(answer: str) -> list[str]:
    """Every ``[chunk_id]``-shaped citation in ``answer``, in order, deduped."""
    seen: list[str] = []
    for match in CITATION_RE.findall(answer):
        if match not in seen:
            seen.append(match)
    return seen


@dataclass(frozen=True)
class CitationValidation:
    """Result of validating an answer's citations against what was retrieved."""

    citation_validation_passed: bool
    invalid_citations: list[str] = field(default_factory=list)


def validate_citations(answer: str, retrieved_chunk_ids: list[str]) -> CitationValidation:
    """Flag any citation in ``answer`` that isn't in ``retrieved_chunk_ids``.

    A citation not present in what was actually retrieved for this query is
    a serving-time hallucination indicator: the model asserted a source it
    was never given.
    """
    retrieved_set = set(retrieved_chunk_ids)
    invalid = [c for c in extract_citations(answer) if c not in retrieved_set]
    return CitationValidation(citation_validation_passed=not invalid, invalid_citations=invalid)
