"""Unit tests for rag/citation_validator.py (Phase 5 runtime faithfulness check)."""

from __future__ import annotations

from rag.citation_validator import extract_citations, validate_citations


def test_extract_citations_dedupes_and_preserves_order() -> None:
    answer = "Fact one [a#0]. Fact two [b#1]. Repeated [a#0]."
    assert extract_citations(answer) == ["a#0", "b#1"]


def test_extract_citations_returns_empty_for_no_citations() -> None:
    assert extract_citations("The supplied context does not contain enough information.") == []


def test_validate_citations_passes_when_all_cited_chunks_were_retrieved() -> None:
    result = validate_citations(
        "The retention period is 90 days [ops-guide#14].",
        retrieved_chunk_ids=["ops-guide#14", "ops-guide#18"],
    )
    assert result.citation_validation_passed is True
    assert result.invalid_citations == []


def test_validate_citations_flags_citation_not_in_retrieved_set() -> None:
    result = validate_citations(
        "See [ops-guide#14] and [ops-guide#99].",
        retrieved_chunk_ids=["ops-guide#14", "ops-guide#18"],
    )
    assert result.citation_validation_passed is False
    assert result.invalid_citations == ["ops-guide#99"]


def test_validate_citations_passes_vacuously_with_no_citations() -> None:
    result = validate_citations("No citations here.", retrieved_chunk_ids=["a#0"])
    assert result.citation_validation_passed is True
    assert result.invalid_citations == []


def test_validate_citations_against_empty_retrieved_set() -> None:
    result = validate_citations("Cites [a#0].", retrieved_chunk_ids=[])
    assert result.citation_validation_passed is False
    assert result.invalid_citations == ["a#0"]
