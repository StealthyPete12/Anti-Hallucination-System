"""LiteLLM-backed judge client for LLM-graded evaluation metrics.

Every judge call is: temperature 0, structured JSON output, timeout-bounded,
retried with exponential backoff on transient failure, logged, and
cost-tracked. Prompts live as versioned text files in ``eval/prompts/``
(never embedded in code) and every call records the SHA256 hash of the exact
prompt template that produced it, so a future prompt edit is immediately
visible as a hash change in the scorecard rather than silently changing what
"faithfulness" or "correctness" means run to run.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from string import Template
from typing import Any

import litellm

from rag.models import TokenUsage

logger = logging.getLogger("eval.judge")

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


class JudgeError(RuntimeError):
    """Raised when a judge call fails after exhausting retries, or its prompt is missing."""


@dataclass(frozen=True)
class PromptTemplate:
    """A versioned prompt file, loaded once and hashed for traceability."""

    name: str
    version: str
    text: str
    sha256: str


@cache
def load_prompt(name: str) -> PromptTemplate:
    """Load and hash a versioned prompt template from ``eval/prompts/<name>.txt``.

    ``name`` is expected to carry its own version suffix, e.g.
    ``"faithfulness_v1"`` -> version ``"v1"``. Reading is cached: a given
    prompt file is only read and hashed once per process.
    """
    path = PROMPTS_DIR / f"{name}.txt"
    if not path.exists():
        raise JudgeError(f"Unknown prompt {name!r}: no such file {path}")
    text = path.read_text(encoding="utf-8")
    version = name.rsplit("_", 1)[-1] if "_" in name else "v0"
    sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return PromptTemplate(name=name, version=version, text=text, sha256=sha256)


@dataclass(frozen=True)
class JudgeResponse:
    """One judge call's full result, kept for traceability and cost accounting."""

    data: dict[str, Any]
    prompt_name: str
    prompt_version: str
    prompt_hash: str
    usage: TokenUsage
    cost_usd: float
    latency_ms: float
    raw_content: str
    attempts: int


class LLMJudge:
    """Deterministic (temperature=0), JSON-structured, retrying LiteLLM judge client."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: float = 60.0,
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay

    async def evaluate(self, prompt_name: str, variables: dict[str, str]) -> JudgeResponse:
        """Render ``prompt_name`` with ``variables``, call the judge model, and parse its JSON.

        Retries (with exponential backoff) on any exception raised by the
        call itself, a timeout, or a response that isn't a parseable JSON
        object - a judge model occasionally wraps its output in prose or
        markdown fences despite instructions, and a retry is cheap and
        usually fixes it. Raises :class:`JudgeError` once retries are
        exhausted.
        """
        template = load_prompt(prompt_name)
        content = Template(template.text).substitute(**variables)

        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            t0 = time.perf_counter()
            try:
                response = await litellm.acompletion(
                    model=self.model,
                    messages=[{"role": "user", "content": content}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    timeout=self.timeout,
                    response_format={"type": "json_object"},
                )
                latency_ms = (time.perf_counter() - t0) * 1000
                raw_content = response.choices[0].message.content or ""
                data = json.loads(raw_content)
                if not isinstance(data, dict):
                    raise ValueError(f"Judge returned non-object JSON: {raw_content!r}")

                usage = _extract_usage(response)
                cost_usd = _extract_cost(response, self.model)
                logger.info(
                    "judge_call prompt=%s prompt_hash=%s model=%s attempt=%d "
                    "latency_ms=%.1f cost_usd=%.6f",
                    prompt_name,
                    template.sha256[:12],
                    self.model,
                    attempt,
                    latency_ms,
                    cost_usd,
                )
                return JudgeResponse(
                    data=data,
                    prompt_name=prompt_name,
                    prompt_version=template.version,
                    prompt_hash=template.sha256,
                    usage=usage,
                    cost_usd=cost_usd,
                    latency_ms=latency_ms,
                    raw_content=raw_content,
                    attempts=attempt,
                )
            except Exception as exc:  # noqa: BLE001 - transient network/provider/JSON errors
                last_exc = exc
                logger.warning(
                    "judge_call_failed prompt=%s model=%s attempt=%d/%d error=%s",
                    prompt_name,
                    self.model,
                    attempt,
                    self.max_retries,
                    exc,
                )
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_base_delay * (2 ** (attempt - 1)))

        raise JudgeError(
            f"Judge call for prompt={prompt_name!r} model={self.model!r} failed after "
            f"{self.max_retries} attempt(s): {last_exc}"
        ) from last_exc


def _extract_usage(response: object) -> TokenUsage:
    usage = getattr(response, "usage", None)
    if usage is None:
        return TokenUsage()
    return TokenUsage(
        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        total_tokens=getattr(usage, "total_tokens", 0) or 0,
    )


def _extract_cost(response: object, model: str) -> float:
    try:
        return float(litellm.completion_cost(completion_response=response, model=model))
    except Exception as exc:  # noqa: BLE001 - cost lookup can fail for unpriced/unknown models
        logger.debug("Could not compute cost for model=%s: %s", model, exc)
        return 0.0
