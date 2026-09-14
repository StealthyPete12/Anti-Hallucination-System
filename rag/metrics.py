"""In-process rolling-window operational metrics (Phase 5).

Deliberately the simplest thing that works: a fixed-size ``deque`` per
process, no external storage. Phoenix (per-trace cost/latency, searchable
and durable across restarts) and Postgres-backed feedback are the durable
sources of truth this project already has; this module exists only to
answer "what's happening right now, in this process" - mean cost and P95
latency over the last N requests - without a round trip to either.

Resets on restart and isn't shared across replicas. That's an accepted
limitation, not an oversight: see the README's Phase 5 section.
"""

from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import dataclass

from config import settings


@dataclass(frozen=True)
class MetricsSnapshot:
    count: int
    mean_cost_usd: float | None
    p95_latency_ms: float | None
    window_size: int


class RollingMetrics:
    """Thread-safe fixed-size rolling window over (cost_usd, latency_ms) pairs."""

    def __init__(self, window_size: int = 100) -> None:
        self.window_size = window_size
        self._costs: deque[float] = deque(maxlen=window_size)
        self._latencies: deque[float] = deque(maxlen=window_size)
        self._lock = threading.Lock()

    def record(self, cost_usd: float, latency_ms: float) -> None:
        with self._lock:
            self._costs.append(cost_usd)
            self._latencies.append(latency_ms)

    def snapshot(self) -> MetricsSnapshot:
        with self._lock:
            costs = list(self._costs)
            latencies = list(self._latencies)

        if not costs:
            return MetricsSnapshot(
                count=0, mean_cost_usd=None, p95_latency_ms=None, window_size=self.window_size
            )

        return MetricsSnapshot(
            count=len(costs),
            mean_cost_usd=sum(costs) / len(costs),
            p95_latency_ms=_percentile(latencies, 0.95),
            window_size=self.window_size,
        )


def _percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile - no numpy/scipy dependency needed for this."""
    ordered = sorted(values)
    rank = math.ceil(p * len(ordered)) - 1
    return ordered[max(0, min(rank, len(ordered) - 1))]


#: Process-wide recorder, sized from settings at import time. A dedicated
#: instance (rather than a fresh one per request) is the point - it's a
#: rolling window across requests.
metrics_recorder = RollingMetrics(window_size=settings.metrics_window_size)
