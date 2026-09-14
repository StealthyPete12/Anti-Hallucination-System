"""Unit tests for rag/metrics.py (Phase 5 rolling-window operational metrics)."""

from __future__ import annotations

from rag.metrics import RollingMetrics


def test_empty_window_reports_no_data() -> None:
    metrics = RollingMetrics(window_size=100)
    snapshot = metrics.snapshot()
    assert snapshot.count == 0
    assert snapshot.mean_cost_usd is None
    assert snapshot.p95_latency_ms is None
    assert snapshot.window_size == 100


def test_mean_cost_over_recorded_requests() -> None:
    metrics = RollingMetrics(window_size=100)
    metrics.record(cost_usd=0.001, latency_ms=100)
    metrics.record(cost_usd=0.003, latency_ms=200)
    snapshot = metrics.snapshot()
    assert snapshot.count == 2
    assert snapshot.mean_cost_usd == 0.002


def test_p95_latency_over_recorded_requests() -> None:
    metrics = RollingMetrics(window_size=100)
    for latency in range(1, 101):  # 1..100 ms
        metrics.record(cost_usd=0.0, latency_ms=float(latency))
    snapshot = metrics.snapshot()
    assert snapshot.p95_latency_ms == 95.0


def test_window_evicts_oldest_entries_beyond_window_size() -> None:
    metrics = RollingMetrics(window_size=3)
    for cost in (0.01, 0.02, 0.03, 0.04):
        metrics.record(cost_usd=cost, latency_ms=1.0)
    snapshot = metrics.snapshot()
    # Only the last 3 (0.02, 0.03, 0.04) should count.
    assert snapshot.count == 3
    assert snapshot.mean_cost_usd == (0.02 + 0.03 + 0.04) / 3
