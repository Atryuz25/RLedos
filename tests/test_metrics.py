"""Metric math: p95 latency, legit drop rate, overprovision ratio — hand-computed oracles."""

from __future__ import annotations

from rl_edos.evaluation.metrics import (
    EpisodeTrace,
    MetricsSummary,
    average_metrics,
    compute_metrics,
    git_sha,
)


def test_compute_metrics_matches_hand_computation():
    trace = EpisodeTrace(
        costs=[1.0, 2.0, 3.0],
        latencies_ms=[10.0, 20.0, 30.0, 40.0],
        legit_rates=[10.0, 10.0, 10.0],
        active_instances=[4, 4, 4],
        legit_incoming=[100.0, 100.0, 100.0],
        legit_dropped=[0.0, 0.0, 50.0],
    )
    metrics = compute_metrics(trace, capacity_rps_per_instance=5.0, min_instances=1)

    assert metrics.mean_cost_under_attack == 2.0  # mean([1,2,3])
    # numpy linear-interpolation p95 of [10,20,30,40]: index 0.95*3=2.85 -> 30 + 0.85*10
    assert metrics.p95_latency_ms == 38.5
    assert metrics.legit_drop_rate == 50.0 / 300.0
    # needed = ceil(10/5) = 2 each interval; provisioned = 4 each -> 4/2
    assert metrics.overprovision_ratio == 2.0


def test_compute_metrics_handles_empty_trace():
    trace = EpisodeTrace()
    metrics = compute_metrics(trace, capacity_rps_per_instance=5.0, min_instances=1)
    assert metrics.mean_cost_under_attack == 0.0
    assert metrics.p95_latency_ms == 0.0
    assert metrics.legit_drop_rate == 0.0
    assert metrics.overprovision_ratio == 0.0


def test_average_metrics_is_elementwise_mean():
    a = MetricsSummary(
        mean_cost_under_attack=1.0,
        p95_latency_ms=10.0,
        legit_drop_rate=0.0,
        overprovision_ratio=1.0,
    )
    b = MetricsSummary(
        mean_cost_under_attack=3.0,
        p95_latency_ms=30.0,
        legit_drop_rate=0.2,
        overprovision_ratio=3.0,
    )
    avg = average_metrics([a, b])
    assert avg.mean_cost_under_attack == 2.0
    assert avg.p95_latency_ms == 20.0
    assert avg.legit_drop_rate == 0.1
    assert avg.overprovision_ratio == 2.0


def test_git_sha_never_raises_outside_a_repo():
    sha = git_sha()
    assert isinstance(sha, str)
    assert sha  # non-empty, even if "unknown"
