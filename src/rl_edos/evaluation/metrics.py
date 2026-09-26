"""MetricsSummary computation and the RunRecord persisted per controller/seed run.

Field definitions and units per docs/03_DATA_SCHEMAS.md.
"""

from __future__ import annotations

import math
import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from rl_edos.config import AttackSpec


@dataclass
class EpisodeTrace:
    """Per-step series collected while running one controller through one episode."""

    costs: list[float] = field(default_factory=list)
    latencies_ms: list[float] = field(default_factory=list)
    legit_rates: list[float] = field(default_factory=list)
    active_instances: list[int] = field(default_factory=list)
    legit_incoming: list[float] = field(default_factory=list)
    legit_dropped: list[float] = field(default_factory=list)


@dataclass
class MetricsSummary:
    """Success-axis metrics for one controller under one attack."""

    mean_cost_under_attack: float  # currency, mean per-interval accrued cost
    p95_latency_ms: float  # milliseconds
    legit_drop_rate: float  # fraction of legitimate requests dropped, in [0, 1]
    overprovision_ratio: float  # provisioned capacity / capacity legit demand alone needs
    reward_curve_ref: str = ""  # path to the learning-curve artifact, if any

    def as_dict(self) -> dict:
        return {
            "mean_cost_under_attack": self.mean_cost_under_attack,
            "p95_latency_ms": self.p95_latency_ms,
            "legit_drop_rate": self.legit_drop_rate,
            "overprovision_ratio": self.overprovision_ratio,
            "reward_curve_ref": self.reward_curve_ref,
        }


def compute_metrics(
    trace: EpisodeTrace, capacity_rps_per_instance: float, min_instances: int
) -> MetricsSummary:
    """Reduce one episode's trace to a MetricsSummary, per docs/03_DATA_SCHEMAS.md definitions."""
    mean_cost = float(np.mean(trace.costs)) if trace.costs else 0.0
    p95_latency = float(np.percentile(trace.latencies_ms, 95)) if trace.latencies_ms else 0.0

    total_legit_incoming = sum(trace.legit_incoming)
    total_legit_dropped = sum(trace.legit_dropped)
    legit_drop_rate = (
        total_legit_dropped / total_legit_incoming if total_legit_incoming > 0 else 0.0
    )

    needed = [
        max(min_instances, math.ceil(rate / capacity_rps_per_instance))
        for rate in trace.legit_rates
    ]
    mean_needed = float(np.mean(needed)) if needed else 0.0
    mean_provisioned = float(np.mean(trace.active_instances)) if trace.active_instances else 0.0
    overprovision_ratio = mean_provisioned / mean_needed if mean_needed > 0 else 0.0

    return MetricsSummary(
        mean_cost_under_attack=mean_cost,
        p95_latency_ms=p95_latency,
        legit_drop_rate=legit_drop_rate,
        overprovision_ratio=overprovision_ratio,
    )


def average_metrics(summaries: list[MetricsSummary]) -> MetricsSummary:
    """Average MetricsSummary across seeds (element-wise mean)."""
    return MetricsSummary(
        mean_cost_under_attack=float(np.mean([m.mean_cost_under_attack for m in summaries])),
        p95_latency_ms=float(np.mean([m.p95_latency_ms for m in summaries])),
        legit_drop_rate=float(np.mean([m.legit_drop_rate for m in summaries])),
        overprovision_ratio=float(np.mean([m.overprovision_ratio for m in summaries])),
    )


# The numeric MetricsSummary fields std_metrics reports a spread for (excludes
# reward_curve_ref, which is a path, not a number).
STD_METRIC_FIELDS = (
    "mean_cost_under_attack",
    "p95_latency_ms",
    "legit_drop_rate",
    "overprovision_ratio",
)


def std_metrics(summaries: list[MetricsSummary]) -> dict[str, float]:
    """Per-metric standard deviation across seeds; 0.0 for every field with a single seed.

    Used by `evaluation/artifacts.py` to add `*_std` columns to `comparison.csv`
    so a multi-seed `evaluate --seeds ...` run reports spread, not just the mean.
    """
    return {
        f"{field}_std": float(np.std([getattr(m, field) for m in summaries]))
        for field in STD_METRIC_FIELDS
    }


def git_sha() -> str:
    """Current git commit SHA, or 'unknown' outside a git repo / before the first commit."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=5
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


@dataclass
class RunRecord:
    """One controller's evaluation run, for reproducibility."""

    run_id: str
    config_hash: str
    controller: str
    attack: AttackSpec
    metrics: MetricsSummary
    artifacts: list[str]
    seed: int
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    git_sha: str = field(default_factory=git_sha)

    @staticmethod
    def new_run_id() -> str:
        return str(uuid.uuid4())

    def as_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "config_hash": self.config_hash,
            "controller": self.controller,
            "attack": self.attack.model_dump(),
            "metrics": self.metrics.as_dict(),
            "artifacts": self.artifacts,
            "created_at": self.created_at,
            "git_sha": self.git_sha,
            "seed": self.seed,
        }
