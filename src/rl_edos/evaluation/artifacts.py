"""Writes the per-run artifact bundle: comparison.csv/.png, curves.png, attack_trace.png,
summary.json, config.yaml — layout per docs/03_DATA_SCHEMAS.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from rl_edos.config import AttackSpec, ExperimentConfig
from rl_edos.env.attacks import build_attack_fn
from rl_edos.env.traffic import TrafficGenerator
from rl_edos.evaluation.metrics import MetricsSummary, RunRecord
from rl_edos.evaluation.plotting import plot_attack_trace, plot_comparison, plot_curves


def write_run_artifacts(
    run_id: str,
    config: ExperimentConfig,
    attack: AttackSpec,
    results: dict[str, tuple[MetricsSummary, list[RunRecord]]],
    results_dir: Path,
) -> Path:
    """Write the full artifact bundle for one `evaluate` invocation.

    Returns the run's output directory.
    """
    out_dir = results_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_by_controller = {name: metrics for name, (metrics, _records) in results.items()}

    rows = [
        {"controller": name, **metrics.as_dict()} for name, metrics in metrics_by_controller.items()
    ]
    pd.DataFrame(rows).to_csv(out_dir / "comparison.csv", index=False)

    (out_dir / "config.yaml").write_text(yaml.safe_dump(config.model_dump()))

    plot_curves([], out_dir / "curves.png")
    plot_comparison(metrics_by_controller, out_dir / "comparison.png")
    write_attack_trace(config, attack, out_dir / "attack_trace.png")

    summary = {
        "run_id": run_id,
        "attack": attack.model_dump(),
        "controllers": {
            name: {"metrics": metrics.as_dict(), "run_records": [r.as_dict() for r in records]}
            for name, (metrics, records) in results.items()
        },
        "text_summary": _text_summary(metrics_by_controller),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    return out_dir


def write_attack_trace(config: ExperimentConfig, attack: AttackSpec, out_path: Path) -> None:
    rng = np.random.default_rng(config.sim.seed)
    traffic = TrafficGenerator(config.traffic, rng)
    attack_fn = build_attack_fn(attack, config.sim, config.traffic.base_rate)
    times = [i * config.sim.control_interval_s for i in range(config.sim.sim_horizon_steps)]
    legit_rates = [traffic.sample(t) for t in times]
    attack_rates = [attack_fn(t) for t in times]
    plot_attack_trace(times, legit_rates, attack_rates, out_path)


def _text_summary(metrics_by_controller: dict[str, MetricsSummary]) -> str:
    lines = [
        "controller, mean_cost_under_attack, p95_latency_ms, legit_drop_rate, overprovision_ratio"
    ]
    for name, m in metrics_by_controller.items():
        lines.append(
            f"{name}: cost={m.mean_cost_under_attack:.6f} "
            f"p95_latency_ms={m.p95_latency_ms:.2f} "
            f"legit_drop_rate={m.legit_drop_rate:.4f} "
            f"overprovision_ratio={m.overprovision_ratio:.3f}"
        )
    return "\n".join(lines)
