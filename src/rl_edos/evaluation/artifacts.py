"""Writes the per-run artifact bundle: comparison.csv/.png, curves.png, attack_trace.png,
summary.json, config.yaml — layout per docs/03_DATA_SCHEMAS.md.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from rl_edos.agents.defender import TRAIN_REWARDS_FILENAME
from rl_edos.config import AttackSpec, ExperimentConfig
from rl_edos.env.attacks import build_attack_fn
from rl_edos.env.traffic import TrafficGenerator
from rl_edos.evaluation.metrics import MetricsSummary, RunRecord, std_metrics
from rl_edos.evaluation.plotting import plot_attack_trace, plot_comparison, plot_curves


def write_run_artifacts(
    run_id: str,
    config: ExperimentConfig,
    attack: AttackSpec,
    results: dict[str, tuple[MetricsSummary, list[RunRecord]]],
    results_dir: Path,
    reward_history_path: Path | None = None,
) -> Path:
    """Write the full artifact bundle for one `evaluate` invocation.

    `reward_history_path`, when given and existing, is a `--policy` checkpoint's
    `train_rewards.csv` (written by `training/train_defender.py::Trainer`):
    its `episode_reward` column becomes `curves.png`'s real learning curve, and
    the `rl_defender` controller's `reward_curve_ref` is set to that plot's
    path. With no history (no `--policy`, or a checkpoint with none saved),
    `curves.png` renders the "no training curve" placeholder instead, and
    `reward_curve_ref` stays `""` for every controller.

    Returns the run's output directory.
    """
    out_dir = results_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_by_controller = {name: metrics for name, (metrics, _records) in results.items()}

    reward_history = load_reward_history(reward_history_path)
    if reward_history and "rl_defender" in metrics_by_controller:
        metrics_by_controller["rl_defender"].reward_curve_ref = str(out_dir / "curves.png")
        # copied alongside comparison.csv so `rl-edos plot --run <id>` can
        # regenerate a real curves.png later without needing the original
        # --policy checkpoint directory to still exist.
        shutil.copy2(reward_history_path, out_dir / TRAIN_REWARDS_FILENAME)

    rows = [
        {
            "controller": name,
            "n_seeds": len(records),
            **metrics.as_dict(),
            **std_metrics([r.metrics for r in records]),
        }
        for name, (metrics, records) in results.items()
    ]
    pd.DataFrame(rows).to_csv(out_dir / "comparison.csv", index=False)

    (out_dir / "config.yaml").write_text(yaml.safe_dump(config.model_dump()))

    plot_curves(reward_history, out_dir / "curves.png")
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


def load_reward_history(path: Path | None) -> list[float]:
    """Read a `train_rewards.csv`'s `episode_reward` column; `[]` if absent/empty.

    Shared with `cli.py`'s `plot` command, which re-reads the copy this module
    writes into `results/<run_id>/train_rewards.csv` rather than the original
    `--policy` checkpoint (which may no longer exist by the time `plot` runs).
    """
    if path is None or not path.is_file():
        return []
    with open(path, newline="") as fh:
        return [float(row["episode_reward"]) for row in csv.DictReader(fh)]


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
