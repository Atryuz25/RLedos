"""rl-edos command surface: train-defender, evaluate, selfplay, plot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

from rl_edos.baselines import TargetTrackingController, train_security_blind
from rl_edos.config import AttackSpec, ConfigError, ExperimentConfig, load_config
from rl_edos.evaluation.artifacts import write_attack_trace, write_run_artifacts
from rl_edos.evaluation.evaluator import Controller, Evaluator
from rl_edos.evaluation.metrics import MetricsSummary, RunRecord
from rl_edos.evaluation.plotting import plot_comparison, plot_curves
from rl_edos.training.sb3_utils import rescale_action

RESULTS_DIR = Path("results")


def _add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="path to a consolidated experiment YAML")


def _train_defender(args: argparse.Namespace) -> int:
    load_config(args.config)
    print("train-defender: not yet implemented in this phase (Phase 3)")
    return 0


def _build_controllers(
    config: ExperimentConfig, baselines_arg: str, policy: str
) -> dict[str, Controller]:
    names = (
        ["target_tracking", "security_blind_rl"]
        if baselines_arg == "all"
        else [b.strip() for b in baselines_arg.split(",") if b.strip()]
    )

    controllers: dict[str, Controller] = {}
    for name in names:
        if name == "target_tracking":
            controllers[name] = TargetTrackingController(
                config.sim, target_utilization=config.baselines.target_utilization
            )
        elif name == "security_blind_rl":
            print("training security_blind_rl baseline...")
            controllers[name] = train_security_blind(config)
        elif name == "randomisation":
            print("skipping randomisation baseline: not implemented (cuttable per Phase 2 spec)")
        else:
            raise ConfigError(f"unknown baseline {name!r}")

    if policy and policy != "none":
        from stable_baselines3 import PPO

        model = PPO.load(policy)

        def _rl_defender(state, _model=model):
            action, _ = _model.predict(state.to_obs(), deterministic=True)
            real_action = rescale_action(action, config.sim.min_instances, config.sim.max_instances)
            clipped = np.clip(real_action, config.sim.min_instances, config.sim.max_instances)
            return np.asarray(clipped, dtype=np.float32)

        controllers["rl_defender"] = _rl_defender

    return controllers


def _evaluate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    controllers = _build_controllers(config, args.baselines, args.policy)
    if not controllers:
        raise ConfigError("no controllers to evaluate")

    evaluator = Evaluator(config)
    results = evaluator.run(controllers)

    run_id = RunRecord.new_run_id()
    out_dir = write_run_artifacts(run_id, config, config.attack, results, RESULTS_DIR)
    print(f"wrote {out_dir}")
    for name, (metrics, _records) in results.items():
        print(f"  {name}: {metrics.as_dict()}")
    return 0


def _selfplay(args: argparse.Namespace) -> int:
    load_config(args.config)
    print("selfplay: not yet implemented in this phase (Phase 4)")
    return 0


def _plot(args: argparse.Namespace) -> int:
    run_dir = RESULTS_DIR / args.run
    config_path = run_dir / "config.yaml"
    summary_path = run_dir / "summary.json"
    if not config_path.is_file() or not summary_path.is_file():
        raise ConfigError(f"no run found at {run_dir}")

    config = ExperimentConfig(**yaml.safe_load(config_path.read_text()))
    summary = json.loads(summary_path.read_text())
    attack = AttackSpec(**summary["attack"])
    metrics_by_controller = {
        name: MetricsSummary(**c["metrics"]) for name, c in summary["controllers"].items()
    }

    plot_curves([], run_dir / "curves.png")
    plot_comparison(metrics_by_controller, run_dir / "comparison.png")
    write_attack_trace(config, attack, run_dir / "attack_trace.png")
    print(f"regenerated plots in {run_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rl-edos", description="RL-EDoS command surface")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train-defender", help="train PPO defender vs scripted attacks")
    _add_config_arg(p_train)
    p_train.set_defaults(func=_train_defender)

    p_eval = sub.add_parser("evaluate", help="run all controllers under identical attacks")
    _add_config_arg(p_eval)
    p_eval.add_argument("--policy", default="none", help="path to a defender policy, or 'none'")
    p_eval.add_argument("--baselines", default="all", help="comma-separated baseline set")
    p_eval.set_defaults(func=_evaluate)

    p_selfplay = sub.add_parser("selfplay", help="co-evolutionary attacker/defender loop")
    _add_config_arg(p_selfplay)
    p_selfplay.set_defaults(func=_selfplay)

    p_plot = sub.add_parser("plot", help="regenerate curves/comparison/attack-trace plots")
    p_plot.add_argument("--run", required=True, help="run_id under results/")
    p_plot.set_defaults(func=_plot)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
