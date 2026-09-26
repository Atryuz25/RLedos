"""rl-edos command surface: train-defender, evaluate, selfplay, plot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from rl_edos.agents.defender import TRAIN_REWARDS_FILENAME, load_defender
from rl_edos.baselines import TargetTrackingController, train_security_blind
from rl_edos.config import AttackSpec, ConfigError, ExperimentConfig, load_config
from rl_edos.env.attacks import build_attack_fn
from rl_edos.evaluation.artifacts import (
    load_reward_history,
    write_attack_trace,
    write_run_artifacts,
)
from rl_edos.evaluation.evaluator import Controller, Evaluator
from rl_edos.evaluation.metrics import MetricsSummary, RunRecord
from rl_edos.evaluation.plotting import plot_comparison, plot_curves, plot_selfplay_stability
from rl_edos.training.sb3_utils import TrainingDivergedError
from rl_edos.training.selfplay import run_selfplay
from rl_edos.training.train_defender import Trainer

RESULTS_DIR = Path("results")
MODELS_DIR = Path("models")


def _add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="path to a consolidated experiment YAML")


def _train_defender(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    attack_fn = build_attack_fn(config.attack, config.sim, config.traffic.base_rate)
    out_dir = Path(args.out) if args.out else MODELS_DIR / RunRecord.new_run_id()

    print(
        f"training defender: {config.agent.total_timesteps} PPO timesteps "
        f"against a {config.attack.pattern!r} attack (intensity={config.attack.intensity})..."
    )
    try:
        ckpt_dir = Trainer().train(config, out_dir, attack_traffic_fn=attack_fn)
    except TrainingDivergedError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"wrote checkpoint to {ckpt_dir}")
    print(
        f"evaluate with: rl-edos evaluate --config {args.config} "
        f"--policy {ckpt_dir} --baselines all"
    )
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
        controllers["rl_defender"] = load_defender(policy, config.sim)

    return controllers


def _evaluate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    controllers = _build_controllers(config, args.baselines, args.policy)
    if not controllers:
        raise ConfigError("no controllers to evaluate")

    evaluator = Evaluator(config)
    results = evaluator.run(controllers, seeds=args.seeds)

    reward_history_path = None
    if args.policy and args.policy != "none":
        reward_history_path = Path(args.policy) / TRAIN_REWARDS_FILENAME

    run_id = RunRecord.new_run_id()
    out_dir = write_run_artifacts(
        run_id, config, config.attack, results, RESULTS_DIR, reward_history_path
    )
    print(f"wrote {out_dir}")
    for name, (metrics, records) in results.items():
        print(f"  {name} (n_seeds={len(records)}): {metrics.as_dict()}")
    return 0


def _selfplay(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    out_dir = Path(args.out) if args.out else MODELS_DIR / "selfplay" / RunRecord.new_run_id()

    print(
        f"selfplay: {config.selfplay.rounds} rounds x {config.selfplay.round_timesteps} "
        f"timesteps/agent, evasion_budget={config.attack.evasion_budget}..."
    )
    result = run_selfplay(config, out_dir, seed=config.sim.seed)

    print(f"wrote checkpoints + stability log to {out_dir}")
    print(
        f"  rounds_completed={result.rounds_completed} "
        f"converged={result.converged} diverged={result.diverged}"
    )
    if result.failure_reason:
        print(f"  note: {result.failure_reason}")
    plot_selfplay_stability(result.round_metrics, out_dir / "stability.png")
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

    reward_history = load_reward_history(run_dir / TRAIN_REWARDS_FILENAME)
    plot_curves(reward_history, run_dir / "curves.png")
    plot_comparison(metrics_by_controller, run_dir / "comparison.png")
    write_attack_trace(config, attack, run_dir / "attack_trace.png")
    print(f"regenerated plots in {run_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rl-edos", description="RL-EDoS command surface")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train-defender", help="train PPO defender vs scripted attacks")
    _add_config_arg(p_train)
    p_train.add_argument(
        "--out", default=None, help="checkpoint output dir (default: models/<run_id>)"
    )
    p_train.set_defaults(func=_train_defender)

    p_eval = sub.add_parser("evaluate", help="run all controllers under identical attacks")
    _add_config_arg(p_eval)
    p_eval.add_argument("--policy", default="none", help="path to a defender policy, or 'none'")
    p_eval.add_argument("--baselines", default="all", help="comma-separated baseline set")
    p_eval.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="one or more episode seeds to average over, e.g. --seeds 0 1 2 "
        "(default: config's sim.seed only, i.e. current single-seed behaviour)",
    )
    p_eval.set_defaults(func=_evaluate)

    p_selfplay = sub.add_parser("selfplay", help="co-evolutionary attacker/defender loop")
    _add_config_arg(p_selfplay)
    p_selfplay.add_argument(
        "--out",
        default=None,
        help="output dir for checkpoints + stability log (default: models/selfplay/<run_id>)",
    )
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
