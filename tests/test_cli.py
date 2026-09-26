"""CLI surface: argument parsing, `evaluate --seeds` wiring (F5), and real
learning-curve data flowing from `train-defender` into `evaluate` (F3).
"""

from __future__ import annotations

import copy

import pandas as pd
import yaml
from rl_edos.cli import build_parser, main

_MINIMAL_CONFIG = {
    "sim": {
        "control_interval_s": 10.0,
        "sim_horizon_steps": 20,
        "instance_warmup_s": 20.0,
        "scale_cooldown_s": 30.0,
        "max_instances": 5,
        "min_instances": 1,
        "seed": 7,
        "capacity_rps_per_instance": 10.0,
        "base_latency_ms": 5.0,
    },
    "billing": {
        "price_per_instance_second": 0.00001,
        "billing_granularity": "per_second",
        "min_service_charge": 0.0,
    },
    "traffic": {
        "legit_pattern": "poisson",
        "base_rate": 20.0,
        "diurnal_amplitude": 0.0,
        "noise_std": 0.0,
    },
    "attack": {"mode": "scripted", "pattern": "steady", "intensity": 0.0, "evasion_budget": 0.0},
    "detection": {"type": "rate_threshold", "window": 5, "threshold": 50.0, "penalty": 1.0},
}


def test_evaluate_parser_accepts_seeds_flag():
    parser = build_parser()

    args = parser.parse_args(["evaluate", "--config", "cfg.yaml", "--seeds", "0", "1", "2"])
    assert args.seeds == [0, 1, 2]


def test_evaluate_parser_defaults_seeds_to_none():
    parser = build_parser()

    args = parser.parse_args(["evaluate", "--config", "cfg.yaml"])
    assert args.seeds is None


def test_evaluate_cli_with_seeds_writes_multi_seed_comparison_csv(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(_MINIMAL_CONFIG))

    monkeypatch.chdir(tmp_path)
    exit_code = main(
        [
            "evaluate",
            "--config",
            str(config_path),
            "--baselines",
            "target_tracking",
            "--seeds",
            "1",
            "2",
        ]
    )
    assert exit_code == 0

    results_dirs = list((tmp_path / "results").iterdir())
    assert len(results_dirs) == 1
    df = pd.read_csv(results_dirs[0] / "comparison.csv")
    row = df[df["controller"] == "target_tracking"].iloc[0]
    assert row["n_seeds"] == 2


def test_train_then_evaluate_produces_a_real_learning_curve(tmp_path, monkeypatch):
    config = copy.deepcopy(_MINIMAL_CONFIG)
    config["sim"]["sim_horizon_steps"] = 5
    config["agent"] = {"total_timesteps": 64, "policy_net": [8]}
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    ckpt_dir = tmp_path / "ckpt"

    monkeypatch.chdir(tmp_path)
    assert main(["train-defender", "--config", str(config_path), "--out", str(ckpt_dir)]) == 0
    assert (ckpt_dir / "train_rewards.csv").is_file()

    assert (
        main(
            [
                "evaluate",
                "--config",
                str(config_path),
                "--policy",
                str(ckpt_dir),
                "--baselines",
                "target_tracking",
            ]
        )
        == 0
    )

    out_dir = next((tmp_path / "results").iterdir())
    assert (out_dir / "train_rewards.csv").is_file()  # copied alongside the run's artifacts
    df = pd.read_csv(out_dir / "comparison.csv")
    ref = df[df["controller"] == "rl_defender"].iloc[0]["reward_curve_ref"]
    assert isinstance(ref, str) and ref  # non-empty, not NaN -- real history was found
