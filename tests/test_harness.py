"""Evaluation harness: determinism under fixed seeds, artifact schema, RunRecord bookkeeping."""

from __future__ import annotations

import json

import pandas as pd
from rl_edos.baselines.target_tracking import TargetTrackingController
from rl_edos.evaluation.artifacts import write_run_artifacts
from rl_edos.evaluation.evaluator import Evaluator
from rl_edos.evaluation.metrics import RunRecord


def test_evaluator_deterministic_for_fixed_seed(make_config):
    cfg = make_config({"attack": {"mode": "scripted", "pattern": "steady", "intensity": 0.3}})
    controllers = {"target_tracking": TargetTrackingController(cfg.sim)}
    evaluator = Evaluator(cfg)

    metrics_a, _ = evaluator.run(controllers, seeds=[1, 2])["target_tracking"]
    metrics_b, _ = evaluator.run(controllers, seeds=[1, 2])["target_tracking"]

    assert metrics_a.as_dict() == metrics_b.as_dict()


def test_run_records_capture_git_sha_and_seed(make_config):
    cfg = make_config()
    controllers = {"target_tracking": TargetTrackingController(cfg.sim)}
    evaluator = Evaluator(cfg)

    _, records = evaluator.run(controllers, seeds=[5, 6])["target_tracking"]

    assert [r.seed for r in records] == [5, 6]
    assert all(isinstance(r.git_sha, str) and r.git_sha for r in records)


def test_write_run_artifacts_writes_expected_schema(make_config, tmp_path):
    cfg = make_config()
    controllers = {"target_tracking": TargetTrackingController(cfg.sim)}
    evaluator = Evaluator(cfg)
    results = evaluator.run(controllers, seeds=[1])

    run_id = RunRecord.new_run_id()
    out_dir = write_run_artifacts(run_id, cfg, cfg.attack, results, tmp_path)

    for name in (
        "comparison.csv",
        "comparison.png",
        "curves.png",
        "attack_trace.png",
        "config.yaml",
        "summary.json",
    ):
        assert (out_dir / name).is_file()

    df = pd.read_csv(out_dir / "comparison.csv")
    expected_cols = {
        "controller",
        "n_seeds",
        "mean_cost_under_attack",
        "mean_cost_under_attack_std",
        "p95_latency_ms",
        "p95_latency_ms_std",
        "legit_drop_rate",
        "legit_drop_rate_std",
        "overprovision_ratio",
        "overprovision_ratio_std",
        "reward_curve_ref",
    }
    assert expected_cols.issubset(set(df.columns))
    assert "target_tracking" in df["controller"].values
    # a single-seed run reports zero spread, not a missing/NaN std
    row = df[df["controller"] == "target_tracking"].iloc[0]
    assert row["n_seeds"] == 1
    assert row["mean_cost_under_attack_std"] == 0.0

    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["run_id"] == run_id
    record = summary["controllers"]["target_tracking"]["run_records"][0]
    assert record["seed"] == 1
    assert "git_sha" in record


def test_write_run_artifacts_reports_spread_for_a_multi_seed_run(make_config, tmp_path):
    cfg = make_config()
    controllers = {"target_tracking": TargetTrackingController(cfg.sim)}
    evaluator = Evaluator(cfg)
    results = evaluator.run(controllers, seeds=[1, 2, 3])

    out_dir = write_run_artifacts(RunRecord.new_run_id(), cfg, cfg.attack, results, tmp_path)
    df = pd.read_csv(out_dir / "comparison.csv")
    row = df[df["controller"] == "target_tracking"].iloc[0]

    assert row["n_seeds"] == 3
    # target_tracking is deterministic given the seed but different seeds still drive
    # different traffic noise draws, so at least one metric should show real spread
    # rather than every *_std column silently reading 0.0.
    std_cols = [c for c in df.columns if c.endswith("_std")]
    assert any(row[c] != 0.0 for c in std_cols)
