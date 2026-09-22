"""Self-play (Phase 4): round-trip checkpointing, stability logging, non-convergence path."""

from __future__ import annotations

import json

import numpy as np
import pytest
from rl_edos.config import ConfigError
from rl_edos.env.cloud_env import CloudEnv
from rl_edos.training.sb3_utils import train_ppo_normalized
from rl_edos.training.selfplay import run_selfplay
from rl_edos.training.selfplay_envs import AttackerSelfPlayEnv


def _selfplay_config(make_config, **selfplay_overrides):
    overrides = {
        "sim": {"sim_horizon_steps": 5},
        "attack": {"mode": "learned", "pattern": "steady", "intensity": 0.0, "evasion_budget": 1.0},
        "agent": {"total_timesteps": 16, "policy_net": [8]},
        "selfplay": {"rounds": 1, "round_timesteps": 16, **selfplay_overrides},
    }
    return make_config(overrides)


def test_selfplay_requires_positive_evasion_budget(make_config, tmp_path):
    cfg = make_config({"attack": {"evasion_budget": 0.0}})
    with pytest.raises(ConfigError, match="evasion_budget"):
        run_selfplay(cfg, tmp_path / "out")


def test_attacker_selfplay_env_rejects_zero_evasion_budget(make_config):
    cfg = make_config({"attack": {"evasion_budget": 0.0}})
    with pytest.raises(ConfigError, match="evasion_budget"):
        AttackerSelfPlayEnv(cfg, defender_model=None, defender_vecnorm=None)


def test_selfplay_round_trip_checkpoints_both_agents_and_logs_stability(make_config, tmp_path):
    cfg = _selfplay_config(make_config, convergence_tolerance=1.0, convergence_patience=1)
    out_dir = tmp_path / "selfplay"

    result = run_selfplay(cfg, out_dir, seed=1)

    assert (out_dir / "defender" / "policy.zip").is_file()
    assert (out_dir / "defender" / "vecnormalize.pkl").is_file()
    assert (out_dir / "attacker" / "policy.zip").is_file()
    assert (out_dir / "attacker" / "vecnormalize.pkl").is_file()
    assert (out_dir / "stability.json").is_file()

    # round 0 (bootstrap) + at least one alternation round
    assert len(result.round_metrics) >= 2
    assert result.round_metrics[0]["round"] == 0
    for m in result.round_metrics:
        assert 0.0 <= m["detection_trip_rate"] <= 1.0
        assert m["mean_cost_under_attack"] >= 0.0

    on_disk = json.loads((out_dir / "stability.json").read_text())
    assert on_disk["round_metrics"] == result.round_metrics
    assert on_disk["converged"] or on_disk["failure_reason"] is not None


def test_selfplay_non_convergence_produces_failure_artifact_not_a_crash(make_config, tmp_path):
    # tolerance=0 makes exact convergence essentially impossible in one round,
    # so this exercises the "exhausted rounds without converging" path.
    cfg = _selfplay_config(make_config, convergence_tolerance=0.0, convergence_patience=5)
    out_dir = tmp_path / "selfplay_no_converge"

    result = run_selfplay(cfg, out_dir, seed=2)

    assert result.converged is False
    assert result.diverged is False
    assert result.failure_reason is not None
    assert "did not converge" in result.failure_reason
    # still a complete, usable deliverable -- not a crash
    assert (out_dir / "defender" / "policy.zip").is_file()
    assert (out_dir / "attacker" / "policy.zip").is_file()


def test_attacker_selfplay_env_action_space_matches_evasion_budget(make_config):
    cfg = make_config(
        {
            "sim": {"sim_horizon_steps": 5},
            "attack": {"mode": "learned", "evasion_budget": 2.5},
            "agent": {"total_timesteps": 16, "policy_net": [8]},
        }
    )
    frozen_defender, frozen_defender_vecnorm = train_ppo_normalized(
        lambda: CloudEnv(cfg),
        total_timesteps=16,
        policy_net=[8],
        seed=1,
    )
    env = AttackerSelfPlayEnv(cfg, frozen_defender, frozen_defender_vecnorm)

    assert env.action_space.low[0] == 0.0
    assert env.action_space.high[0] == 2.5

    obs, _ = env.reset(seed=1)
    next_obs, reward, terminated, truncated, info = env.step(np.array([1.0], dtype=np.float32))
    assert next_obs.shape == obs.shape
    assert isinstance(reward, float)
    assert "cost" in info and "detection_score" in info
