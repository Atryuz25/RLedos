"""Learned-attacker evaluation wiring (F2): `attack.mode = "learned"` routes every
controller in `Evaluator.run()` against a trained self-play attacker checkpoint
instead of a scripted pattern, via `agents/attacker.py::load_attacker`.
"""

from __future__ import annotations

from pathlib import Path

from rl_edos.baselines.target_tracking import TargetTrackingController
from rl_edos.env.cloud_env import CloudEnv
from rl_edos.evaluation.evaluator import Evaluator
from rl_edos.training.sb3_utils import train_ppo_normalized
from rl_edos.training.selfplay_envs import AttackerSelfPlayEnv


def _train_tiny_attacker_checkpoint(cfg, out_dir: Path) -> Path:
    """Builds a checkpoint in the exact format training/selfplay.py saves: a
    directory with policy.zip + vecnormalize.pkl, trained against a frozen
    (also tiny) defender -- without running the full selfplay loop.
    """
    defender_model, defender_vecnorm = train_ppo_normalized(
        lambda: CloudEnv(cfg), total_timesteps=32, policy_net=[8], seed=1
    )
    attacker_model, attacker_vecnorm = train_ppo_normalized(
        lambda: AttackerSelfPlayEnv(cfg, defender_model, defender_vecnorm),
        total_timesteps=32,
        policy_net=[8],
        seed=2,
    )
    ckpt_dir = out_dir / "attacker_ckpt"
    ckpt_dir.mkdir()
    attacker_model.save(ckpt_dir / "policy.zip")
    attacker_vecnorm.save(str(ckpt_dir / "vecnormalize.pkl"))
    return ckpt_dir


def test_evaluate_runs_end_to_end_against_a_learned_attacker(make_config, tmp_path):
    bootstrap_cfg = make_config(
        {"sim": {"sim_horizon_steps": 5}, "attack": {"mode": "scripted", "evasion_budget": 1.0}}
    )
    ckpt_dir = _train_tiny_attacker_checkpoint(bootstrap_cfg, tmp_path)

    learned_cfg = make_config(
        {
            "sim": {"sim_horizon_steps": 5},
            "attack": {
                "mode": "learned",
                "evasion_budget": 1.0,
                "attacker_checkpoint": str(ckpt_dir),
            },
        }
    )
    learned_cfg.attack.validate_learned_attacker()  # must not raise: checkpoint is real

    evaluator = Evaluator(learned_cfg)
    results = evaluator.run({"target_tracking": TargetTrackingController(learned_cfg.sim)})

    assert "target_tracking" in results
    metrics, records = results["target_tracking"]
    assert metrics.mean_cost_under_attack >= 0.0
    assert records[0].attack.mode == "learned"


def test_scripted_mode_is_unaffected_by_learned_attacker_plumbing(make_config):
    # Regression: attack.mode defaults to "scripted" and must take the exact
    # same code path as before F2 -- no attacker checkpoint loaded, no crash,
    # same behaviour the pre-existing harness tests already pin down.
    cfg = make_config({"attack": {"mode": "scripted", "pattern": "steady", "intensity": 0.3}})
    evaluator = Evaluator(cfg)

    results_a = evaluator.run({"target_tracking": TargetTrackingController(cfg.sim)})
    results_b = evaluator.run({"target_tracking": TargetTrackingController(cfg.sim)})

    assert results_a["target_tracking"][0].as_dict() == results_b["target_tracking"][0].as_dict()
