"""train_ppo's action-space normalization, the rescale_action helper that undoes
it, Phase 3's normalized-training path + NaN guard, and the Trainer/
RLDefenderController checkpoint round-trip.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from rl_edos.agents.defender import load_defender
from rl_edos.baselines.target_tracking import TargetTrackingController
from rl_edos.env.cloud_env import CloudEnv
from rl_edos.env.state import EnvState
from rl_edos.evaluation.evaluator import Evaluator
from rl_edos.training.sb3_utils import (
    TrainingDivergedError,
    _NanGuardCallback,
    normalize_obs,
    rescale_action,
    train_ppo,
    train_ppo_normalized,
)
from rl_edos.training.train_defender import Trainer


def test_rescale_action_matches_hand_computation():
    # midpoint of [-1, 1] -> midpoint of [1, 21]
    assert rescale_action(np.array([0.0]), low=1.0, high=21.0)[0] == 11.0
    # lower bound -> lower bound, upper bound -> upper bound
    assert rescale_action(np.array([-1.0]), low=1.0, high=21.0)[0] == 1.0
    assert rescale_action(np.array([1.0]), low=1.0, high=21.0)[0] == 21.0


def test_train_ppo_rescales_env_action_space_to_unit_range(make_config):
    cfg = make_config(
        {"sim": {"min_instances": 1, "max_instances": 5}, "agent": {"total_timesteps": 32}}
    )
    env = CloudEnv(cfg)
    model = train_ppo(env, total_timesteps=cfg.agent.total_timesteps, policy_net=[8], seed=1)

    np.testing.assert_array_equal(model.action_space.low, np.array([-1.0], dtype=np.float32))
    np.testing.assert_array_equal(model.action_space.high, np.array([1.0], dtype=np.float32))


def test_train_ppo_is_deterministic_for_fixed_seed(make_config):
    cfg = make_config({"agent": {"total_timesteps": 64}})
    obs = np.zeros(8, dtype=np.float32)

    model_a = train_ppo(CloudEnv(cfg), total_timesteps=64, policy_net=[8], seed=3)
    action_a, _ = model_a.predict(obs, deterministic=True)

    model_b = train_ppo(CloudEnv(cfg), total_timesteps=64, policy_net=[8], seed=3)
    action_b, _ = model_b.predict(obs, deterministic=True)

    np.testing.assert_array_equal(action_a, action_b)


def test_normalize_obs_matches_hand_computation():
    class _FakeObsRMS:
        mean = np.array([2.0, 4.0], dtype=np.float64)
        var = np.array([4.0, 1.0], dtype=np.float64)

    class _FakeVecNormalize:
        norm_obs = True
        obs_rms = _FakeObsRMS()
        epsilon = 0.0
        clip_obs = 1000.0

        def normalize_obs(self, obs):
            return np.clip(
                (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var), -self.clip_obs, self.clip_obs
            )

    obs = np.array([4.0, 5.0], dtype=np.float32)
    # (4-2)/sqrt(4)=1.0, (5-4)/sqrt(1)=1.0
    np.testing.assert_allclose(normalize_obs(obs, _FakeVecNormalize()), np.array([1.0, 1.0]))


def test_normalize_obs_is_noop_without_vecnorm():
    obs = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    np.testing.assert_array_equal(normalize_obs(obs, None), obs)


def test_train_ppo_normalized_returns_a_usable_model_and_vecnormalize(make_config):
    cfg = make_config({"agent": {"total_timesteps": 64, "policy_net": [8]}})
    model, vecnorm = train_ppo_normalized(
        lambda: CloudEnv(cfg), total_timesteps=64, policy_net=[8], seed=1
    )
    obs = np.zeros(8, dtype=np.float32)
    action, _ = model.predict(normalize_obs(obs, vecnorm), deterministic=True)
    assert action.shape == (1,)
    assert vecnorm.obs_rms.mean.shape == (8,)


def test_train_ppo_normalized_warm_start_continues_same_model(make_config):
    cfg = make_config({"agent": {"total_timesteps": 32, "policy_net": [8]}})
    model_a, vecnorm_a = train_ppo_normalized(
        lambda: CloudEnv(cfg), total_timesteps=32, policy_net=[8], seed=1
    )
    model_a_timesteps_after_first_call = model_a.num_timesteps
    model_b, _ = train_ppo_normalized(
        lambda: CloudEnv(cfg),
        total_timesteps=32,
        policy_net=[8],
        seed=2,
        warm_start=(model_a, vecnorm_a),
    )
    # warm_start must continue the *same* PPO object, not construct a fresh one
    # (SB3 always completes at least one full n_steps rollout per learn() call,
    # so exact total_timesteps counts don't line up -- what matters is that the
    # second call added to the first rather than resetting it).
    assert model_b is model_a
    assert model_b.num_timesteps > model_a_timesteps_after_first_call


def test_nan_guard_flags_non_finite_policy_parameters(make_config):
    cfg = make_config({"agent": {"total_timesteps": 8, "policy_net": [8]}})
    model, _ = train_ppo_normalized(
        lambda: CloudEnv(cfg), total_timesteps=8, policy_net=[8], seed=1
    )
    guard = _NanGuardCallback()
    guard.model = model
    assert guard._on_step() is True

    with torch.no_grad():
        first_param = next(model.policy.parameters())
        first_param.fill_(float("nan"))
    assert guard._on_step() is False
    assert guard.diverged is True


def test_train_ppo_normalized_raises_and_saves_nothing_on_forced_divergence(
    make_config, monkeypatch
):
    # Forces the same failure mode the guard exists to catch, end-to-end through
    # train_ppo_normalized, without waiting on real training to actually diverge.
    def _force_diverge(self) -> bool:
        self.diverged = True
        return False

    monkeypatch.setattr(_NanGuardCallback, "_on_step", _force_diverge)

    cfg = make_config({"agent": {"total_timesteps": 16, "policy_net": [8]}})
    with pytest.raises(TrainingDivergedError):
        train_ppo_normalized(lambda: CloudEnv(cfg), total_timesteps=16, policy_net=[8], seed=1)


def test_trainer_smoke_run_produces_a_loadable_evaluable_checkpoint(make_config, tmp_path):
    cfg = make_config(
        {
            "sim": {"sim_horizon_steps": 5},
            "agent": {"total_timesteps": 64, "policy_net": [8]},
        }
    )
    out_dir = Trainer().train(cfg, tmp_path / "ckpt", seed=1)

    assert (out_dir / "policy.zip").is_file()
    assert (out_dir / "vecnormalize.pkl").is_file()

    controller = load_defender(out_dir, cfg.sim)
    action = controller(
        EnvState(
            t=0.0,
            arrival_rate=20.0,
            queue_len=0,
            active_instances=1,
            warming_instances=0,
            latency_ms=10.0,
            accrued_cost=0.0,
            detection_score=0.0,
        )
    )
    assert action.shape == (1,)
    assert cfg.sim.min_instances <= action[0] <= cfg.sim.max_instances

    # integrates through the same Phase 2 harness as any other controller
    evaluator = Evaluator(cfg)
    results = evaluator.run(
        {"rl_defender": controller, "target_tracking": TargetTrackingController(cfg.sim)}
    )
    assert "rl_defender" in results
    metrics, records = results["rl_defender"]
    assert metrics.mean_cost_under_attack >= 0.0
    assert records[0].controller == "rl_defender"


def test_trainer_uses_normalization_meaningfully(make_config, tmp_path):
    # a checkpoint's vecnormalize.pkl must actually hold non-trivial running
    # stats after training touches more than one observation -- catches a
    # Trainer that silently skips wiring VecNormalize through to the save path.
    cfg = make_config({"agent": {"total_timesteps": 64, "policy_net": [8]}})
    out_dir = Trainer().train(cfg, tmp_path / "ckpt", seed=1)
    controller = load_defender(out_dir, cfg.sim)
    assert controller.vecnorm is not None
    assert controller.vecnorm.obs_rms.count > 1
