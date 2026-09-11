"""train_ppo's action-space normalization and the rescale_action helper that undoes it."""

from __future__ import annotations

import numpy as np
from rl_edos.env.cloud_env import CloudEnv
from rl_edos.training.sb3_utils import rescale_action, train_ppo


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
