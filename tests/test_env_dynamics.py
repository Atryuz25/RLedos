"""CloudEnv dynamics: warm-up transition, cooldown enforcement, queue/latency, bounds, seeding."""

from __future__ import annotations

import numpy as np
from rl_edos.env.cloud_env import CloudEnv


def test_warmup_transition_moves_warming_to_active(make_config):
    cfg = make_config(
        {"sim": {"instance_warmup_s": 20.0, "control_interval_s": 10.0, "scale_cooldown_s": 5.0}}
    )
    env = CloudEnv(cfg)
    env.reset(seed=1)

    env.step(np.array([5.0], dtype=np.float32))  # t 0->10, launches 4 warming at ready_time=20
    obs, *_ = env.step(np.array([5.0], dtype=np.float32))  # t 10->20, ready_time==20 promotes
    active_instances = obs[3]
    warming_instances = obs[4]
    assert active_instances == 5
    assert warming_instances == 0


def test_cooldown_blocks_scaling_until_elapsed(make_config):
    cfg = make_config(
        {
            "sim": {
                "instance_warmup_s": 1000.0,  # no promotion inside this test's window
                "control_interval_s": 10.0,
                "scale_cooldown_s": 30.0,
                "max_instances": 5,
                "min_instances": 1,
            }
        }
    )
    env = CloudEnv(cfg)
    env.reset(seed=1)

    obs, *_ = env.step(np.array([5.0], dtype=np.float32))  # t 0->10: scale to 5, cooldown until 30
    total = obs[3] + obs[4]
    assert total == 5

    obs, *_ = env.step(np.array([2.0], dtype=np.float32))  # t 10->20: still cooling down
    assert obs[3] + obs[4] == 5

    obs, *_ = env.step(np.array([2.0], dtype=np.float32))  # t 20->30: still cooling down
    assert obs[3] + obs[4] == 5

    obs, *_ = env.step(np.array([2.0], dtype=np.float32))  # t=30: cooldown elapsed, scale applies
    assert obs[3] + obs[4] == 2


def test_action_clipped_to_instance_bounds(make_config):
    cfg = make_config({"sim": {"max_instances": 5, "min_instances": 1, "scale_cooldown_s": 0.0}})
    env = CloudEnv(cfg)
    env.reset(seed=1)

    obs, *_ = env.step(np.array([999.0], dtype=np.float32))
    assert obs[3] + obs[4] == 5

    obs, *_ = env.step(np.array([-999.0], dtype=np.float32))
    assert obs[3] + obs[4] == 1


def test_queue_and_latency_grow_under_sustained_overload(make_config):
    cfg = make_config(
        {
            "sim": {
                "min_instances": 1,
                "max_instances": 1,
                "capacity_rps_per_instance": 5.0,
                "control_interval_s": 10.0,
            },
            "traffic": {
                "legit_pattern": "poisson",
                "base_rate": 20.0,  # arrival >> capacity (5 rps), noise_std=0 -> deterministic
                "noise_std": 0.0,
            },
        }
    )
    env = CloudEnv(cfg)
    env.reset(seed=1)

    queue_lens = []
    latencies = []
    for _ in range(3):
        obs, *_ = env.step(np.array([1.0], dtype=np.float32))
        queue_lens.append(obs[2])
        latencies.append(obs[5])

    assert queue_lens[0] < queue_lens[1] < queue_lens[2]
    assert latencies[0] < latencies[1] < latencies[2]


def test_reset_seed_reproduces_identical_episode(make_config):
    cfg = make_config()
    actions = [np.array([3.0], dtype=np.float32) for _ in range(5)]

    env_a = CloudEnv(cfg)
    obs_a, _ = env_a.reset(seed=123)
    trace_a = [obs_a]
    for action in actions:
        obs, reward, *_ = env_a.step(action)
        trace_a.append(obs)

    env_b = CloudEnv(cfg)
    obs_b, _ = env_b.reset(seed=123)
    trace_b = [obs_b]
    for action in actions:
        obs, reward, *_ = env_b.step(action)
        trace_b.append(obs)

    for a, b in zip(trace_a, trace_b):
        np.testing.assert_array_equal(a, b)


def test_sim_horizon_truncates_episode(make_config):
    cfg = make_config({"sim": {"sim_horizon_steps": 3}})
    env = CloudEnv(cfg)
    env.reset(seed=1)
    for _ in range(2):
        _, _, terminated, truncated, _ = env.step(np.array([1.0], dtype=np.float32))
        assert not terminated
        assert not truncated
    _, _, terminated, truncated, _ = env.step(np.array([1.0], dtype=np.float32))
    assert not terminated
    assert truncated
