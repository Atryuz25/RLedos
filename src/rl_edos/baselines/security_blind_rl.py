"""security_blind_rl: PPO trained on the joint cost+latency reward, with NO detection-score
observation — a security-blind comparison point per docs/PHASE_2_BASELINES_AND_EVAL.md.
"""

from __future__ import annotations

from typing import Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO

from rl_edos.config import ExperimentConfig, SimConfig
from rl_edos.env.cloud_env import OBS_DIM, CloudEnv
from rl_edos.env.state import EnvState
from rl_edos.training.sb3_utils import rescale_action, train_ppo

# EnvState.to_obs() order ends with detection_score; masking it out (dropping the
# last dimension) is what makes this controller "security-blind".
BLIND_OBS_DIM = OBS_DIM - 1


class _BlindObsWrapper(gym.ObservationWrapper):
    """Drops the detection_score dimension from CloudEnv's observation."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=0.0, high=np.inf, shape=(BLIND_OBS_DIM,), dtype=np.float32
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return obs[:BLIND_OBS_DIM]


class SecurityBlindController:
    """Wraps a PPO model trained without detection awareness; same action contract as baselines.

    `train_ppo` trains against a [-1, 1]-rescaled action space (see
    `training/sb3_utils.py`), so predictions must be mapped back to real
    instance counts with `rescale_action` before use. SB3's continuous PPO
    policy is also an unbounded Gaussian, so raw predictions aren't guaranteed
    to land inside [-1, 1] either; clip after rescaling so this controller
    honours the same bounded-action contract every baseline does.
    """

    def __init__(self, model: PPO, sim: SimConfig) -> None:
        self.model = model
        self.sim = sim

    def __call__(self, state: EnvState) -> np.ndarray:
        obs = state.to_obs()[:BLIND_OBS_DIM]
        action, _ = self.model.predict(obs, deterministic=True)
        real_action = rescale_action(action, self.sim.min_instances, self.sim.max_instances)
        clipped = np.clip(real_action, self.sim.min_instances, self.sim.max_instances)
        return np.asarray(clipped, dtype=np.float32)


def train_security_blind(
    config: ExperimentConfig,
    attack_traffic_fn: Callable[[float], float] | None = None,
    seed: int | None = None,
) -> SecurityBlindController:
    """Train the security-blind PPO baseline for one experiment config."""
    env = _BlindObsWrapper(CloudEnv(config, attack_traffic_fn=attack_traffic_fn))
    model = train_ppo(
        env,
        total_timesteps=config.agent.total_timesteps,
        policy_net=config.agent.policy_net,
        seed=seed if seed is not None else config.sim.seed,
    )
    return SecurityBlindController(model, config.sim)
