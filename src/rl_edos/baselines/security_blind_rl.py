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
from rl_edos.env.cloud_env import CloudEnv
from rl_edos.env.state import EnvState
from rl_edos.training.sb3_utils import train_ppo

# EnvState.to_obs() order ends with detection_score; masking it out is what
# makes this controller "security-blind".
BLIND_OBS_DIM = 7


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

    SB3's continuous PPO policy is an unbounded Gaussian, so raw predictions
    aren't guaranteed to land inside the action space; clip here so this
    controller honours the same bounded-action contract every baseline does.
    """

    def __init__(self, model: PPO, sim: SimConfig) -> None:
        self.model = model
        self.sim = sim

    def __call__(self, state: EnvState) -> np.ndarray:
        obs = state.to_obs()[:BLIND_OBS_DIM]
        action, _ = self.model.predict(obs, deterministic=True)
        clipped = np.clip(action, self.sim.min_instances, self.sim.max_instances)
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
