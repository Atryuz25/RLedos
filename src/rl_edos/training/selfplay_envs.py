"""Self-play wrapper environments (Phase 4): one agent trains while the other
is frozen inside the env dynamics.

`CloudEnv.step` normally samples attack rate from a time-indexed scripted
function. These wrappers instead have the *frozen* opponent's PPO model
predict an action from the current observation each step and feed that into
`CloudEnv.step(..., attack_rate_override=...)` -- so from the trainee's SB3
point of view this is an ordinary single-agent `gymnasium.Env`, and the
opponent is just part of the environment's transition dynamics for this round.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from rl_edos.agents.reward import attacker_reward
from rl_edos.config import ConfigError, ExperimentConfig
from rl_edos.env.cloud_env import CloudEnv
from rl_edos.training.sb3_utils import normalize_obs, rescale_action


class DefenderSelfPlayEnv(gym.Env):
    """Defender's side: action = scaling decision; attack rate comes from a frozen attacker."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: ExperimentConfig,
        attacker_model: PPO,
        attacker_vecnorm: VecNormalize | None,
    ) -> None:
        super().__init__()
        self._cloud = CloudEnv(config)
        self.observation_space = self._cloud.observation_space
        self.action_space = self._cloud.action_space
        self._attacker_model = attacker_model
        self._attacker_vecnorm = attacker_vecnorm
        self._evasion_budget = config.attack.evasion_budget
        self._base_rate = config.traffic.base_rate
        self._last_obs: np.ndarray | None = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self._cloud.reset(seed=seed)
        self._last_obs = obs
        return obs, info

    def step(self, action: np.ndarray):
        assert self._last_obs is not None, "reset() must be called before step()"
        attacker_obs = normalize_obs(self._last_obs, self._attacker_vecnorm)
        raw_attacker_action, _ = self._attacker_model.predict(attacker_obs, deterministic=True)
        intensity = float(
            np.clip(
                rescale_action(raw_attacker_action, 0.0, self._evasion_budget)[0],
                0.0,
                self._evasion_budget,
            )
        )
        attack_rate = intensity * self._base_rate

        obs, reward, terminated, truncated, info = self._cloud.step(
            action, attack_rate_override=attack_rate
        )
        self._last_obs = obs
        return obs, reward, terminated, truncated, info


class AttackerSelfPlayEnv(gym.Env):
    """Attacker's perspective: action = shaping intensity; scaling comes from a frozen defender."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: ExperimentConfig,
        defender_model: PPO,
        defender_vecnorm: VecNormalize | None,
    ) -> None:
        super().__init__()
        if config.attack.evasion_budget <= 0:
            raise ConfigError(
                "self-play requires attack.evasion_budget > 0 to define the "
                "learned attacker's action range"
            )
        self._cloud = CloudEnv(config)
        self.observation_space = self._cloud.observation_space
        self.action_space = spaces.Box(
            low=0.0, high=float(config.attack.evasion_budget), shape=(1,), dtype=np.float32
        )
        self._defender_model = defender_model
        self._defender_vecnorm = defender_vecnorm
        self._sim = config.sim
        self._base_rate = config.traffic.base_rate
        self._reward_weights = config.agent.reward_weights
        self._detection_penalty = config.detection.penalty
        self._last_obs: np.ndarray | None = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self._cloud.reset(seed=seed)
        self._last_obs = obs
        return obs, info

    def step(self, action: np.ndarray):
        assert self._last_obs is not None, "reset() must be called before step()"
        defender_obs = normalize_obs(self._last_obs, self._defender_vecnorm)
        raw_defender_action, _ = self._defender_model.predict(defender_obs, deterministic=True)
        defender_action = np.clip(
            rescale_action(raw_defender_action, self._sim.min_instances, self._sim.max_instances),
            self._sim.min_instances,
            self._sim.max_instances,
        ).astype(np.float32)

        intensity = float(np.clip(action[0], 0.0, self.action_space.high[0]))
        attack_rate = intensity * self._base_rate

        obs, _defender_reward, terminated, truncated, info = self._cloud.step(
            defender_action, attack_rate_override=attack_rate
        )
        self._last_obs = obs
        reward = attacker_reward(
            info["cost"], info["detection_score"], self._detection_penalty, self._reward_weights
        )
        return obs, reward, terminated, truncated, info
