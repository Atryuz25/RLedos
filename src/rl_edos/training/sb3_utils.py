"""Thin Stable-Baselines3 PPO wrapper, shared by security_blind_rl (Phase 2) and the
joint-reward defender (Phase 3). No hand-rolled RL — this only configures SB3's PPO.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO


def train_ppo(env: gym.Env, total_timesteps: int, policy_net: list[int], seed: int) -> PPO:
    """Train a PPO policy on `env` for `total_timesteps`, MLP hidden sizes `policy_net`.

    `env`'s action space is rescaled to [-1, 1] before training -- SB3 recommends a
    symmetric, normalized action space for PPO's Gaussian policy. Callers must
    un-rescale `model.predict()`'s output back to the real action range with
    `rescale_action` before using it as a scaling decision.

    `torch.use_deterministic_algorithms(True)` makes "same config + seed -> same
    result" an enforced guarantee rather than an incidental property of running
    single-process on CPU: it raises loudly if a non-deterministic op is ever
    pulled in (e.g. by a GPU move or an architecture change), instead of silently
    producing irreproducible training runs.
    """
    torch.use_deterministic_algorithms(True)
    wrapped = gym.wrappers.RescaleAction(env, min_action=-1.0, max_action=1.0)
    model = PPO(
        "MlpPolicy",
        wrapped,
        policy_kwargs={"net_arch": policy_net},
        seed=seed,
        verbose=0,
    )
    model.learn(total_timesteps=total_timesteps)
    return model


def rescale_action(action: np.ndarray, low: float, high: float) -> np.ndarray:
    """Map a `train_ppo`-trained policy's [-1, 1] action to the real [low, high] range."""
    return low + (np.asarray(action, dtype=np.float32) + 1.0) / 2.0 * (high - low)


def load_ppo(path: str) -> PPO:
    """Load a saved PPO checkpoint (SB3 .zip)."""
    return PPO.load(path)
