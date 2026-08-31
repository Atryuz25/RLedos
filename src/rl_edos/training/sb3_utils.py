"""Thin Stable-Baselines3 PPO wrapper, shared by security_blind_rl (Phase 2) and the
joint-reward defender (Phase 3). No hand-rolled RL — this only configures SB3's PPO.
"""

from __future__ import annotations

import gymnasium as gym
from stable_baselines3 import PPO


def train_ppo(env: gym.Env, total_timesteps: int, policy_net: list[int], seed: int) -> PPO:
    """Train a PPO policy on `env` for `total_timesteps`, MLP hidden sizes `policy_net`."""
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs={"net_arch": policy_net},
        seed=seed,
        verbose=0,
    )
    model.learn(total_timesteps=total_timesteps)
    return model


def load_ppo(path: str) -> PPO:
    """Load a saved PPO checkpoint (SB3 .zip)."""
    return PPO.load(path)
