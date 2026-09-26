"""RLDefenderController: wraps a trained Phase 3 PPO checkpoint into the same
callable action contract every baseline exposes (`EnvState -> np.ndarray`).

A checkpoint is a directory containing `policy.zip` (the SB3 PPO model) and
`vecnormalize.pkl` (the `VecNormalize` running obs stats saved alongside it,
per `training/sb3_utils.py::train_ppo_normalized`) -- both are required to
evaluate the policy correctly. Evaluating with the checkpoint but without its
`vecnormalize.pkl` would silently feed the policy raw, unnormalized
observations after it was trained on normalized ones: it wouldn't crash, it
would just be quietly miscalibrated.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from rl_edos.config import SimConfig
from rl_edos.env.state import EnvState
from rl_edos.training.sb3_utils import normalize_obs, rescale_action

POLICY_FILENAME = "policy.zip"
VECNORMALIZE_FILENAME = "vecnormalize.pkl"
TRAIN_REWARDS_FILENAME = "train_rewards.csv"


class RLDefenderController:
    """Callable defender: normalizes obs, predicts, rescales/clips to a real instance count."""

    def __init__(self, model: PPO, sim: SimConfig, vecnorm: VecNormalize | None = None) -> None:
        self.model = model
        self.sim = sim
        self.vecnorm = vecnorm

    def __call__(self, state: EnvState) -> np.ndarray:
        obs = normalize_obs(state.to_obs(), self.vecnorm)
        action, _ = self.model.predict(obs, deterministic=True)
        real_action = rescale_action(action, self.sim.min_instances, self.sim.max_instances)
        clipped = np.clip(real_action, self.sim.min_instances, self.sim.max_instances)
        return np.asarray(clipped, dtype=np.float32)


def load_defender(policy_dir: str | Path, sim: SimConfig) -> RLDefenderController:
    """Load a `Trainer.train()` checkpoint directory into a callable controller.

    `vecnormalize.pkl` is loaded via plain `pickle.load` rather than SB3's
    `VecNormalize.load()` classmethod: that classmethod re-attaches a live
    `VecEnv`, which inference doesn't need -- only the pickled object's running
    obs mean/var are used, through `normalize_obs`. Missing the file is
    tolerated (falls back to no normalization) so a Phase 2-style checkpoint
    without one still loads, just unnormalized.
    """
    policy_dir = Path(policy_dir)
    model = PPO.load(policy_dir / POLICY_FILENAME)
    vecnorm_path = policy_dir / VECNORMALIZE_FILENAME
    vecnorm = None
    if vecnorm_path.is_file():
        with open(vecnorm_path, "rb") as fh:
            vecnorm = pickle.load(fh)
    return RLDefenderController(model, sim, vecnorm)
