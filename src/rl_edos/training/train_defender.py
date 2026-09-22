"""Trainer: trains the Phase 3 RL defender via PPO against a (scripted) attack.

Wraps `training/sb3_utils.py::train_ppo_normalized` -- see that function's
docstring for why observation normalization (`VecNormalize`) is required and
why its stats must be saved alongside the SB3 checkpoint. This module owns the
on-disk checkpoint *directory* layout; `agents/defender.py` owns loading it
back into a callable controller.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import gymnasium as gym

from rl_edos.agents.defender import POLICY_FILENAME, VECNORMALIZE_FILENAME
from rl_edos.config import ExperimentConfig
from rl_edos.env.cloud_env import CloudEnv
from rl_edos.training.sb3_utils import train_ppo_normalized


class Trainer:
    """`Trainer.train(config, out_dir) -> policy_ckpt` per docs/02_ARCHITECTURE.md."""

    def train(
        self,
        config: ExperimentConfig,
        out_dir: str | Path,
        attack_traffic_fn: Callable[[float], float] | None = None,
        seed: int | None = None,
    ) -> Path:
        """Train the defender; returns `out_dir`, the checkpoint directory.

        `out_dir` contains `policy.zip` (SB3 PPO) and `vecnormalize.pkl`
        (running obs stats) -- both required for a correct `evaluate`. Raises
        `training.sb3_utils.TrainingDivergedError` (and writes nothing) if
        training produces non-finite policy parameters.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        seed = seed if seed is not None else config.sim.seed

        def make_env() -> gym.Env:
            return CloudEnv(config, attack_traffic_fn=attack_traffic_fn)

        model, vecnorm = train_ppo_normalized(
            make_env,
            total_timesteps=config.agent.total_timesteps,
            policy_net=config.agent.policy_net,
            seed=seed,
            tensorboard_log=str(out_dir / "tensorboard_logs"),
        )
        model.save(out_dir / POLICY_FILENAME)
        vecnorm.save(str(out_dir / VECNORMALIZE_FILENAME))
        return out_dir
