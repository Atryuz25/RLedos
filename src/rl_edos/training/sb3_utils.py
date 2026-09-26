"""Thin Stable-Baselines3 PPO wrapper, shared by security_blind_rl (Phase 2), the
joint-reward defender (Phase 3), and the self-play attacker/defender (Phase 4).
No hand-rolled RL — this only configures SB3's PPO.
"""

from __future__ import annotations

import importlib.util
import warnings
from typing import Callable

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


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


class TrainingDivergedError(Exception):
    """Raised when PPO training produces non-finite policy parameters.

    Per docs/11_CLAUDE_CODE_RULES.md ("no silent failures"), a divergent run
    must never be persisted as a valid checkpoint -- callers of
    `train_ppo_normalized` get this exception instead of a `.zip` to save.
    """


class _NanGuardCallback(BaseCallback):
    """Stops training and flags `diverged` the moment any policy parameter goes non-finite."""

    def __init__(self) -> None:
        super().__init__()
        self.diverged = False

    def _on_step(self) -> bool:
        for param in self.model.policy.parameters():
            if not torch.isfinite(param).all():
                self.diverged = True
                return False
        return True


class _RewardHistoryCallback(BaseCallback):
    """Appends one entry per completed training episode to `history`.

    Relies on `stable_baselines3.common.monitor.Monitor` wrapping the env
    (done unconditionally in `train_ppo_normalized`), which adds an
    `info["episode"] = {"r": total_reward, "l": length, "t": elapsed_s}`
    entry to `infos` the step an episode ends.
    """

    def __init__(self, history: list[dict]) -> None:
        super().__init__()
        self._history = history

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            episode = info.get("episode")
            if episode is not None:
                self._history.append(
                    {
                        "timestep": self.num_timesteps,
                        "episode_reward": float(episode["r"]),
                        "episode_length": int(episode["l"]),
                    }
                )
        return True


def normalize_obs(obs: np.ndarray, vecnorm: VecNormalize | None) -> np.ndarray:
    """Apply a `VecNormalize`'s running obs mean/std to a single raw observation.

    No-op if `vecnorm` is None (e.g. a controller with no saved normalization,
    or Phase 2's `security_blind_rl`, which predates this mechanism). Used both
    at inference time (`agents/defender.py`, `agents/attacker.py`) and inside
    the Phase 4 self-play wrapper envs to normalize a frozen opponent's input
    the same way it saw observations during its own training.
    """
    if vecnorm is None:
        return obs
    return vecnorm.normalize_obs(obs.reshape(1, -1)).astype(np.float32)[0]


def train_ppo_normalized(
    make_env: Callable[[], gym.Env],
    total_timesteps: int,
    policy_net: list[int],
    seed: int,
    tensorboard_log: str | None = None,
    warm_start: tuple[PPO, VecNormalize] | None = None,
    reward_history_out: list[dict] | None = None,
) -> tuple[PPO, VecNormalize]:
    """Train PPO with normalized observations: `VecNormalize(norm_obs=True, norm_reward=False)`.

    `CloudEnv`'s observation space is `Box(low=0, high=inf)` on every dimension
    (arrival rate, queue length, accrued cost, ...) -- unbounded observations
    destabilise PPO's value-function learning (flagged by
    `gymnasium.utils.env_checker.check_env`). `norm_reward` is deliberately
    `False`: normalizing the reward would rescale the cost/latency balance the
    MVP claim is measured on (see docs/PHASE_3_RL_DEFENDER.md item 2).

    `VecNormalize`'s running obs stats are returned alongside the model because
    they live outside the SB3 `.zip` format and must be saved/loaded together
    (`VecNormalize.save`) or evaluation silently normalizes with the wrong
    statistics.

    `warm_start=(model, vecnorm)` continues training that model on a *new* env
    (a different frozen opponent, in self-play) rather than starting fresh --
    used by `training/selfplay.py` so each round builds on the last instead of
    re-learning from scratch. The new VecNormalize inherits the previous
    round's running obs stats before continuing.

    The env is always wrapped in SB3's `Monitor` so per-episode reward/length are
    tracked; when `reward_history_out` is given, it is mutated in place with one
    `{"timestep", "episode_reward", "episode_length"}` entry per completed
    episode (used by `training/train_defender.py::Trainer` to persist real
    learning-curve data instead of a placeholder -- see `docs/03_DATA_SCHEMAS.md`'s
    `reward_curve_ref`). Callers that don't need it (self-play's per-round calls)
    simply omit it.

    Raises `TrainingDivergedError` (and returns nothing to save) if any policy
    parameter goes non-finite during training.
    """
    torch.use_deterministic_algorithms(True)
    if tensorboard_log is not None and importlib.util.find_spec("tensorboard") is None:
        # `tensorboard` is pinned in pyproject.toml, so this should only trip in an
        # environment that skipped it -- SB3's own logger raises ImportError rather
        # than degrading gracefully if tensorboard_log is set without it installed.
        # Warn loudly (per CLAUDE.md Rule 6, "no silent failures") and continue
        # training without logs rather than fail training over a missing viz dep.
        warnings.warn(
            "tensorboard is not installed; training will proceed without "
            "TensorBoard logs (tensorboard_log is being ignored). Install it "
            "with `pip install -e '.[dev]'` (it's a pinned dependency) to get logs.",
            RuntimeWarning,
            stacklevel=2,
        )
        tensorboard_log = None

    def _make() -> gym.Env:
        return gym.wrappers.RescaleAction(Monitor(make_env()), min_action=-1.0, max_action=1.0)

    venv = DummyVecEnv([_make])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False)

    guard = _NanGuardCallback()
    history_cb = _RewardHistoryCallback(
        reward_history_out if reward_history_out is not None else []
    )
    callback = CallbackList([guard, history_cb])
    if warm_start is not None:
        model, prev_venv = warm_start
        venv.obs_rms = prev_venv.obs_rms
        model.set_random_seed(seed)
        model.set_env(venv)
        model.learn(total_timesteps=total_timesteps, callback=callback, reset_num_timesteps=False)
    else:
        model = PPO(
            "MlpPolicy",
            venv,
            policy_kwargs={"net_arch": policy_net},
            seed=seed,
            verbose=0,
            tensorboard_log=tensorboard_log,
        )
        model.learn(total_timesteps=total_timesteps, callback=callback)

    if guard.diverged:
        raise TrainingDivergedError(
            "PPO training diverged: non-finite policy parameters detected; no checkpoint saved."
        )
    return model, venv
