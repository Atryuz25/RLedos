"""RLAttackerController: wraps a trained Phase 4 PPO attacker checkpoint.

Same checkpoint-pair contract as `agents/defender.py` (`policy.zip` +
`vecnormalize.pkl`), but the action isn't a scaling decision -- it's a
traffic-shaping intensity in `[0, evasion_budget]` (relative to `base_rate`,
same unit as `AttackSpec.intensity`), so this exposes `.rate(obs)` rather than
the baselines' `Controller` protocol.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from rl_edos.training.sb3_utils import normalize_obs, rescale_action

POLICY_FILENAME = "policy.zip"
VECNORMALIZE_FILENAME = "vecnormalize.pkl"


class RLAttackerController:
    """Callable attacker: EnvState-shaped obs -> attack rate (requests/second)."""

    def __init__(
        self,
        model: PPO,
        evasion_budget: float,
        base_rate: float,
        vecnorm: VecNormalize | None = None,
    ) -> None:
        self.model = model
        self.evasion_budget = evasion_budget
        self.base_rate = base_rate
        self.vecnorm = vecnorm

    def intensity(self, obs: np.ndarray) -> float:
        """Attack magnitude relative to base_rate, clipped to [0, evasion_budget]."""
        normalized = normalize_obs(obs, self.vecnorm)
        action, _ = self.model.predict(normalized, deterministic=True)
        real_action = rescale_action(action, 0.0, self.evasion_budget)
        return float(np.clip(real_action[0], 0.0, self.evasion_budget))

    def rate(self, obs: np.ndarray) -> float:
        """Attack request rate, requests/second, for the given observation."""
        return self.intensity(obs) * self.base_rate


def load_attacker(
    policy_dir: str | Path, evasion_budget: float, base_rate: float
) -> RLAttackerController:
    """Load a self-play attacker checkpoint directory into a callable controller."""
    policy_dir = Path(policy_dir)
    model = PPO.load(policy_dir / POLICY_FILENAME)
    vecnorm_path = policy_dir / VECNORMALIZE_FILENAME
    vecnorm = None
    if vecnorm_path.is_file():
        with open(vecnorm_path, "rb") as fh:
            vecnorm = pickle.load(fh)
    return RLAttackerController(model, evasion_budget, base_rate, vecnorm)
