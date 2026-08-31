"""target_tracking: classic autoscaling to a utilisation target (AWS ASG target-tracking style)."""

from __future__ import annotations

import numpy as np

from rl_edos.config import SimConfig
from rl_edos.env.state import EnvState

DEFAULT_TARGET_UTILIZATION = 0.7


class TargetTrackingController:
    """Sizes the fleet so that arrival_rate / (active_instances * capacity_rps) ~= target."""

    def __init__(
        self, sim: SimConfig, target_utilization: float = DEFAULT_TARGET_UTILIZATION
    ) -> None:
        self.sim = sim
        self.target_utilization = target_utilization

    def __call__(self, state: EnvState) -> np.ndarray:
        needed_capacity = state.arrival_rate / self.target_utilization
        desired = int(np.ceil(needed_capacity / self.sim.capacity_rps_per_instance))
        desired = int(np.clip(desired, self.sim.min_instances, self.sim.max_instances))
        return np.array([float(desired)], dtype=np.float32)
