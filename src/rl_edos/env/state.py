"""EnvState: the per-step simulation state, also the defender observation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EnvState:
    """Snapshot of the simulated cloud at one control interval.

    Units: t in seconds; arrival_rate in requests/second; queue_len in
    requests; active_instances/warming_instances as counts; latency_ms in
    milliseconds; accrued_cost in currency; detection_score is a
    dimensionless scalar.
    """

    t: float
    arrival_rate: float
    queue_len: int
    active_instances: int
    warming_instances: int
    latency_ms: float
    accrued_cost: float
    detection_score: float

    @classmethod
    def from_obs(cls, obs: np.ndarray) -> "EnvState":
        """Inverse of to_obs(), for controllers that take an EnvState rather than a raw vector."""
        return cls(
            t=float(obs[0]),
            arrival_rate=float(obs[1]),
            queue_len=int(round(obs[2])),
            active_instances=int(round(obs[3])),
            warming_instances=int(round(obs[4])),
            latency_ms=float(obs[5]),
            accrued_cost=float(obs[6]),
            detection_score=float(obs[7]),
        )

    def to_obs(self) -> np.ndarray:
        """Vectorise for the defender observation space, float32."""
        return np.array(
            [
                self.t,
                self.arrival_rate,
                float(self.queue_len),
                float(self.active_instances),
                float(self.warming_instances),
                self.latency_ms,
                self.accrued_cost,
                self.detection_score,
            ],
            dtype=np.float32,
        )
