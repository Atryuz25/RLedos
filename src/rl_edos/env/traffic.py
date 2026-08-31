"""TrafficGenerator: legitimate request-rate generation, seeded and reproducible.

Scripted attack traffic is Phase 2; CloudEnv accepts a pluggable attack-rate
function so this module stays legit-only.
"""

from __future__ import annotations

import math

import numpy as np

from rl_edos.config import TrafficSpec

SECONDS_PER_DAY = 86400.0


class TrafficGenerator:
    """Generates legitimate arrival rates (requests/second) per control interval."""

    def __init__(self, spec: TrafficSpec, rng: np.random.Generator) -> None:
        self.spec = spec
        self.rng = rng

    def sample(self, t: float) -> float:
        """Arrival rate, requests/second, at simulation time `t` seconds."""
        if self.spec.legit_pattern == "poisson":
            mean_rate = self.spec.base_rate
        else:  # diurnal
            mean_rate = self.spec.base_rate * (
                1.0 + self.spec.diurnal_amplitude * math.sin(2 * math.pi * t / SECONDS_PER_DAY)
            )
        noise = self.rng.normal(0.0, self.spec.noise_std)
        return max(0.0, mean_rate + noise)
