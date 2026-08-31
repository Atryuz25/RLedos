"""Scripted attack traffic patterns, injected via CloudEnv's attack_traffic_fn hook.

Deterministic (pure functions of simulation time), so "seeded and reproducible"
holds trivially — no randomness to seed.
"""

from __future__ import annotations

import math
from typing import Callable

from rl_edos.config import AttackSpec, SimConfig

BURST_DUTY_CYCLE = 0.2  # fraction of each cycle a burst attack is "on"


class ScriptedAttack:
    """Attack request-rate shape, parameterised by AttackSpec.intensity (relative to base_rate).

    oscillation/burst period is derived from the episode length (4 cycles per
    episode) rather than hard-coded, so it stays config-driven without adding
    an unused schema field.
    """

    def __init__(self, spec: AttackSpec, sim: SimConfig, base_rate: float) -> None:
        self.spec = spec
        self.base_rate = base_rate
        horizon_s = sim.sim_horizon_steps * sim.control_interval_s
        self._period_s = max(sim.control_interval_s, horizon_s / 4)

    def rate(self, t: float) -> float:
        """Attack request rate, requests/second, at simulation time `t` seconds."""
        peak = self.spec.intensity * self.base_rate
        if self.spec.pattern == "steady":
            return peak
        phase = (t % self._period_s) / self._period_s
        if self.spec.pattern == "oscillation":
            return peak * (0.5 + 0.5 * math.sin(2 * math.pi * phase))
        # burst
        return peak if phase < BURST_DUTY_CYCLE else 0.0


def build_attack_fn(spec: AttackSpec, sim: SimConfig, base_rate: float) -> Callable[[float], float]:
    """Build the CloudEnv attack_traffic_fn hook for a given AttackSpec."""
    if spec.intensity <= 0:
        return lambda t: 0.0
    return ScriptedAttack(spec, sim, base_rate).rate
