"""CloudEnv: the simulated cloud as a gymnasium.Env.

Instance pool + request queue + latency model + autoscaler (cooldown,
warm-up) + billing + detection, stepped one control interval at a time.
This is the load-bearing correctness layer — nothing learns until it is
verified (Phase 1 STOP).
"""

from __future__ import annotations

from typing import Any, Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rl_edos.agents.reward import defender_reward
from rl_edos.config import ExperimentConfig
from rl_edos.env.billing import BillingModel
from rl_edos.env.detection import DetectionModel
from rl_edos.env.state import EnvState
from rl_edos.env.traffic import TrafficGenerator

# Observation vector order (see EnvState.to_obs):
# [t, arrival_rate, queue_len, active_instances, warming_instances,
#  latency_ms, accrued_cost, detection_score]
OBS_DIM = 8


class CloudEnv(gym.Env):
    """A discrete-event-ish simulated cloud, stepped in fixed control intervals.

    `attack_traffic_fn(t) -> requests/second` is an optional hook for
    injecting attack traffic on top of legitimate traffic; scripted attack
    generators are wired in here starting Phase 2. Defaults to no attack
    traffic (Phase 1 scope: legit-only dynamics).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: ExperimentConfig,
        attack_traffic_fn: Callable[[float], float] | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.sim = config.sim
        self._attack_traffic_fn = attack_traffic_fn or (lambda t: 0.0)

        self.billing = BillingModel(config.billing)
        self.detection = DetectionModel(config.detection)

        self.action_space = spaces.Box(
            low=float(self.sim.min_instances),
            high=float(self.sim.max_instances),
            shape=(1,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=0.0, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )

        self._rng: np.random.Generator | None = None
        self._traffic: TrafficGenerator | None = None
        self._active = 0
        self._warming: list[float] = []
        self._cooldown_until = 0.0
        self._queue_len = 0.0
        self._t = 0.0
        self._accrued_cost = 0.0
        self._rate_history: list[float] = []
        self._step_count = 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        effective_seed = seed if seed is not None else self.sim.seed
        self._rng = np.random.default_rng(effective_seed)
        self._traffic = TrafficGenerator(self.config.traffic, self._rng)

        self._active = self.sim.min_instances
        self._warming = []
        self._cooldown_until = 0.0
        self._queue_len = 0.0
        self._t = 0.0
        self._accrued_cost = 0.0
        self._step_count = 0

        arrival_rate = self._traffic.sample(self._t) + self._attack_traffic_fn(self._t)
        self._rate_history = [arrival_rate]
        latency_ms = self._latency_ms(self._active, self._queue_len)
        detection_score = self.detection.score(np.array(self._rate_history))

        state = EnvState(
            t=self._t,
            arrival_rate=arrival_rate,
            queue_len=int(round(self._queue_len)),
            active_instances=self._active,
            warming_instances=len(self._warming),
            latency_ms=latency_ms,
            accrued_cost=self._accrued_cost,
            detection_score=detection_score,
        )
        return state.to_obs(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self._rng is None or self._traffic is None:
            raise RuntimeError("CloudEnv.step called before reset")

        self._apply_scaling(action)

        t_next = self._t + self.sim.control_interval_s
        self._promote_warming(t_next)

        legit_rate = self._traffic.sample(t_next)
        attack_rate = self._attack_traffic_fn(t_next)
        arrival_rate = legit_rate + attack_rate
        self._rate_history.append(arrival_rate)
        if len(self._rate_history) > self.config.detection.window:
            self._rate_history = self._rate_history[-self.config.detection.window :]

        queue_len_next, latency_ms, legit_dropped, attack_dropped = self._advance_queue(
            legit_rate, attack_rate
        )

        cost_delta = self.billing.accrue(
            self._active, len(self._warming), self.sim.control_interval_s
        )
        self._accrued_cost += cost_delta
        detection_score = self.detection.score(np.array(self._rate_history))

        self._t = t_next
        self._queue_len = queue_len_next
        self._step_count += 1

        state = EnvState(
            t=self._t,
            arrival_rate=arrival_rate,
            queue_len=int(round(self._queue_len)),
            active_instances=self._active,
            warming_instances=len(self._warming),
            latency_ms=latency_ms,
            accrued_cost=self._accrued_cost,
            detection_score=detection_score,
        )
        reward = defender_reward(cost_delta, latency_ms, self.config.agent.reward_weights)
        terminated = False
        truncated = self._step_count >= self.sim.sim_horizon_steps
        info = {
            "cost": cost_delta,
            "latency": latency_ms,
            "detection_score": detection_score,
            "legit_rate": legit_rate,
            "attack_rate": attack_rate,
            "legit_dropped": legit_dropped,
            "attack_dropped": attack_dropped,
        }
        return state.to_obs(), reward, terminated, truncated, info

    def _apply_scaling(self, action: np.ndarray) -> None:
        desired = int(
            np.clip(round(float(action[0])), self.sim.min_instances, self.sim.max_instances)
        )
        current_total = self._active + len(self._warming)
        if self._t < self._cooldown_until or desired == current_total:
            return

        if desired > current_total:
            n_launch = desired - current_total
            self._warming.extend([self._t + self.sim.instance_warmup_s] * n_launch)
        else:
            n_terminate = current_total - desired
            n_cancel_warming = min(n_terminate, len(self._warming))
            if n_cancel_warming:
                self._warming = self._warming[: len(self._warming) - n_cancel_warming]
            n_terminate -= n_cancel_warming
            self._active -= n_terminate

        self._cooldown_until = self._t + self.sim.scale_cooldown_s

    def _promote_warming(self, t_next: float) -> None:
        still_warming = [r for r in self._warming if r > t_next]
        promoted = len(self._warming) - len(still_warming)
        self._active += promoted
        self._warming = still_warming

    def _advance_queue(
        self, legit_rate: float, attack_rate: float
    ) -> tuple[float, float, float, float]:
        capacity_rps = self._active * self.sim.capacity_rps_per_instance
        legit_incoming = legit_rate * self.sim.control_interval_s
        attack_incoming = attack_rate * self.sim.control_interval_s
        incoming = legit_incoming + attack_incoming + self._queue_len
        capacity_reqs = capacity_rps * self.sim.control_interval_s
        served = min(incoming, capacity_reqs)
        backlog = incoming - served

        queue_len_next = min(backlog, self.sim.max_queue_len)
        overflow = max(0.0, backlog - self.sim.max_queue_len)
        # Overflow (requests beyond max_queue_len, dropped rather than queued) is
        # apportioned between legit/attack by each stream's share of *this
        # interval's new* arrivals — a disclosed simplification, see docs/ASSUMPTIONS.md.
        new_arrivals = legit_incoming + attack_incoming
        legit_share = legit_incoming / new_arrivals if new_arrivals > 0 else 0.0
        legit_dropped = overflow * legit_share
        attack_dropped = overflow * (1.0 - legit_share)

        latency_ms = self._latency_ms(self._active, queue_len_next, capacity_rps=capacity_rps)
        return queue_len_next, latency_ms, legit_dropped, attack_dropped

    def _latency_ms(
        self, active_instances: int, queue_len: float, capacity_rps: float | None = None
    ) -> float:
        if capacity_rps is None:
            capacity_rps = active_instances * self.sim.capacity_rps_per_instance
        if capacity_rps <= 0:
            # No serving capacity at all (reachable when min_instances=0 and the
            # fleet is scaled to zero): the normal capacity-denominated formula
            # divides by zero, so fall back to a flat worst-case 1000ms/request
            # penalty rather than a queueing-theory derivation. See
            # docs/ASSUMPTIONS.md "Zero-capacity latency fallback".
            return self.sim.base_latency_ms + queue_len * 1000.0
        return self.sim.base_latency_ms + (queue_len / capacity_rps) * 1000.0
