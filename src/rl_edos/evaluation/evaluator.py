"""Evaluator: run every controller under identical attack seeds/conditions."""

from __future__ import annotations

import hashlib
import json
from typing import Protocol

import numpy as np

from rl_edos.config import AttackSpec, ExperimentConfig
from rl_edos.env.attacks import build_attack_fn
from rl_edos.env.cloud_env import CloudEnv
from rl_edos.env.state import EnvState
from rl_edos.evaluation.metrics import (
    EpisodeTrace,
    MetricsSummary,
    RunRecord,
    average_metrics,
    compute_metrics,
)


class Controller(Protocol):
    def __call__(self, state: EnvState) -> np.ndarray: ...


def config_hash(config: ExperimentConfig) -> str:
    payload = json.dumps(config.model_dump(), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


class Evaluator:
    """Runs controllers under identical attack seeds/conditions, per docs/02_ARCHITECTURE.md."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    def run(
        self,
        controllers: dict[str, Controller],
        attack: AttackSpec | None = None,
        seeds: list[int] | None = None,
    ) -> dict[str, tuple[MetricsSummary, list[RunRecord]]]:
        """Evaluate each controller under identical (attack, seeds).

        Returns, per controller name, the seed-averaged MetricsSummary and one
        RunRecord per seed (for reproducibility bookkeeping).
        """
        attack = attack or self.config.attack
        seeds = seeds if seeds is not None else [self.config.sim.seed]
        hash_ = config_hash(self.config)

        results: dict[str, tuple[MetricsSummary, list[RunRecord]]] = {}
        for name, controller in controllers.items():
            per_seed_metrics: list[MetricsSummary] = []
            run_records: list[RunRecord] = []
            for seed in seeds:
                trace = self._run_episode(controller, attack, seed)
                metrics = compute_metrics(
                    trace,
                    self.config.sim.capacity_rps_per_instance,
                    self.config.sim.min_instances,
                )
                per_seed_metrics.append(metrics)
                run_records.append(
                    RunRecord(
                        run_id=RunRecord.new_run_id(),
                        config_hash=hash_,
                        controller=name,
                        attack=attack,
                        metrics=metrics,
                        artifacts=[],
                        seed=seed,
                    )
                )
            results[name] = (average_metrics(per_seed_metrics), run_records)
        return results

    def _run_episode(self, controller: Controller, attack: AttackSpec, seed: int) -> EpisodeTrace:
        attack_fn = build_attack_fn(attack, self.config.sim, self.config.traffic.base_rate)
        env = CloudEnv(self.config, attack_traffic_fn=attack_fn)
        obs, _ = env.reset(seed=seed)
        trace = EpisodeTrace()
        terminated = truncated = False
        while not (terminated or truncated):
            state = EnvState.from_obs(obs)
            action = controller(state)
            obs, _reward, terminated, truncated, info = env.step(action)
            trace.costs.append(info["cost"])
            trace.latencies_ms.append(info["latency"])
            trace.legit_rates.append(info["legit_rate"])
            trace.active_instances.append(int(round(obs[3])))
            trace.legit_incoming.append(info["legit_rate"] * self.config.sim.control_interval_s)
            trace.legit_dropped.append(info["legit_dropped"])
        return trace
