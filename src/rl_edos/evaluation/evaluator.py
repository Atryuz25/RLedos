"""Evaluator: run every controller under identical attack seeds/conditions."""

from __future__ import annotations

import hashlib
import json
from typing import Protocol

import numpy as np

from rl_edos.agents.attacker import RLAttackerController, load_attacker
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

        When `attack.mode == "learned"`, every controller is evaluated against
        the trained attacker at `attack.attacker_checkpoint` instead of a
        scripted pattern -- `AttackSpec.validate_learned_attacker` is called
        here (fail fast, per CLAUDE.md Rule 6) rather than at config-load time,
        since `selfplay --config <path>` legitimately loads a `mode: learned`
        config with no checkpoint yet (it's what produces one). `mode ==
        "scripted"` (the default) is unaffected -- same code path as before
        this existed.

        Returns, per controller name, the seed-averaged MetricsSummary and one
        RunRecord per seed (for reproducibility bookkeeping).
        """
        attack = attack or self.config.attack
        seeds = seeds if seeds is not None else [self.config.sim.seed]
        hash_ = config_hash(self.config)

        attacker: RLAttackerController | None = None
        if attack.mode == "learned":
            attack.validate_learned_attacker()
            attacker = load_attacker(
                attack.attacker_checkpoint, attack.evasion_budget, self.config.traffic.base_rate
            )

        results: dict[str, tuple[MetricsSummary, list[RunRecord]]] = {}
        for name, controller in controllers.items():
            per_seed_metrics: list[MetricsSummary] = []
            run_records: list[RunRecord] = []
            for seed in seeds:
                trace = self._run_episode(controller, attack, seed, attacker=attacker)
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

    def _run_episode(
        self,
        controller: Controller,
        attack: AttackSpec,
        seed: int,
        attacker: RLAttackerController | None = None,
    ) -> EpisodeTrace:
        """Run one episode; `attacker`, if given, drives the attack rate instead of `attack`.

        A learned attacker's rate depends on the current observation, not
        simulation time, so it can't be expressed as `CloudEnv`'s time-indexed
        `attack_traffic_fn` hook (the same reason self-play uses
        `attack_rate_override` -- see `training/selfplay_envs.py`).
        """
        attack_fn = (
            None
            if attacker is not None
            else build_attack_fn(attack, self.config.sim, self.config.traffic.base_rate)
        )
        env = CloudEnv(self.config, attack_traffic_fn=attack_fn)
        obs, _ = env.reset(seed=seed)
        trace = EpisodeTrace()
        terminated = truncated = False
        while not (terminated or truncated):
            state = EnvState.from_obs(obs)
            action = controller(state)
            attack_rate_override = attacker.rate(obs) if attacker is not None else None
            obs, _reward, terminated, truncated, info = env.step(
                action, attack_rate_override=attack_rate_override
            )
            trace.costs.append(info["cost"])
            trace.latencies_ms.append(info["latency"])
            trace.legit_rates.append(info["legit_rate"])
            trace.active_instances.append(int(round(obs[3])))
            trace.legit_incoming.append(info["legit_rate"] * self.config.sim.control_interval_s)
            trace.legit_dropped.append(info["legit_dropped"])
        return trace
