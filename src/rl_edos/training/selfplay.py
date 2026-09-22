"""selfplay: co-evolutionary attacker/defender training loop (Phase 4, stretch).

Alternates freeze/train: freeze the defender and train an attacker to inflate
its bill under the detection constraint; freeze that attacker and train the
defender to counter while staying cheap; repeat for `config.selfplay.rounds`
rounds. Each round is warm-started from the previous round's weights for both
agents (`training/sb3_utils.py::train_ppo_normalized`'s `warm_start`), so this
is genuine co-evolution, not `rounds` independent training runs.

Convergence is declared when `mean_cost_under_attack` (the attacker's actual
achieved cost against that round's defender, from a deterministic evaluation
episode) changes by less than `selfplay.convergence_tolerance` (relative) for
`selfplay.convergence_patience` consecutive rounds. Divergence (NaN in either
agent) or exhausting `rounds` without convergence both degrade to a documented
stability/failure-analysis artifact (`stability.json`) rather than raising --
both are acceptable Phase 4 outcomes per docs/01_PROJECT_SPEC.md.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from rl_edos.config import ConfigError, ExperimentConfig
from rl_edos.env.attacks import build_attack_fn
from rl_edos.env.cloud_env import CloudEnv
from rl_edos.evaluation.metrics import EpisodeTrace, compute_metrics
from rl_edos.training.sb3_utils import (
    TrainingDivergedError,
    normalize_obs,
    rescale_action,
    train_ppo_normalized,
)
from rl_edos.training.selfplay_envs import AttackerSelfPlayEnv, DefenderSelfPlayEnv

DEFENDER_SUBDIR = "defender"
ATTACKER_SUBDIR = "attacker"
POLICY_FILENAME = "policy.zip"
VECNORMALIZE_FILENAME = "vecnormalize.pkl"


@dataclass
class SelfPlayResult:
    """Self-play outcome: either `converged` or a `failure_reason` is set, never neither."""

    rounds_completed: int
    converged: bool
    diverged: bool
    failure_reason: str | None
    round_metrics: list[dict] = field(default_factory=list)
    defender_ckpt: str | None = None
    attacker_ckpt: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def _save_agent(model, vecnorm, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir / POLICY_FILENAME)
    vecnorm.save(str(out_dir / VECNORMALIZE_FILENAME))
    return out_dir


def _evaluate_round(
    config: ExperimentConfig,
    defender_model,
    defender_vecnorm,
    attacker_model,
    attacker_vecnorm,
    seed: int,
) -> dict:
    """One deterministic episode of defender vs. attacker, both acting greedily.

    A separate small loop rather than `evaluation.Evaluator`: the Evaluator
    assumes one controller against a fixed/scripted attack, but self-play
    needs *both* agents acting from the same shared observation each step.
    """
    cloud = CloudEnv(config)
    obs, _ = cloud.reset(seed=seed)
    trace = EpisodeTrace()
    detection_trips = 0
    steps = 0
    terminated = truncated = False
    while not (terminated or truncated):
        defender_obs = normalize_obs(obs, defender_vecnorm)
        raw_defender_action, _ = defender_model.predict(defender_obs, deterministic=True)
        defender_action = np.clip(
            rescale_action(raw_defender_action, config.sim.min_instances, config.sim.max_instances),
            config.sim.min_instances,
            config.sim.max_instances,
        ).astype(np.float32)

        attacker_obs = normalize_obs(obs, attacker_vecnorm)
        raw_attacker_action, _ = attacker_model.predict(attacker_obs, deterministic=True)
        intensity = float(
            np.clip(
                rescale_action(raw_attacker_action, 0.0, config.attack.evasion_budget)[0],
                0.0,
                config.attack.evasion_budget,
            )
        )
        attack_rate = intensity * config.traffic.base_rate

        obs, _reward, terminated, truncated, info = cloud.step(
            defender_action, attack_rate_override=attack_rate
        )
        trace.costs.append(info["cost"])
        trace.latencies_ms.append(info["latency"])
        trace.legit_rates.append(info["legit_rate"])
        trace.active_instances.append(int(round(obs[3])))
        trace.legit_incoming.append(info["legit_rate"] * config.sim.control_interval_s)
        trace.legit_dropped.append(info["legit_dropped"])
        detection_trips += int(info["detection_score"] >= 1.0)
        steps += 1

    metrics = compute_metrics(trace, config.sim.capacity_rps_per_instance, config.sim.min_instances)
    return {
        "mean_cost_under_attack": metrics.mean_cost_under_attack,
        "p95_latency_ms": metrics.p95_latency_ms,
        "legit_drop_rate": metrics.legit_drop_rate,
        "detection_trip_rate": detection_trips / steps if steps else 0.0,
    }


def run_selfplay(
    config: ExperimentConfig, out_dir: str | Path, seed: int | None = None
) -> SelfPlayResult:
    """Run the alternating freeze/train self-play loop; never raises on non-convergence.

    Round 0 seeds a defender against the *scripted* `config.attack` (the Phase
    3 pathway) so the attacker has a non-trivial opponent from the start, then
    trains round 0's attacker against it. Rounds 1..N alternate freeze/train.
    Always writes `out_dir/stability.json` and checkpoints for both agents,
    even when the loop ends via divergence or exhausted rounds.
    """
    if config.attack.evasion_budget <= 0:
        raise ConfigError(
            "selfplay requires attack.evasion_budget > 0 to define the learned "
            "attacker's action range (AttackSpec.evasion_budget)"
        )

    out_dir = Path(out_dir)
    base_seed = seed if seed is not None else config.sim.seed
    sp = config.selfplay
    round_metrics: list[dict] = []

    scripted_attack_fn = build_attack_fn(config.attack, config.sim, config.traffic.base_rate)
    defender_model, defender_vecnorm = train_ppo_normalized(
        lambda: CloudEnv(config, attack_traffic_fn=scripted_attack_fn),
        total_timesteps=sp.round_timesteps,
        policy_net=config.agent.policy_net,
        seed=base_seed,
    )
    attacker_model, attacker_vecnorm = train_ppo_normalized(
        lambda: AttackerSelfPlayEnv(config, defender_model, defender_vecnorm),
        total_timesteps=sp.round_timesteps,
        policy_net=config.agent.policy_net,
        seed=base_seed + 1,
    )
    round_metrics.append(
        {
            "round": 0,
            **_evaluate_round(
                config,
                defender_model,
                defender_vecnorm,
                attacker_model,
                attacker_vecnorm,
                base_seed,
            ),
        }
    )

    converged = False
    diverged = False
    failure_reason: str | None = None
    stable_rounds = 0
    rounds_completed = 0

    for r in range(1, sp.rounds + 1):
        round_seed = base_seed + 2 * r
        try:
            defender_model, defender_vecnorm = train_ppo_normalized(
                lambda: DefenderSelfPlayEnv(config, attacker_model, attacker_vecnorm),
                total_timesteps=sp.round_timesteps,
                policy_net=config.agent.policy_net,
                seed=round_seed,
                warm_start=(defender_model, defender_vecnorm),
            )
            attacker_model, attacker_vecnorm = train_ppo_normalized(
                lambda: AttackerSelfPlayEnv(config, defender_model, defender_vecnorm),
                total_timesteps=sp.round_timesteps,
                policy_net=config.agent.policy_net,
                seed=round_seed + 1,
                warm_start=(attacker_model, attacker_vecnorm),
            )
        except TrainingDivergedError as exc:
            diverged = True
            failure_reason = (
                f"round {r} diverged ({exc}); checkpointing the last known-good pair "
                "(from before this round) instead of a broken one."
            )
            break

        metrics = _evaluate_round(
            config, defender_model, defender_vecnorm, attacker_model, attacker_vecnorm, round_seed
        )
        round_metrics.append({"round": r, **metrics})
        rounds_completed = r

        prev_cost = round_metrics[-2]["mean_cost_under_attack"]
        curr_cost = metrics["mean_cost_under_attack"]
        rel_change = abs(curr_cost - prev_cost) / prev_cost if prev_cost > 0 else abs(curr_cost)
        stable_rounds = stable_rounds + 1 if rel_change <= sp.convergence_tolerance else 0
        if stable_rounds >= sp.convergence_patience:
            converged = True
            break

    defender_ckpt = _save_agent(defender_model, defender_vecnorm, out_dir / DEFENDER_SUBDIR)
    attacker_ckpt = _save_agent(attacker_model, attacker_vecnorm, out_dir / ATTACKER_SUBDIR)

    if not converged and not diverged:
        failure_reason = (
            f"self-play did not converge within {sp.rounds} rounds "
            f"(tolerance={sp.convergence_tolerance}, patience={sp.convergence_patience}); "
            "reporting the best available checkpoints and full stability log as a "
            "documented failure analysis, per docs/PHASE_4_SELFPLAY_AND_RELEASE.md."
        )

    result = SelfPlayResult(
        rounds_completed=rounds_completed,
        converged=converged,
        diverged=diverged,
        failure_reason=failure_reason,
        round_metrics=round_metrics,
        defender_ckpt=str(defender_ckpt),
        attacker_ckpt=str(attacker_ckpt),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stability.json").write_text(json.dumps(result.as_dict(), indent=2))
    return result
