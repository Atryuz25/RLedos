"""Reward functions. All reward math lives here, never scattered in env/.

Phase 1 wires the joint cost+latency defender reward so CloudEnv.step returns
a number; weight tuning is Phase 3. `attacker_reward` is Phase 4.
"""

from __future__ import annotations


def defender_reward(
    cost_delta: float,
    latency_ms: float,
    reward_weights: dict[str, float],
    latency_threshold_ms: float = 0.0,
) -> float:
    """Joint cost+latency defender reward:
    -(cost_w * cost_delta + latency_w * max(0, latency_ms - latency_threshold_ms)).

    `cost_delta` is currency for the interval, `latency_ms` is milliseconds.
    Weights come from AgentConfig.reward_weights; `latency_threshold_ms` comes
    from AgentConfig.latency_threshold_ms (default 0.0, which reduces to the
    original always-penalize-any-latency formula -- existing configs are
    unaffected).

    Without a threshold, *any* latency above zero costs something, however
    small latency_w is -- which makes "keep the queue perfectly empty" a
    dominant strategy regardless of weight scale, since under-provisioning
    even briefly spikes latency by orders of magnitude more than any single
    step's cost saving (see docs/PHASE_3_RL_DEFENDER.md's "Current status").
    A positive threshold models a realistic SLA: latency at or below it is
    free, so the defender can actually trade some queueing for lower cost
    without being penalized for service that's still perfectly fine.
    """
    cost_w = reward_weights.get("cost_w", 1.0)
    latency_w = reward_weights.get("latency_w", 0.0)
    latency_penalty = max(0.0, latency_ms - latency_threshold_ms)
    return -(cost_w * cost_delta + latency_w * latency_penalty)


def attacker_reward(
    cost_inflicted: float,
    detection_score: float,
    detection_penalty: float,
    reward_weights: dict[str, float],
) -> float:
    """Attacker reward: cost_gain_w * cost_inflicted - evasion_w * detection_penalty (if tripped).

    `cost_inflicted` is the defender's accrued `cost_delta` for the interval
    (currency) -- the attacker is rewarded for the bill it causes, not for a
    separate metric. `detection_score` >= 1.0 means DetectionModel tripped this
    interval, applying `detection_penalty` (DetectionConfig.penalty) scaled by
    `evasion_w`. Weights come from AgentConfig.reward_weights.
    """
    cost_gain_w = reward_weights.get("cost_gain_w", 1.0)
    evasion_w = reward_weights.get("evasion_w", 1.0)
    detected = 1.0 if detection_score >= 1.0 else 0.0
    return cost_gain_w * cost_inflicted - evasion_w * detected * detection_penalty
