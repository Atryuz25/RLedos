"""Reward functions. All reward math lives here, never scattered in env/.

Phase 1 wires the joint cost+latency defender reward so CloudEnv.step returns
a number; weight tuning is Phase 3.
"""

from __future__ import annotations


def defender_reward(
    cost_delta: float, latency_ms: float, reward_weights: dict[str, float]
) -> float:
    """Joint cost+latency defender reward: -(cost_w * cost_delta + latency_w * latency_ms).

    `cost_delta` is currency for the interval, `latency_ms` is milliseconds.
    Weights come from AgentConfig.reward_weights.
    """
    cost_w = reward_weights.get("cost_w", 1.0)
    latency_w = reward_weights.get("latency_w", 0.0)
    return -(cost_w * cost_delta + latency_w * latency_ms)
