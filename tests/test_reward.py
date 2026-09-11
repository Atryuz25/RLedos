"""defender_reward: sign, cost/latency scaling, weight application."""

from __future__ import annotations

from rl_edos.agents.reward import defender_reward


def test_reward_is_negative_for_positive_cost_and_latency():
    assert defender_reward(1.0, 10.0, {"cost_w": 1.0, "latency_w": 0.01}) < 0


def test_reward_matches_hand_computation():
    # -(1.0 * 2.0 + 0.01 * 100.0) = -(2.0 + 1.0) = -3.0
    assert defender_reward(2.0, 100.0, {"cost_w": 1.0, "latency_w": 0.01}) == -3.0


def test_higher_cost_delta_reduces_reward():
    weights = {"cost_w": 1.0, "latency_w": 0.01}
    assert defender_reward(2.0, 10.0, weights) < defender_reward(1.0, 10.0, weights)


def test_higher_latency_reduces_reward():
    weights = {"cost_w": 1.0, "latency_w": 0.01}
    assert defender_reward(1.0, 200.0, weights) < defender_reward(1.0, 10.0, weights)


def test_cost_weight_scales_cost_contribution():
    low_w = defender_reward(1.0, 0.0, {"cost_w": 1.0, "latency_w": 0.0})
    high_w = defender_reward(1.0, 0.0, {"cost_w": 2.0, "latency_w": 0.0})
    assert high_w == 2 * low_w


def test_missing_weights_default_to_zero():
    # defender_reward's own fallback (cost_w=1.0, latency_w=0.0) differs from
    # AgentConfig's populated default ({"cost_w": 1.0, "latency_w": 0.01}) -- an
    # empty dict only reaches here if a caller bypasses config, so the function's
    # bare fallback is cost-only.
    assert defender_reward(3.0, 1000.0, {}) == -3.0
