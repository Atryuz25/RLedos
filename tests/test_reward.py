"""defender_reward / attacker_reward: sign, cost/latency scaling, weight application."""

from __future__ import annotations

from rl_edos.agents.reward import attacker_reward, defender_reward


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


def test_attacker_reward_matches_hand_computation_when_undetected():
    # cost_gain_w * cost_inflicted - evasion_w * 0 (score < 1.0, not tripped) = 1.5 * 2.0 = 3.0
    weights = {"cost_gain_w": 1.5, "evasion_w": 1.0}
    assert attacker_reward(2.0, 0.5, detection_penalty=1.0, reward_weights=weights) == 3.0


def test_attacker_reward_subtracts_penalty_when_detected():
    weights = {"cost_gain_w": 1.0, "evasion_w": 2.0}
    # score >= 1.0 trips detection: 1.0*2.0 - 2.0*1.0*0.5 = 2.0 - 1.0 = 1.0
    assert attacker_reward(2.0, 1.0, detection_penalty=0.5, reward_weights=weights) == 1.0


def test_attacker_reward_detection_boundary_is_inclusive():
    weights = {"cost_gain_w": 1.0, "evasion_w": 1.0}
    just_under = attacker_reward(1.0, 0.999999, detection_penalty=10.0, reward_weights=weights)
    at_threshold = attacker_reward(1.0, 1.0, detection_penalty=10.0, reward_weights=weights)
    assert just_under == 1.0
    assert at_threshold == 1.0 - 10.0


def test_attacker_reward_higher_cost_inflicted_increases_reward():
    weights = {"cost_gain_w": 1.0, "evasion_w": 1.0}
    low = attacker_reward(1.0, 0.0, detection_penalty=1.0, reward_weights=weights)
    high = attacker_reward(5.0, 0.0, detection_penalty=1.0, reward_weights=weights)
    assert high > low


def test_attacker_reward_missing_weights_default_to_one():
    # cost_gain_w/evasion_w default to 1.0 (see agents/reward.py), unlike
    # defender_reward's latency_w which defaults to 0.0 -- different fallback
    # semantics per function, each documented at its own call site.
    assert attacker_reward(2.0, 0.0, detection_penalty=1.0, reward_weights={}) == 2.0
