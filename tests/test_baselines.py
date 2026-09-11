"""Baseline controllers: shared action contract, sensible behaviour under load."""

from __future__ import annotations

import numpy as np
import pytest
from rl_edos.baselines.randomisation import RandomisationController
from rl_edos.baselines.security_blind_rl import BLIND_OBS_DIM, train_security_blind
from rl_edos.baselines.target_tracking import TargetTrackingController
from rl_edos.env.cloud_env import OBS_DIM
from rl_edos.env.state import EnvState


def _state(**overrides) -> EnvState:
    base = dict(
        t=0.0,
        arrival_rate=20.0,
        queue_len=0,
        active_instances=1,
        warming_instances=0,
        latency_ms=10.0,
        accrued_cost=0.0,
        detection_score=0.0,
    )
    base.update(overrides)
    return EnvState(**base)


def test_target_tracking_scales_up_under_higher_arrival(make_config):
    cfg = make_config()
    controller = TargetTrackingController(cfg.sim)
    low = controller(_state(arrival_rate=5.0))[0]
    high = controller(_state(arrival_rate=500.0))[0]
    assert high > low


def test_target_tracking_respects_bounds(make_config):
    cfg = make_config({"sim": {"min_instances": 2, "max_instances": 4}})
    controller = TargetTrackingController(cfg.sim)
    action_low = controller(_state(arrival_rate=0.0))
    action_high = controller(_state(arrival_rate=1e6))
    assert cfg.sim.min_instances <= action_low[0] <= cfg.sim.max_instances
    assert cfg.sim.min_instances <= action_high[0] <= cfg.sim.max_instances


def test_target_tracking_action_shape_matches_contract(make_config):
    cfg = make_config()
    controller = TargetTrackingController(cfg.sim)
    action = controller(_state())
    assert isinstance(action, np.ndarray)
    assert action.shape == (1,)


def test_randomisation_stub_raises():
    controller = RandomisationController()
    with pytest.raises(NotImplementedError):
        controller(_state())


def test_security_blind_controller_returns_valid_bounded_action(make_config):
    cfg = make_config({"agent": {"total_timesteps": 64}})
    controller = train_security_blind(cfg, seed=1)
    action = controller(_state())
    assert action.shape == (1,)
    assert cfg.sim.min_instances <= action[0] <= cfg.sim.max_instances


def test_blind_obs_dim_tracks_full_obs_dim():
    # BLIND_OBS_DIM must stay derived from OBS_DIM (drops exactly detection_score,
    # the last field of EnvState.to_obs()) so it can't silently drift if the
    # observation vector grows.
    assert BLIND_OBS_DIM == OBS_DIM - 1
