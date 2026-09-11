"""DetectionModel: scoring on labelled legit/attack windows, both detector types."""

from __future__ import annotations

import numpy as np
import pytest
from rl_edos.config import DetectionConfig
from rl_edos.env.cloud_env import CloudEnv
from rl_edos.env.detection import DetectionModel
from rl_edos.env.traffic import TrafficGenerator


def test_rate_threshold_scores_legit_window_below_one():
    cfg = DetectionConfig(type="rate_threshold", window=5, threshold=100.0, penalty=1.0)
    model = DetectionModel(cfg)
    legit_window = np.array([20.0, 22.0, 19.0, 21.0, 20.0])  # mean ~20.4, well under threshold
    assert model.score(legit_window) < 1.0


def test_rate_threshold_scores_attack_window_above_one():
    cfg = DetectionConfig(type="rate_threshold", window=5, threshold=100.0, penalty=1.0)
    model = DetectionModel(cfg)
    attack_window = np.array([150.0, 160.0, 140.0, 155.0, 145.0])  # mean ~150, over threshold
    assert model.score(attack_window) > 1.0


def test_variance_threshold_scores_bursty_window_higher_than_steady():
    cfg = DetectionConfig(type="variance_threshold", window=6, threshold=10.0, penalty=1.0)
    model = DetectionModel(cfg)
    steady = np.array([50.0, 51.0, 49.0, 50.0, 50.0, 51.0])
    bursty = np.array([10.0, 90.0, 15.0, 85.0, 20.0, 80.0])
    assert model.score(bursty) > model.score(steady)


def test_empty_window_scores_zero():
    cfg = DetectionConfig(type="rate_threshold", window=5, threshold=100.0, penalty=1.0)
    model = DetectionModel(cfg)
    assert model.score(np.array([])) == 0.0


def test_score_is_deterministic():
    cfg = DetectionConfig(type="rate_threshold", window=5, threshold=100.0, penalty=1.0)
    model = DetectionModel(cfg)
    window = np.array([30.0, 32.0, 28.0, 31.0, 29.0])
    assert model.score(window) == model.score(window)


def test_detection_score_reflects_growing_then_capped_window(make_config):
    # CloudEnv feeds DetectionModel a *growing* window for the first `window - 1`
    # steps of every episode (reset() starts it at length 1), then caps it at
    # `window`. DetectionModel's own unit tests above only ever pass fixed,
    # already-full-length arrays, so this integration path was previously untested.
    cfg = make_config(
        {
            "detection": {"type": "rate_threshold", "window": 5, "threshold": 50.0, "penalty": 1.0},
            "traffic": {"legit_pattern": "poisson", "base_rate": 20.0, "noise_std": 0.0},
            "attack": {"mode": "scripted", "pattern": "steady", "intensity": 0.0},
        }
    )
    env = CloudEnv(cfg)
    obs, _ = env.reset(seed=1)

    # Mirror CloudEnv's traffic-sampling call pattern with an independently seeded
    # TrafficGenerator (noise_std=0 makes this an exact reconstruction, not just an
    # expected match) to build the same growing/capped rate history CloudEnv holds
    # internally but doesn't expose.
    rng = np.random.default_rng(1)
    traffic = TrafficGenerator(cfg.traffic, rng)
    detector = DetectionModel(cfg.detection)

    history = [traffic.sample(0.0)]
    assert obs[7] == pytest.approx(detector.score(np.array(history)))

    for step in range(1, 8):  # more steps than window=5, to exercise both phases
        t_next = step * cfg.sim.control_interval_s
        obs, *_ = env.step(np.array([1.0], dtype=np.float32))
        history.append(traffic.sample(t_next))
        if len(history) > cfg.detection.window:
            history = history[-cfg.detection.window :]

        assert len(history) == min(step + 1, cfg.detection.window)
        assert obs[7] == pytest.approx(detector.score(np.array(history)))
