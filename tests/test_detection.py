"""DetectionModel: scoring on labelled legit/attack windows, both detector types."""

from __future__ import annotations

import numpy as np
from rl_edos.config import DetectionConfig
from rl_edos.env.detection import DetectionModel


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
