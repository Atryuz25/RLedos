"""Shared test fixtures: a minimal valid ExperimentConfig, overridable per test."""

from __future__ import annotations

import copy

import pytest
from rl_edos.config import ExperimentConfig

BASE_CONFIG: dict = {
    "sim": {
        "control_interval_s": 10.0,
        "sim_horizon_steps": 50,
        "instance_warmup_s": 20.0,
        "scale_cooldown_s": 30.0,
        "max_instances": 5,
        "min_instances": 1,
        "seed": 7,
        "capacity_rps_per_instance": 10.0,
        "base_latency_ms": 5.0,
    },
    "billing": {
        "price_per_instance_second": 0.00001,
        "billing_granularity": "per_second",
        "min_service_charge": 0.0,
    },
    "traffic": {
        "legit_pattern": "poisson",
        "base_rate": 20.0,
        "diurnal_amplitude": 0.0,
        "noise_std": 0.0,
    },
    "attack": {"mode": "scripted", "pattern": "steady", "intensity": 0.0, "evasion_budget": 0.0},
    "detection": {"type": "rate_threshold", "window": 5, "threshold": 50.0, "penalty": 1.0},
    "agent": {
        "algo": "PPO",
        "reward_weights": {"cost_w": 1.0, "latency_w": 0.01},
        "total_timesteps": 100,
        "policy_net": [8],
    },
}


@pytest.fixture
def make_config():
    def _make(overrides: dict | None = None) -> ExperimentConfig:
        cfg = copy.deepcopy(BASE_CONFIG)
        overrides = overrides or {}
        for section, fields in overrides.items():
            cfg.setdefault(section, {})
            if isinstance(fields, dict):
                cfg[section].update(fields)
            else:
                cfg[section] = fields
        return ExperimentConfig(**cfg)

    return _make


@pytest.fixture
def base_config(make_config) -> ExperimentConfig:
    return make_config()
