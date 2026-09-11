"""Config schema validation: valid loads, fail-fast on invalid input."""

from __future__ import annotations

from pathlib import Path

import pytest
from rl_edos.config import ConfigError, load_config

DEFAULT_CONFIG = Path(__file__).parent.parent / "src/rl_edos/configs/default.yaml"


def test_default_config_loads():
    cfg = load_config(DEFAULT_CONFIG)
    assert cfg.sim.min_instances <= cfg.sim.max_instances
    assert cfg.billing.billing_granularity == "per_second"


def test_missing_file_fails_fast(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "does_not_exist.yaml")


def test_malformed_yaml_fails_fast(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("sim: [unclosed")
    with pytest.raises(ConfigError):
        load_config(bad)


def test_missing_required_field_fails_fast(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("sim: {}\n")
    with pytest.raises(ConfigError):
        load_config(cfg_path)


def test_min_greater_than_max_fails_fast(tmp_path, make_config):
    # build a full valid config then break the cross-field invariant
    cfg = make_config({"sim": {"min_instances": 10, "max_instances": 2}})
    with pytest.raises(ConfigError, match="min_instances"):
        cfg.sim.validate_bounds()


def test_negative_control_interval_fails_fast(make_config):
    with pytest.raises(Exception):  # pydantic.ValidationError
        make_config({"sim": {"control_interval_s": -1.0}})


def test_unknown_reward_weight_key_fails_fast(make_config):
    # a typo'd key (e.g. "cost_weight") must not silently fall back to defaults
    with pytest.raises(Exception, match="unknown reward_weights key"):
        make_config({"agent": {"reward_weights": {"cost_weight": 1.0}}})


def test_baselines_target_utilization_defaults_and_overrides(make_config):
    assert make_config().baselines.target_utilization == 0.7
    assert (
        make_config({"baselines": {"target_utilization": 0.5}}).baselines.target_utilization == 0.5
    )
