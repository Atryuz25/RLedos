"""Config schema validation: valid loads, fail-fast on invalid input."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
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


def test_selfplay_defaults_and_overrides(make_config):
    cfg = make_config()
    assert cfg.selfplay.rounds == 3
    assert cfg.selfplay.round_timesteps == 5_000

    overridden = make_config({"selfplay": {"rounds": 1, "round_timesteps": 64}})
    assert overridden.selfplay.rounds == 1
    assert overridden.selfplay.round_timesteps == 64


def test_selfplay_zero_rounds_fails_fast(make_config):
    with pytest.raises(Exception):  # pydantic.ValidationError
        make_config({"selfplay": {"rounds": 0}})


def test_learned_attack_mode_without_checkpoint_fails_fast(make_config):
    cfg = make_config({"attack": {"mode": "learned", "evasion_budget": 1.0}})
    with pytest.raises(ConfigError, match="attacker_checkpoint"):
        cfg.attack.validate_learned_attacker()


def test_learned_attack_mode_with_missing_checkpoint_file_fails_fast(make_config, tmp_path):
    cfg = make_config(
        {
            "attack": {
                "mode": "learned",
                "evasion_budget": 1.0,
                "attacker_checkpoint": str(tmp_path / "does_not_exist"),
            }
        }
    )
    with pytest.raises(ConfigError, match="policy.zip"):
        cfg.attack.validate_learned_attacker()


def test_scripted_attack_mode_ignores_missing_attacker_checkpoint(make_config):
    # attacker_checkpoint is only required/validated when mode == "learned"
    cfg = make_config({"attack": {"mode": "scripted"}})
    cfg.attack.validate_learned_attacker()  # must not raise


def test_load_config_accepts_learned_mode_without_a_checkpoint_yet(tmp_path, make_config):
    # `selfplay --config <path>` legitimately loads a mode: learned config with
    # no attacker_checkpoint set yet -- it's what *produces* that checkpoint.
    # validate_learned_attacker must NOT be a load_config-time check (that
    # would break `selfplay` outright); it belongs to Evaluator.run() instead.
    cfg_path = tmp_path / "selfplay_style.yaml"
    cfg = make_config({"attack": {"mode": "learned", "evasion_budget": 1.0}})
    cfg_path.write_text(yaml.safe_dump(cfg.model_dump()))

    loaded = load_config(cfg_path)  # must not raise
    assert loaded.attack.mode == "learned"
    assert loaded.attack.attacker_checkpoint is None
