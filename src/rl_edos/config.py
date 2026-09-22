"""Config schemas for RL-EDoS experiments.

All numeric fields carry explicit units in their docstring/description, per
docs/03_DATA_SCHEMAS.md. Everything a run needs flows through
``ExperimentConfig`` loaded from a single consolidated YAML file — no magic
numbers in code paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator


class ConfigError(Exception):
    """Raised when an experiment config fails validation. Fails fast with a clear message."""


# The only reward_weights keys any reward function reads (agents/reward.py, defender today;
# cost_gain_w/evasion_w reserved for the Phase 4 attacker reward). An unrecognised key is almost
# always a typo, and silently falling back to defaults would corrupt a training run with no error.
KNOWN_REWARD_WEIGHT_KEYS = {"cost_w", "latency_w", "cost_gain_w", "evasion_w"}


class SimConfig(BaseModel):
    """Simulation timing/topology parameters. All time fields are seconds."""

    control_interval_s: float = Field(gt=0, description="seconds between scaling decisions")
    sim_horizon_steps: int = Field(gt=0, description="number of control intervals per episode")
    instance_warmup_s: float = Field(
        ge=0, description="seconds a new instance spends warming before serving"
    )
    scale_cooldown_s: float = Field(
        ge=0, description="seconds after a scaling action before another is allowed"
    )
    max_instances: int = Field(gt=0, description="ceiling on active instances")
    min_instances: int = Field(ge=0, description="floor on active instances")
    seed: int = Field(description="master seed for the episode")
    capacity_rps_per_instance: float = Field(
        gt=0, description="requests/second one active instance can serve (queueing model input)"
    )
    base_latency_ms: float = Field(
        ge=0, description="fixed per-request latency floor before queueing delay, milliseconds"
    )
    max_queue_len: float = Field(
        default=float("inf"),
        ge=0,
        description="backlog ceiling, requests; overflow beyond this is dropped rather than queued",
    )

    def validate_bounds(self) -> None:
        if self.min_instances > self.max_instances:
            raise ConfigError(
                f"min_instances ({self.min_instances}) must be <= max_instances "
                f"({self.max_instances})"
            )


class BillingConfig(BaseModel):
    """Billing economics. Currency unit is a single abstract unit (e.g. USD), documented once here.

    price_per_instance_second default derives from AWS EC2 t3.medium on-demand,
    us-east-1: $0.0416/hour -> 0.0416 / 3600 ~= 0.00001156 currency/instance-second.
    Source: https://aws.amazon.com/ec2/pricing/on-demand/ (re-verify at build time; prices drift).
    """

    price_per_instance_second: float = Field(
        gt=0, description="currency per instance-second (derived from real per-hour pricing)"
    )
    billing_granularity: Literal["per_second", "per_hour"] = Field(
        description="provider billing model"
    )
    min_service_charge: float = Field(
        ge=0,
        description=(
            "currency floor charged per existing instance per interval, modelling the AWS "
            "60-second minimum billing duration (per-second billing, min 60s: "
            "https://aws.amazon.com/ec2/pricing/)"
        ),
    )


class TrafficSpec(BaseModel):
    """Legitimate traffic generation parameters."""

    legit_pattern: Literal["poisson", "diurnal"] = Field(description="legitimate traffic shape")
    base_rate: float = Field(gt=0, description="requests/second baseline")
    diurnal_amplitude: float = Field(
        ge=0, le=1, description="fractional swing of the diurnal cycle"
    )
    noise_std: float = Field(ge=0, description="std-dev of arrival noise, requests/second")


class AttackSpec(BaseModel):
    """Attack traffic parameters, constrained by DetectionConfig.

    Scripted patterns land in Phase 2.
    """

    mode: Literal["scripted", "learned"] = Field(
        default="scripted", description="scripted for MVP, learned for self-play"
    )
    pattern: Literal["steady", "oscillation", "burst"] = Field(
        default="steady", description="scripted attack shape"
    )
    intensity: float = Field(
        default=0.0, ge=0, description="attack magnitude relative to base_rate"
    )
    evasion_budget: float = Field(
        default=0.0, ge=0, description="how hard the attacker may push against detection"
    )


class DetectionConfig(BaseModel):
    """Statistical evasion-constraint detector. Not a production IDS."""

    type: Literal["rate_threshold", "variance_threshold"] = Field(
        description="statistical detector"
    )
    window: int = Field(gt=0, description="number of intervals in the detection window")
    threshold: float = Field(gt=0, description="detector trip level")
    penalty: float = Field(ge=0, description="penalty applied to attacker reward on detection")


class AgentConfig(BaseModel):
    """PPO agent configuration (defender; attacker reuses this shape in Phase 4)."""

    algo: Literal["PPO"] = "PPO"
    obs_space_spec: dict = Field(
        default_factory=dict, description="observation space definition (fixed in agents/)"
    )
    action_space_spec: dict = Field(
        default_factory=dict, description="action space definition (fixed in agents/)"
    )
    reward_weights: dict[str, float] = Field(
        default_factory=lambda: {"cost_w": 1.0, "latency_w": 0.01},
        description="{cost_w, latency_w} for defender; + {cost_gain_w, evasion_w} for attacker",
    )
    total_timesteps: int = Field(default=10_000, gt=0, description="SB3 training budget")

    @field_validator("reward_weights")
    @classmethod
    def _known_reward_weight_keys(cls, v: dict[str, float]) -> dict[str, float]:
        unknown = set(v) - KNOWN_REWARD_WEIGHT_KEYS
        if unknown:
            raise ValueError(
                f"unknown reward_weights key(s) {sorted(unknown)}; "
                f"expected a subset of {sorted(KNOWN_REWARD_WEIGHT_KEYS)}"
            )
        return v

    policy_net: list[int] = Field(
        default_factory=lambda: [64, 64], description="hidden-layer sizes"
    )


class SelfPlayConfig(BaseModel):
    """Co-evolutionary self-play loop parameters (Phase 4, stretch)."""

    rounds: int = Field(
        default=3, gt=0, description="number of attacker/defender freeze-train alternations"
    )
    round_timesteps: int = Field(
        default=5_000, gt=0, description="SB3 PPO training budget per agent, per round"
    )
    convergence_tolerance: float = Field(
        default=0.05,
        ge=0,
        description=(
            "relative change in mean_cost_under_attack between consecutive rounds "
            "below which the pair is considered stable"
        ),
    )
    convergence_patience: int = Field(
        default=2,
        gt=0,
        description="consecutive stable rounds required before declaring convergence",
    )


class BaselineConfig(BaseModel):
    """Reference-controller parameters (not PPO agents, hence kept out of AgentConfig)."""

    target_utilization: float = Field(
        default=0.7,
        gt=0,
        description="target_tracking's utilisation setpoint (AWS ASG target-tracking style)",
    )


class ExperimentConfig(BaseModel):
    """The single consolidated config for one experiment run."""

    sim: SimConfig
    billing: BillingConfig
    traffic: TrafficSpec
    attack: AttackSpec = Field(default_factory=AttackSpec)
    detection: DetectionConfig
    agent: AgentConfig = Field(default_factory=AgentConfig)
    baselines: BaselineConfig = Field(default_factory=BaselineConfig)
    selfplay: SelfPlayConfig = Field(default_factory=SelfPlayConfig)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate a consolidated experiment config from YAML.

    Fails fast with a clear ConfigError on missing file, malformed YAML, or
    schema violations.
    """
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"config file not found: {p}")
    try:
        raw = yaml.safe_load(p.read_text())
    except yaml.YAMLError as exc:
        raise ConfigError(f"malformed YAML in {p}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"config in {p} must be a YAML mapping at the top level")
    try:
        cfg = ExperimentConfig(**raw)
    except ValidationError as exc:
        raise ConfigError(f"invalid config in {p}:\n{exc}") from exc
    cfg.sim.validate_bounds()
    return cfg
