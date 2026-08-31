# 03 — Data Schemas

No relational DB. These are the simulation entities, their state/config schemas, and the persisted experiment records. Implement config schemas as validated dataclasses or Pydantic models (choose one and be consistent). **Every numeric field carries a unit — state it in the docstring.** No magic numbers in code paths; everything flows through these schemas.

## Config schemas

### SimConfig
| Field | Type | Unit / notes |
|-------|------|--------------|
| `control_interval_s` | float | seconds between scaling decisions |
| `sim_horizon_steps` | int | number of control intervals per episode |
| `instance_warmup_s` | float | seconds a new instance spends warming before serving |
| `scale_cooldown_s` | float | seconds after a scaling action before another is allowed |
| `max_instances` | int | ceiling on active instances |
| `min_instances` | int | floor on active instances |
| `seed` | int | master seed for the episode |

### BillingConfig (parameterises SimConfig)
| Field | Type | Unit / notes |
|-------|------|--------------|
| `price_per_instance_second` | float | currency per instance-second (derive from real per-hour pricing; cite source) |
| `billing_granularity` | enum{per_second, per_hour} | provider billing model |
| `min_service_charge` | float | currency; floor charge if the provider imposes one |

### TrafficSpec
| Field | Type | Unit / notes |
|-------|------|--------------|
| `legit_pattern` | enum{poisson, diurnal} | legitimate traffic shape |
| `base_rate` | float | requests/second baseline |
| `diurnal_amplitude` | float | fractional swing of the diurnal cycle |
| `noise_std` | float | std-dev of arrival noise |

### AttackSpec (constrained by DetectionConfig)
| Field | Type | Unit / notes |
|-------|------|--------------|
| `mode` | enum{scripted, learned} | scripted for MVP, learned for self-play |
| `pattern` | enum{steady, oscillation, burst} | scripted attack shapes |
| `intensity` | float | attack magnitude relative to base_rate |
| `evasion_budget` | float | how hard the attacker may push against detection |

### DetectionConfig
| Field | Type | Unit / notes |
|-------|------|--------------|
| `type` | enum{rate_threshold, variance_threshold} | statistical detector |
| `window` | int | number of intervals in the detection window |
| `threshold` | float | detector trip level |
| `penalty` | float | penalty applied to attacker reward on detection |

### AgentConfig (attacker / defender)
| Field | Type | Unit / notes |
|-------|------|--------------|
| `algo` | "PPO" | fixed |
| `obs_space_spec` | spec | observation space definition |
| `action_space_spec` | spec | action space definition |
| `reward_weights` | dict | `{cost_w, latency_w}` for defender; `+ {cost_gain_w, evasion_w}` for attacker |
| `total_timesteps` | int | SB3 training budget |
| `policy_net` | list[int] | hidden-layer sizes |

## Runtime state

### EnvState (per step)
| Field | Type | Unit / notes |
|-------|------|--------------|
| `t` | float | seconds since episode start |
| `arrival_rate` | float | requests/second this interval |
| `queue_len` | int | pending requests |
| `active_instances` | int | serving instances |
| `warming_instances` | int | instances warming up |
| `latency_ms` | float | milliseconds; service latency this interval |
| `accrued_cost` | float | currency; cumulative cost so far |
| `detection_score` | float | scalar detector output this interval |

## Persisted records

### RunRecord
| Field | Type | Unit / notes |
|-------|------|--------------|
| `run_id` | uuid | unique run identifier |
| `config_hash` | str | hash of the resolved config |
| `controller` | enum{rl_defender, target_tracking, security_blind_rl, randomisation} | which controller produced this run |
| `attack` | AttackSpec | attack used |
| `metrics` | MetricsSummary | see below |
| `artifacts` | list[path] | generated file paths |
| `created_at` | timestamp | run time |
| `git_sha` | str | code version (add for reproducibility) |
| `seed` | int | master seed (add for reproducibility) |

### MetricsSummary (belongs to RunRecord)
| Field | Type | Unit / notes |
|-------|------|--------------|
| `mean_cost_under_attack` | float | currency; cost axis of success |
| `p95_latency_ms` | float | milliseconds; quality axis of success |
| `legit_drop_rate` | float | fraction of legitimate requests dropped |
| `overprovision_ratio` | float | provisioned capacity ÷ needed capacity |
| `reward_curve_ref` | path | pointer to the learning-curve artifact |

## Artifact layout (predictable, per run)

```
results/<run_id>/
  comparison.csv      # controller × {mean_cost_under_attack, p95_latency_ms, legit_drop_rate, overprovision_ratio}
  curves.png          # learning curves
  attack_trace.png    # attack-pattern visualisation
  summary.json        # text summary accompanying every chart (colour-independent legibility)
  config.yaml         # resolved config snapshot for this run
  report.html         # (post-MVP, optional) aggregated view
```

## Persistence formats

- Metrics → **Parquet/CSV**
- Run configs → **JSON/YAML**
- Model checkpoints → SB3 **`.zip`**
- Optional run registry → local **SQLite** (only if needed; not required for MVP)
