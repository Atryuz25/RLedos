# 02 — Architecture

## System shape

RL-EDoS is a **headless research system**: no web server, no database service, no auth. The "cloud" is simulated. Everything runs locally from a CLI over config files, producing file artifacts.

The programmatic contract is the **Gymnasium environment interface** plus internal module APIs — there is no HTTP layer.

## High-level data flow

```
                 config (YAML)
                     │
                     ▼
        ┌────────────────────────────┐
        │        CloudEnv            │  gymnasium.Env
        │  ┌──────────────────────┐  │
  step  │  │ TrafficGenerator     │  │  legit (diurnal+poisson) + attack traffic
 ─────▶ │  │ Autoscaler + Queue   │  │  instance pool, warm-up, cooldown, latency
        │  │ BillingModel         │  │  accrued cost per interval
        │  │ DetectionModel       │  │  detection_score (evasion constraint)
        │  └──────────────────────┘  │
        └──────────┬─────────────────┘
                   │ (obs, reward, terminated, truncated, info)
      ┌────────────┼───────────────────────────┐
      ▼            ▼                            ▼
  RL Defender   Baselines                   RL Attacker (Phase 4)
  (PPO/SB3)     target-tracking,            (PPO/SB3, learned mode)
                security-blind RL,
                randomisation
      │            │                            │
      └────────────┴────────────┬───────────────┘
                                ▼
                          Evaluator
                   (identical seeds/attacks)
                                │
                                ▼
                   results/<run_id>/  comparison.csv, curves.png,
                                      attack_trace.png, summary.json
```

## Module boundaries (map 1:1 to `src/rl_edos/`)

- **`env/`** — the simulation. `CloudEnv` (the `gymnasium.Env`), `BillingModel`, `TrafficGenerator`, `DetectionModel`, `Autoscaler`/queue dynamics. This is the load-bearing correctness layer; nothing learns until this is verified (Phase 1 STOP).
- **`agents/`** — PPO wrappers (SB3) and the observation-space, action-space, and reward definitions for defender and (Phase 4) attacker. Reward functions live here, not scattered in the env.
- **`baselines/`** — reference controllers exposing the **same action contract** as the RL defender: `target_tracking`, `security_blind_rl`, `randomisation`.
- **`training/`** — `train_defender` and the `selfplay` loop (Phase 4).
- **`evaluation/`** — `Evaluator`, metrics computation, plotting.
- **`configs/`** — YAML experiment configs (one consolidated file per experiment).
- **`cli.py`** — the command surface (`train-defender`, `selfplay`, `evaluate`, `plot`).

## The environment interface contract (the "API")

There is no HTTP API. These are the programmatic contracts every component must honour.

- **`CloudEnv.reset(seed) -> (obs, info)`** — initialise a simulation episode; returns initial `EnvState` observation. Must fully seed all randomness.
- **`CloudEnv.step(action) -> (obs, reward, terminated, truncated, info)`** — apply a scaling action (defender) or traffic-shaping action (attacker); advance one control interval; return next state, reward, and `info` carrying `cost`, `latency`, `detection_score`.
- **`BillingModel.accrue(state) -> cost_delta`** — compute cost for the interval under configured pricing/granularity.
- **`DetectionModel.score(traffic_window) -> detection_score`** — statistical evasion constraint; input traffic window → scalar score used in the attacker penalty.
- **`Baselines.target_tracking(state) -> action`**, **`Baselines.randomisation(state) -> action`** — reference controllers exposing the same action contract as the RL defender.
- **`Trainer.train(agent_cfg, env) -> policy_ckpt`** — wraps SB3 PPO; input config + env → saved policy.
- **`Evaluator.run(controllers, attack, seeds) -> MetricsSummary`** — batch-evaluate controllers under identical conditions → metrics + artifact paths.

## Observation / action / reward (design intent)

- **Observation (defender):** a vector derived from `EnvState` — arrival rate, queue length, active/warming instances, latency, accrued cost rate, detection score. Exact shape is fixed in `agents/` and documented in `03_DATA_SCHEMAS.md`.
- **Action (defender):** a scaling decision (e.g. target instance count or a delta), same contract the baselines emit.
- **Reward (defender):** jointly balances cost and latency — `-(cost_w * cost_delta + latency_w * latency_penalty)`. Exact form and weights are config-driven (`AgentConfig.reward_weights`).
- **Action (attacker, Phase 4):** traffic-shaping parameters bounded by the detection constraint.
- **Reward (attacker, Phase 4):** inflate defender cost while staying under the detection threshold — `cost_gain_w * cost_inflicted - evasion_w * detection_penalty`.

## Reproducibility architecture

Seed flows top-down: CLI/config seed → `CloudEnv.reset(seed)` → all stochastic components (traffic Poisson draws, noise, SB3 policy init/rollout). A `RunRecord` captures `config_hash`, git SHA, and seed so any run is reconstructable. No component may introduce uncontrolled randomness.

## What is explicitly out of the architecture

No database service, no HTTP/REST layer, no authentication, no live cloud API calls, no browser storage, no SPA. Persistence is files only (Parquet/CSV/JSON/YAML + SB3 `.zip` checkpoints).
