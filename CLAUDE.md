# CLAUDE.md — RL-EDoS

Persistent project context for Claude Code. Read this fully at the start of **every** session before writing code. It is authoritative; the `*.md` docs referenced below hold the detail.

---

## What this project is

**RL-EDoS** is a headless research testbed: a simulated cloud in which an attacker agent and a defender agent learn against each other, producing an autoscaling policy hardened against **Economic Denial of Sustainability (EDoS)** — attacks that keep the autoscaler over-provisioning so the cloud bill inflates while uptime looks fine.

**Success = one thing:** an RL defender that, under attack, beats `target_tracking` **and** `security_blind_rl` on **both** cost-under-attack **and** legit-traffic quality, *simultaneously*, shown in `results/<run_id>/comparison.csv` + plots. Everything serves that claim.

This is **not**: a production autoscaler, a sim-to-real project, a web app / dashboard / SaaS, a novel RL algorithm, or a real-traffic IDS. If a task drifts toward any of these, stop and flag it.

---

## The 8 rules (violating any of these is a defect)

1. **Phase order is strict: 1 → 2 → 3 → 4.** Never start a phase before the previous one's STOP has been reviewed and approved. Never start self-play (Phase 4) until the Phase 3 defender demonstrably beats baselines.
2. **STOP points are hard.** At a `→ STOP`, summarise what was built, give exact verify commands, and **wait**. Do not run into the next phase.
3. **Ask before installing** anything outside the fixed stack (below). Justify it against the baseline first.
4. **Determinism first.** Every stochastic component accepts and respects a `seed`. `CloudEnv.reset(seed)` must reproduce identical episodes. No uncontrolled randomness in env, training, or evaluation.
5. **Fidelity is non-negotiable.** Billing, cooldown, and warm-up constants come from real provider docs and are cited in a comment next to the constant. Never invent economics. See `05_BILLING_AND_FIDELITY.md`.
6. **No silent failures.** Validate configs on load (fail fast, clear message). Detect NaN/divergence in training and surface it loudly. Never persist a broken policy as valid.
7. **Test as you go.** Any new env / billing / detection / baseline / reward logic ships with a unit test in the same change.
8. **One consolidated config per experiment.** No magic numbers in code paths — everything flows through the config schema in `03_DATA_SCHEMAS.md`.

---

## Fixed stack — do NOT add without asking

Python **3.11** · **Gymnasium** (env API) · **Stable-Baselines3** PPO on **PyTorch** · **NumPy** · **Pandas** · Parquet/CSV + JSON/YAML persistence · SB3 `.zip` checkpoints · **TensorBoard** / Matplotlib / Seaborn. Dev: **black**, **ruff**, **pytest**.

**Must NOT:** hand-rolled RL algorithms (use SB3 PPO) · browser storage/localStorage · proprietary datasets · live cloud API calls (all simulated, offline) · paid services for the MVP · HTTP server, DB service, or auth (none exist here).

---

## Repo layout (module = responsibility; keep 1:1)

```
rl-edos/
  src/rl_edos/
    env/          # CloudEnv (gymnasium.Env), BillingModel, TrafficGenerator,
                  # DetectionModel, Autoscaler + queue/latency dynamics
                  # ← LOAD-BEARING CORRECTNESS LAYER. Nothing learns until this is verified.
    agents/       # SB3 PPO wrappers + obs/action/REWARD defs (defender; attacker in P4)
                  # ← all reward functions live HERE, never scattered in env/
    baselines/    # target_tracking, security_blind_rl, randomisation
                  # ← every baseline emits the SAME action contract as the RL defender
    training/     # train_defender; selfplay loop (P4)
    evaluation/   # Evaluator, metrics, plotting
    configs/      # YAML experiment configs (one consolidated file per experiment)
    cli.py        # command surface
  tests/          # unit tests (env, billing, detection, baselines, reward)
  results/        # generated artifacts — GITIGNORED
  README.md  pyproject.toml  LICENSE
```

Packaging: `src/` layout, package `rl_edos`, console entry point `rl-edos` → `rl_edos.cli:main`. Pin exact versions of gymnasium, stable-baselines3, torch, numpy, pandas, matplotlib, seaborn, pyyaml, pyarrow, and the config-validation lib.

---

## Interface contracts (honour these signatures exactly)

- `CloudEnv.reset(seed) -> (obs, info)` — fully seed all randomness; returns initial `EnvState` observation.
- `CloudEnv.step(action) -> (obs, reward, terminated, truncated, info)` — apply scaling action (defender) or traffic-shaping action (attacker); advance one control interval; `info` carries `cost`, `latency`, `detection_score`.
- `BillingModel.accrue(state) -> cost_delta` — cost for the interval under configured pricing/granularity; warming instances billed from launch.
- `DetectionModel.score(traffic_window) -> detection_score` — statistical evasion constraint; validatable in isolation.
- `Baselines.target_tracking(state) -> action` / `Baselines.randomisation(state) -> action` — same action contract as the RL defender.
- `Trainer.train(agent_cfg, env) -> policy_ckpt` — thin SB3 PPO wrapper.
- `Evaluator.run(controllers, attack, seeds) -> MetricsSummary` — all controllers under **identical** attack seeds/conditions.

**Defender obs:** vector from `EnvState` — arrival_rate, queue_len, active_instances, warming_instances, latency_ms, accrued-cost rate, detection_score.
**Defender action:** scaling decision (target instance count or delta), same contract as baselines.
**Defender reward:** `-(cost_w * cost_delta + latency_w * latency_penalty)`, weights from `AgentConfig.reward_weights`.
**Attacker (P4) action/reward:** traffic-shaping params bounded by detection; `cost_gain_w * cost_inflicted - evasion_w * detection_penalty`.

Exact schema fields and units are in `03_DATA_SCHEMAS.md` — treat it as the single source of truth for config/state/record shapes.

---

## CLI surface (the only user entry points)

```bash
rl-edos train-defender --config <path>                          # P3: train PPO defender vs scripted attacks
rl-edos evaluate --policy <path|none> --baselines <set> --config <path>  # run all controllers, identical attacks
rl-edos selfplay --config <path>                                # P4: co-evolutionary loop (attack.mode=learned)
rl-edos plot --run <run_id>                                     # regenerate curves / comparison / attack traces
```

Artifact layout per run: `results/<run_id>/comparison.csv`, `curves.png`, `attack_trace.png`, `summary.json`, `config.yaml` snapshot. Every `RunRecord` captures `config_hash` + `git_sha` + `seed`.

---

## Phases (build one at a time; full prompts in `phases/`)

| Phase | File | Deliverable | STOP gate |
|-------|------|-------------|-----------|
| 1 | `phases/PHASE_1_ENVIRONMENT_CORE.md` | Scaffold + `CloudEnv` + billing + traffic + detection + config validation + CLI stubs + tests | Sim is faithful & seeded; fidelity checklist ticked |
| 2 | `phases/PHASE_2_BASELINES_AND_EVAL.md` | Scripted attacks (steady/oscillation/burst) + baselines + `Evaluator` + metrics + plots | Baselines & metrics trustworthy; deterministic under fixed seeds |
| 3 | `phases/PHASE_3_RL_DEFENDER.md` | PPO defender + joint reward + train + headline comparison | **MVP:** defender wins both axes vs both baselines, reproducibly |
| 4 | `phases/PHASE_4_SELFPLAY_AND_RELEASE.md` | Self-play (stretch) + error handling + docs + reproducibility + release | Converged hardened defender **or** documented failure analysis; DoD met |

At each STOP, verify against the matching gate in `10_TESTING_STRATEGY.md` before proceeding.

---

## Definition of done (whole project)

- Runs locally from README: create env, install pinned deps, `train-defender` then `evaluate` on the default config, reproduce the headline comparison.
- MVP acceptance met (defender wins both axes vs `target_tracking` and `security_blind_rl`).
- Tests green for env dynamics, billing accrual, detection scoring, baselines, reward math.
- Threat model, billing assumptions, and sim-to-real gap documented (`docs/ASSUMPTIONS.md`).
- (Stretch) self-play run with a converged hardened defender **or** a documented stability/failure analysis.

---

## Conventions

- **Commits:** small and phase-scoped, e.g. `phase1(env): add billing accrual with per-second granularity`. **No `Co-Authored-By: Claude`** and no AI attribution anywhere.
- **Code:** PEP 8; type hints on public functions; `black` + `ruff` clean; docstrings on modules/public APIs stating inputs, outputs, and **units** (seconds, currency, ms). Prefer small testable functions over monoliths.
- **Gates that must stay green before any STOP:** `black .`, `ruff check .`, `pytest`.

---

## Session bootstrap

At the start of a session, state which phase we're on, confirm the previous STOP was approved, then implement **only** what that phase's `phases/PHASE_N_*.md` specifies. If unsure whether the prior gate passed, ask before writing code.