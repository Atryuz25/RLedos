# CLAUDE.md — RL-EDoS

Persistent project context for Claude Code. Read this fully at the start of **every** session before writing code. It is authoritative; the `docs/*.md` files referenced below hold the detail.

**Current state:** Phases 1 and 2 are **done and audited** (53 tests green; `black` + `ruff` clean). Phase 3 has **not started** — `rl-edos train-defender` is still a CLI stub and `training/` holds only shared SB3 helpers, no `Trainer`. Read the Phase 3 note below before writing any training code.

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
4. **Determinism first.** Every stochastic component accepts and respects a `seed`. `CloudEnv.reset(seed)` must reproduce identical episodes. `train_ppo` sets `torch.use_deterministic_algorithms(True)` — it is meant to raise loudly rather than let a future change (GPU move, new op) silently produce irreproducible runs. Do not disable it to make something pass.
5. **Fidelity is non-negotiable.** Billing, cooldown, and warm-up constants come from real provider docs and are cited in a comment next to the constant. Never invent economics — and never fabricate a plausible-sounding citation. If there is no specific real source, say so honestly in the comment (see `instance_warmup_s` in `configs/default.yaml`, described as typical EC2 status-check timing, explicitly *not* a fixed AWS default). Detail in `docs/05_BILLING_AND_FIDELITY.md`.
6. **No silent failures.** Validate configs on load (fail fast, clear message). Detect NaN/divergence in training and surface it loudly. Never persist a broken policy as valid.
7. **Test as you go.** Any new env / billing / detection / baseline / reward logic ships with a unit test in the same change. Test behaviour as the env actually drives it, not just in isolation (e.g. `DetectionModel` sees a *growing* window for the first `detection.window - 1` steps of every episode, then a capped sliding one — `tests/test_detection.py` covers both phases through `CloudEnv`).
8. **One consolidated config per experiment.** No magic numbers in code paths — everything flows through the config schema in `docs/03_DATA_SCHEMAS.md`. Baseline tunables included: `target_tracking`'s setpoint is `baselines.target_utilization`, never a module constant.

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
                  # DetectionModel, attacks, Autoscaler + queue/latency dynamics
                  # ← LOAD-BEARING CORRECTNESS LAYER. Nothing learns until this is verified.
    agents/       # reward.py — obs/action/REWARD defs (defender; attacker in P4)
                  # ← all reward functions live HERE, never scattered in env/
    baselines/    # target_tracking, security_blind_rl, randomisation
                  # ← every baseline emits the SAME action contract as the RL defender
    training/     # sb3_utils.py: train_ppo + rescale_action — shared by the P2
                  # security_blind_rl baseline and the P3 Trainer. train_defender
                  # does NOT exist yet (Phase 3); selfplay loop is P4.
    evaluation/   # Evaluator, metrics, plotting, artifacts
    config.py     # pydantic ExperimentConfig (validated on load)
    configs/      # YAML experiment configs (one consolidated file per experiment)
    cli.py        # command surface
  tests/          # unit tests (env, billing, detection, attacks, baselines, reward,
                  # metrics, harness, training, config)
  docs/           # spec, architecture, schemas, fidelity, testing, assumptions, PHASE_1..4
  results/        # generated artifacts — GITIGNORED
  README.md  pyproject.toml  LICENSE
```

Packaging: `src/` layout, package `rl_edos`, console entry point `rl-edos` → `rl_edos.cli:main`. Pin exact versions of gymnasium, stable-baselines3, torch, numpy, pandas, matplotlib, seaborn, pyyaml, pyarrow, pydantic.

---

## Interface contracts (honour these signatures exactly)

- `CloudEnv.reset(seed) -> (obs, info)` — fully seed all randomness; returns initial `EnvState` observation.
- `CloudEnv.step(action) -> (obs, reward, terminated, truncated, info)` — apply scaling action (defender) or traffic-shaping action (attacker); advance one control interval; `info` carries `cost`, `latency`, `detection_score`.
- `BillingModel.accrue(active_instances, warming_instances, elapsed_s) -> cost_delta` — cost for the interval under configured pricing/granularity; warming instances billed from launch.
- `DetectionModel.score(traffic_window) -> detection_score` — statistical evasion constraint; validatable in isolation.
- `Baselines.target_tracking(state) -> action` / `Baselines.randomisation(state) -> action` — same action contract as the RL defender.
- `Trainer.train(agent_cfg, env) -> policy_ckpt` — thin wrapper over `training/sb3_utils.py::train_ppo` (P3; not built yet).
- `Evaluator.run(controllers, attack, seeds) -> MetricsSummary` — all controllers under **identical** attack seeds/conditions.

> ### ⚠️ Action-space rescale — read before calling `.predict()`
> `train_ppo` wraps the env in `gymnasium.wrappers.RescaleAction(env, -1.0, 1.0)` before PPO sees it (SB3 wants a symmetric action space for its Gaussian policy; raw `CloudEnv` actions are `[min_instances, max_instances]`, e.g. `[1, 20]`).
> **So `model.predict()` returns an action in `[-1, 1]`, not an instance count.** Every caller MUST map it back with `training/sb3_utils.py::rescale_action(action, low, high)`. Existing call sites (`baselines/security_blind_rl.py::SecurityBlindController.__call__`, `cli.py`'s `--policy` path) already do. Skipping it fails *silently*: clipping a `[-1, 1]` value into `[min_instances, max_instances]` looks plausible while pinning the controller near `min_instances`.

**Defender obs:** vector from `EnvState.to_obs()` — arrival_rate, queue_len, active_instances, warming_instances, latency_ms, accrued-cost rate, detection_score (+ `OBS_DIM = 8` in `env/cloud_env.py`). `security_blind_rl` derives its dimension as `BLIND_OBS_DIM = OBS_DIM - 1` (detection_score is the last field, and it is dropped) — never re-hardcode it.
**Defender action:** scaling decision (target instance count or delta), same contract as baselines.
**Defender reward:** `-(cost_w * cost_delta + latency_w * latency_penalty)`, weights from `AgentConfig.reward_weights`; covered by `tests/test_reward.py`. Keys are validated against `config.py::KNOWN_REWARD_WEIGHT_KEYS` = `{cost_w, latency_w, cost_gain_w, evasion_w}` — a typo now fails on config load instead of silently reverting to defaults. Adding a weight means adding it to that set.
**Attacker (P4) action/reward:** traffic-shaping params bounded by detection; `cost_gain_w * cost_inflicted - evasion_w * detection_penalty`.

Exact schema fields and units are in `docs/03_DATA_SCHEMAS.md` — treat it as the single source of truth for config/state/record shapes.

---

## CLI surface (the only user entry points)

```bash
rl-edos train-defender --config <path>                          # P3 STUB — not implemented yet
rl-edos evaluate --policy <path|none> --baselines <set> --config <path>  # run all controllers, identical attacks
rl-edos selfplay --config <path>                                # P4 STUB (attack.mode=learned)
rl-edos plot --run <run_id>                                     # regenerate curves / comparison / attack traces
```

Artifact layout per run: `results/<run_id>/comparison.csv`, `curves.png`, `attack_trace.png`, `summary.json`, `config.yaml` snapshot. Every `RunRecord` captures `config_hash` + `git_sha` + `seed`.

---

## Phases (build one at a time; full prompts in `docs/`)

| Phase | File | Deliverable | Status / STOP gate |
|-------|------|-------------|--------------------|
| 1 | `docs/PHASE_1_ENVIRONMENT_CORE.md` | Scaffold + `CloudEnv` + billing + traffic + detection + config validation + CLI stubs + tests | **DONE, audited** — sim faithful & seeded; fidelity checklist ticked |
| 2 | `docs/PHASE_2_BASELINES_AND_EVAL.md` | Scripted attacks (steady/oscillation/burst) + baselines + `Evaluator` + metrics + plots | **DONE, audited** — deterministic under fixed seeds |
| 3 | `docs/PHASE_3_RL_DEFENDER.md` | PPO defender + `Trainer` + joint reward + train + headline comparison | **NOT STARTED.** MVP: defender wins both axes vs both baselines, reproducibly |
| 4 | `docs/PHASE_4_SELFPLAY_AND_RELEASE.md` | Self-play (stretch) + error handling + docs + reproducibility + release | Converged hardened defender **or** documented failure analysis; DoD met |

> **⚠️ Open blocker for Phase 3 — design for this on day one, do not bolt it on.**
> `CloudEnv`'s observation space is `Box(low=0, high=inf)` on all 8 dims (arrival rate, queue length, accrued cost…). `gymnasium.utils.env_checker.check_env` flags it, and unbounded observations destabilise PPO's value function — a direct threat to the MVP claim. The fix is `VecNormalize(norm_obs=True, norm_reward=False)` — **never** normalise reward, it would change the joint cost+latency semantics the MVP is measured on. `VecNormalize`'s running stats live *outside* the SB3 `.zip`, so the `Trainer` / `policy_ckpt` format must save and load them alongside the checkpoint (`VecNormalize.save`/`.load`); get it wrong and you silently evaluate a normalised-obs policy on raw obs. Not retrofitted to `security_blind_rl` (Phase 2, no checkpoint-companion mechanism). **Full design note: `docs/PHASE_3_RL_DEFENDER.md`, item 2 (Trainer scope).**

At each STOP, verify against the matching gate in `docs/10_TESTING_STRATEGY.md` before proceeding.

---

## Definition of done (whole project)

- Runs locally from README: create env, install pinned deps, `train-defender` then `evaluate` on the default config, reproduce the headline comparison.
- MVP acceptance met (defender wins both axes vs `target_tracking` and `security_blind_rl`).
- Tests green for env dynamics, billing accrual, detection scoring, attacks, baselines, and reward math — **all covered today (53 passing)**; keep it that way as Phase 3 lands.
- Threat model, billing assumptions, and sim-to-real gap documented (`docs/ASSUMPTIONS.md`).
- (Stretch) self-play run with a converged hardened defender **or** a documented stability/failure analysis.

---

## Conventions

- **Commits:** small and phase-scoped, e.g. `phase1(env): add billing accrual with per-second granularity`. **No `Co-Authored-By: Claude`** and no AI attribution anywhere.
- **Code:** PEP 8; type hints on public functions; `black` + `ruff` clean; docstrings on modules/public APIs stating inputs, outputs, and **units** (seconds, currency, ms). Prefer small testable functions over monoliths.
- **Gates that must stay green before any STOP:** `black .`, `ruff check .`, `pytest`.

---

## Session bootstrap

At the start of a session, state which phase we're on, confirm the previous STOP was approved, then implement **only** what that phase's `docs/PHASE_N_*.md` specifies. Phases 1–2 are approved and audited; Phase 3 is next and unstarted. If unsure whether the prior gate passed, ask before writing code.
