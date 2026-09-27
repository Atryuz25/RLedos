# CLAUDE.md — RL-EDoS

Persistent project context for Claude Code. Read this fully at the start of **every** session before writing code. It is authoritative; the `docs/*.md` files referenced below hold the detail.

**Current state (updated 2026-09-27, post plumbing-fix pass):** Phases 1–4 are **implemented and tested** (92 tests green; `black` + `ruff` clean). `train-defender`, `evaluate`, and `selfplay` are all real, wired CLI commands with working `Trainer` / self-play loop code, `VecNormalize`-based checkpoints, a NaN/divergence guard, real learning-curve capture, multi-seed evaluation with reported spread, and a working `evaluate` path against a trained self-play attacker (`attack.mode: learned`) — all verified via real (not mocked) end-to-end CLI runs, not just unit tests. **The Phase 3 MVP acceptance gate itself is still not met**: three real end-to-end runs on `configs/default.yaml` (10k, 150k, and a re-run 10k smoke test) all show `target_tracking` beating the RL defender on `mean_cost_under_attack` — the RL policies (both `security_blind_rl` and the joint-reward defender) converge to an over-conservative "keep the fleet near `max_instances`" policy. A concrete, evidence-backed likely cause has now been identified (not just "needs more budget"): `defender_reward`'s `cost_w`/`latency_w` terms are scaled roughly 50x apart given `default.yaml`'s real cost/latency magnitudes, so the policy gradient is dominated by latency almost regardless of cost. Full tried/result/cause writeup and the recommended retune + long-run command: `docs/PHASE_3_RL_DEFENDER.md`'s "Current status" section. Next step for anyone picking this up: apply that retune, run the recommended 500k-2M timestep training run, then re-run `evaluate --seeds ...` and check `comparison.csv` (mean **and** std) before declaring the gate passed.

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
    training/     # sb3_utils.py: train_ppo/train_ppo_normalized + rescale_action,
                  # shared by security_blind_rl (P2), train_defender.py::Trainer (P3),
                  # and selfplay.py + selfplay_envs.py (P4)
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
- `Trainer.train(config, out_dir) -> policy_ckpt` (`training/train_defender.py`) — thin wrapper over `training/sb3_utils.py::train_ppo_normalized`; `policy_ckpt` is a directory (`policy.zip` + `vecnormalize.pkl`), loaded back via `agents/defender.py::load_defender`.
- `Evaluator.run(controllers, attack, seeds) -> MetricsSummary` — all controllers under **identical** attack seeds/conditions.

> ### ⚠️ Action-space rescale — read before calling `.predict()`
> `train_ppo` / `train_ppo_normalized` wrap the env in `gymnasium.wrappers.RescaleAction(env, -1.0, 1.0)` before PPO sees it (SB3 wants a symmetric action space for its Gaussian policy; raw `CloudEnv` actions are `[min_instances, max_instances]`, e.g. `[1, 20]`; the Phase 4 attacker's are `[0, evasion_budget]`).
> **So `model.predict()` returns an action in `[-1, 1]`, not a real instance count / intensity.** Every caller MUST map it back with `training/sb3_utils.py::rescale_action(action, low, high)`. Existing call sites (`agents/defender.py`, `agents/attacker.py`, `baselines/security_blind_rl.py`, `training/selfplay_envs.py`, `cli.py`'s `--policy` path) already do. Skipping it fails *silently*: clipping a `[-1, 1]` value into `[min_instances, max_instances]` looks plausible while pinning the controller near `min_instances`.

> ### ⚠️ Observation normalization — read before evaluating a checkpoint
> `train_ppo_normalized` wraps training in `VecNormalize(norm_obs=True, norm_reward=False)` (`CloudEnv`'s obs space is `Box(low=0, high=inf)` on all 8 dims, which destabilises PPO's value function otherwise). `VecNormalize`'s running obs stats live *outside* the SB3 `.zip`, so every checkpoint is a **directory**: `policy.zip` + `vecnormalize.pkl`. `agents/defender.py::load_defender` / `agents/attacker.py::load_attacker` load both and normalize via `training/sb3_utils.py::normalize_obs` before `.predict()`. Loading only `policy.zip` (or evaluating with a raw obs) silently miscalibrates the policy — it still runs, it's just wrong.

**Defender obs:** vector from `EnvState.to_obs()` — arrival_rate, queue_len, active_instances, warming_instances, latency_ms, accrued-cost rate, detection_score (+ `OBS_DIM = 8` in `env/cloud_env.py`). `security_blind_rl` derives its dimension as `BLIND_OBS_DIM = OBS_DIM - 1` (detection_score is the last field, and it is dropped) — never re-hardcode it.
**Defender action:** scaling decision (target instance count or delta), same contract as baselines.
**Defender reward:** `-(cost_w * cost_delta + latency_w * latency_penalty)`, weights from `AgentConfig.reward_weights`; covered by `tests/test_reward.py`. Keys are validated against `config.py::KNOWN_REWARD_WEIGHT_KEYS` = `{cost_w, latency_w, cost_gain_w, evasion_w}` — a typo now fails on config load instead of silently reverting to defaults. Adding a weight means adding it to that set.
**Attacker (P4) action/reward:** traffic-shaping params bounded by detection; `cost_gain_w * cost_inflicted - evasion_w * detection_penalty`.

Exact schema fields and units are in `docs/03_DATA_SCHEMAS.md` — treat it as the single source of truth for config/state/record shapes.

---

## CLI surface (the only user entry points)

```bash
rl-edos train-defender --config <path> [--out <dir>]             # PPO defender vs scripted attack -> checkpoint dir
rl-edos evaluate --policy <path|none> --baselines <set> --config <path> [--seeds N [N ...]]  # run all controllers, identical attacks (single seed by default; averages + reports std across the given seeds otherwise)
rl-edos selfplay --config <path> [--out <dir>]                   # co-evolutionary attacker/defender loop
rl-edos plot --run <run_id>                                      # regenerate curves / comparison / attack traces
```

`evaluate`'s `attack.mode` (in the config, not a CLI flag): `scripted` (default) uses `AttackSpec.pattern`/`intensity` as before; `learned` requires `attack.attacker_checkpoint` (a `selfplay` attacker checkpoint dir) and evaluates every controller against that trained attacker's own observation-conditioned policy instead — validated fail-fast in `Evaluator.run()` (not `load_config`, since `selfplay` itself legitimately loads a `mode: learned` config with no checkpoint yet).

Artifact layout per run: `results/<run_id>/comparison.csv` (mean + `*_std` per metric, `n_seeds`), `curves.png` (real PPO training reward history when `--policy` names a checkpoint with one, else a placeholder), `train_rewards.csv` (copied from the checkpoint), `attack_trace.png`, `summary.json`, `config.yaml` snapshot. Every `RunRecord` captures `config_hash` + `git_sha` + `seed`. `train-defender` writes `models/<run_id>/{policy.zip,vecnormalize.pkl,train_rewards.csv,tensorboard_logs/}`; `selfplay` writes `models/selfplay/<run_id>/{defender,attacker}/{policy.zip,vecnormalize.pkl}` + `stability.json` + `stability.png`, and prints the exact follow-up `evaluate` invocation to test the hardened defender against its own trained attacker.

---

## Phases (full prompts in `docs/`)

| Phase | File | Deliverable | Status |
|-------|------|-------------|--------|
| 1 | `docs/PHASE_1_ENVIRONMENT_CORE.md` | Scaffold + `CloudEnv` + billing + traffic + detection + config validation + CLI stubs + tests | **DONE, audited** — sim faithful & seeded; fidelity checklist ticked |
| 2 | `docs/PHASE_2_BASELINES_AND_EVAL.md` | Scripted attacks (steady/oscillation/burst) + baselines + `Evaluator` + metrics + plots | **DONE, audited** — deterministic under fixed seeds |
| 3 | `docs/PHASE_3_RL_DEFENDER.md` | PPO defender + `Trainer` + joint reward + train + headline comparison | **GATE NOT MET (OPEN).** Code (`Trainer`, `VecNormalize` checkpointing, NaN guard, `evaluate --policy` integration, real learning-curve capture, multi-seed evaluation) is implemented and tested (92 tests, see `tests/test_training.py`, `tests/test_harness.py`, `tests/test_cli.py`). Three real runs on `configs/default.yaml` (10k, 150k, and a re-run 10k smoke test) all show `target_tracking` still beating the RL defender on `mean_cost_under_attack`. A concrete likely cause is now identified and documented (a ~50x scale mismatch between `cost_w`/`latency_w`'s effective contributions) — see `docs/PHASE_3_RL_DEFENDER.md`'s "Current status" section for the full tried/result/cause writeup and the recommended retune + long-run command. Needs that retune plus a much larger training budget before the MVP claim holds; re-run `evaluate --seeds ...` and check `comparison.csv` (mean + std) before declaring it passed. |
| 4 | `docs/PHASE_4_SELFPLAY_AND_RELEASE.md` | Self-play (stretch) + error handling + docs + reproducibility + release | **IMPLEMENTED, NOT GATED — results not valid until Phase 3 passes.** `run_selfplay` (alternating freeze/train, warm-started per round, `stability.json` + convergence/divergence handling) is implemented and tested (`tests/test_selfplay.py`) and smoke-tested end-to-end via `configs/selfplay_demo.yaml`. `attack.mode: learned` now actually works end-to-end: `evaluate` can run the defender + baselines against a trained self-play attacker checkpoint (`agents/attacker.py`, previously dead code, is now live via `evaluation/evaluator.py` and `training/selfplay.py::_evaluate_round`), verified via a real CLI smoke run (train → evaluate → selfplay → evaluate-vs-learned-attacker, all four commands, real checkpoints). This phase was built in the same commit as Phase 3, before the Phase 3 MVP gate passed — see the process note below. Treat any self-play run to date as a mechanical smoke test, not a research finding, until Phase 3 passes and self-play is re-run against a real defender. |

> ### ⚠️ Process note: the Phase 3 → Phase 4 gate was not honoured (logged 2026-09-27)
> Commit `0ea1b19` ("phase3+phase4: PPO defender training, self-play loop") built and committed Phase 4's self-play loop in the same commit as Phase 3's training pipeline, despite that same commit's message stating the Phase 3 MVP gate had not been met. This breaches Rule 1 ("never start self-play (Phase 4) until the Phase 3 defender demonstrably beats baselines") and Rule 2 ("STOP points are hard ... do not run into the next phase"). No code has been reverted over this — the self-play implementation is sound on its own merits and stays in the repo — but Phase 4's status is downgraded to **implemented, not gated** (see the table above) until Phase 3's gate passes for real and self-play is re-run against that defender. Any session picking this up: treat the Phase 3 STOP as still open, and do not report or build on a self-play result as a research finding until this is resolved.

> ### Resolved: the Phase 3 observation-normalization design note
> The `Box(low=0, high=inf)` unbounded-observation-space issue flagged before Phase 3 started is handled: `training/sb3_utils.py::train_ppo_normalized` wraps training in `VecNormalize(norm_obs=True, norm_reward=False)`, and `training/train_defender.py::Trainer` saves its running stats (`vecnormalize.pkl`) alongside the SB3 `.zip`. See the "Observation normalization" callout above for the load-time contract.

At each STOP, verify against the matching gate in `docs/10_TESTING_STRATEGY.md` before proceeding.

---

## Definition of done (whole project)

- Runs locally from README: create env, install pinned deps, `train-defender` then `evaluate` on the default config, reproduce the comparison. **Works today** — verified by an actual clean-venv run.
- MVP acceptance **not yet met**: defender does not currently win both axes vs `target_tracking` and `security_blind_rl` on `configs/default.yaml` at 10k or 150k timesteps (see the Phase 3 row above, and `docs/PHASE_3_RL_DEFENDER.md`'s "Current status" for the identified likely cause). The plumbing (training, checkpointing, evaluation integration, multi-seed stats, real learning curves, learned-attacker evaluation) is done and tested; the training result itself needs the documented reward retune plus more budget.
- Tests green for env dynamics, billing accrual, detection scoring, attacks, baselines, reward math, normalized training, the Trainer round-trip, multi-seed evaluation, real learning-curve capture, the learned-attacker evaluation path, and the self-play loop — **all covered today (92 passing)**.
- Threat model, billing assumptions, and sim-to-real gap documented (`docs/ASSUMPTIONS.md`).
- Self-play loop runs end-to-end (`configs/selfplay_demo.yaml`) and produces checkpoints + `stability.json` + `stability.png`; whether a given real run converges or reports a documented failure analysis depends on the underlying defender/attacker actually being trained enough to be interesting, per the Phase 3 caveat above. **This phase's status is "implemented, not gated"** — see the process note under Phases — since it was built before Phase 3's gate passed; do not count a self-play run against the Definition of Done until Phase 3 passes.

---

## Conventions

- **Commits:** small and phase-scoped, e.g. `phase1(env): add billing accrual with per-second granularity`. **No `Co-Authored-By: Claude`** and no AI attribution anywhere.
- **Code:** PEP 8; type hints on public functions; `black` + `ruff` clean; docstrings on modules/public APIs stating inputs, outputs, and **units** (seconds, currency, ms). Prefer small testable functions over monoliths.
- **Gates that must stay green before any STOP:** `black .`, `ruff check .`, `pytest`.

---

## Session bootstrap

At the start of a session, state which phase we're on, confirm the previous STOP was approved, then implement **only** what that phase's `docs/PHASE_N_*.md` specifies. Phases 1–4 are all code-complete, but **Phase 3's gate is still open and Phase 4 is not gated** (see the process note under Phases) — the open item is Phase 3's MVP training result, not new plumbing, and Phase 4 must not be treated as a passed milestone until Phase 3 passes. If unsure whether the prior gate passed, ask before writing code.
