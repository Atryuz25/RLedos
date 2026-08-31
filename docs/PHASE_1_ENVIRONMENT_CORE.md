# 06 — PHASE 1 PROMPT: Scaffold + Environment Core

> Hand this file to Claude Code together with `../11_CLAUDE_CODE_RULES.md`. Also ensure it has read `../01_PROJECT_SPEC.md`, `../02_ARCHITECTURE.md`, `../03_DATA_SCHEMAS.md`, `../04_TECH_STACK_AND_SETUP.md`, and `../05_BILLING_AND_FIDELITY.md`.

## Goal of this phase

Build the repo skeleton and the **simulation core** — the load-bearing correctness layer. Nothing learns yet. At the end, we must be able to trust that the simulated cloud is economically and physically faithful.

## Scope (implement exactly this, nothing more)

1. **Repo scaffold** per the layout in `../04_TECH_STACK_AND_SETUP.md`: `src/rl_edos/` package, `pyproject.toml` (pinned deps + dev tools + `rl-edos` entry point), `.gitignore`, `README.md` stub, `LICENSE`, `tests/`, empty `results/` (gitignored).
2. **Config schemas** (`env/` or a `config.py`): `SimConfig`, `BillingConfig`, `TrafficSpec`, `AttackSpec`, `DetectionConfig` as validated dataclasses/Pydantic models per `../03_DATA_SCHEMAS.md`, loaded from YAML, schema-checked, fail-fast on invalid input. One consolidated config file per experiment; no magic numbers in code.
3. **`CloudEnv` (`gymnasium.Env`)** with:
   - instance pool with `active_instances` / `warming_instances`, `min`/`max` bounds
   - request queue and a latency model
   - an autoscaler applying a scaling action, respecting `scale_cooldown_s`
   - `instance_warmup_s` delay moving instances warming → active
   - `reset(seed)` fully seeding all randomness; `step(action)` returning `(obs, reward, terminated, truncated, info)` with `info` carrying `cost`, `latency`, `detection_score`
   - `EnvState` per the schema, exposed as the observation
4. **`BillingModel`** per `../05_BILLING_AND_FIDELITY.md`: `accrue(state) -> cost_delta`, warming instances billed from launch, per-second granularity + 60s minimum, real cited constants, deterministic.
5. **`TrafficGenerator`**: legitimate traffic (`poisson`, `diurnal`) with `base_rate`, `diurnal_amplitude`, `noise_std`; seeded and reproducible. (Scripted *attack* traffic patterns arrive in Phase 2 — here provide the legit generator and the interface for attack injection.)
6. **`DetectionModel`**: `score(traffic_window) -> detection_score` (`rate_threshold` / `variance_threshold` over `window`). Must be validatable in isolation.
7. **CLI skeleton** (`cli.py`): argument parsing and command stubs (`train-defender`, `selfplay`, `evaluate`, `plot`) that load and validate a config and print a clear "not yet implemented in this phase" message where appropriate. `rl-edos --help` must work.
8. **Unit tests** (`tests/`): env dynamics (warm-up transition, cooldown enforcement, queue/latency behaviour, bounds), billing accrual (including warming-instance cost and 60s minimum), detection scoring on labelled legit/attack windows, config validation (valid loads, invalid fails fast).

## Reward in this phase

A placeholder/joint cost+latency reward may be wired so `step` returns a number, but the **defender reward tuning is Phase 3**. Keep the reward function in `agents/` even if minimal, per architecture.

## Determinism & fidelity requirements

- Every stochastic component accepts and respects the seed; `reset(seed)` reproduces identical episodes.
- Billing/cooldown/warm-up constants cite real sources in comments (see `05`).
- No network calls, no external data, no hand-rolled RL.

## Tooling gates before STOP

- `black .` clean, `ruff check .` clean, `pytest` all green
- Type hints + unit-stating docstrings on public functions
- README documents setup and how to run the tests

## → STOP FOR REVIEW

Stop here and hand back. Do **not** start Phase 2 (baselines/attacks/eval). Present:
- a short summary of the env dynamics and billing model with the cited constants,
- test output showing the fidelity gates pass,
- the list from the **Fidelity checklist** in `../05_BILLING_AND_FIDELITY.md`, each item ticked.

Karthik verifies the sim is faithful (per `../10_TESTING_STRATEGY.md` Phase 1 gate) before unlocking Phase 2.
