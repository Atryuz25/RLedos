# 04 — Tech Stack & Setup

## Stack (fixed — ask before adding anything)

- **Language:** Python 3.11
- **Simulation:** a `gymnasium.Env`; discrete-event scheduling via a plain step-loop (SimPy optional only if event complexity grows)
- **RL:** Stable-Baselines3 (PPO) on PyTorch
- **Numerics:** NumPy
- **Data / persistence:** Pandas; Parquet/CSV (metrics), JSON/YAML (configs), SB3 `.zip` (checkpoints); optional local SQLite only if a run registry is needed
- **Experiment tracking / viz:** TensorBoard (built into SB3), Matplotlib/Seaborn, Pandas. Optional Weights & Biases post-MVP
- **Frontend:** None. Optional post-MVP results view: Streamlit **or** static HTML + Matplotlib exports. No SPA framework
- **Auth / DB service / hosting:** None. Single-user local tool; the "cloud" is simulated; CPU sufficient, single GPU optional

### MUST use
Python, Gymnasium API, Stable-Baselines3, PyTorch. Real, documented provider pricing values for the billing model.

### Must NOT
- No hand-rolled RL algorithms (use SB3)
- No browser-storage/localStorage
- No proprietary datasets
- No live cloud API calls (everything simulated and offline)
- No dependency on paid services for the MVP

## Repo layout (create in Phase 1)

```
rl-edos/
  src/rl_edos/
    env/            # CloudEnv, billing, traffic, detection
    agents/         # PPO wrappers, obs/action/reward defs
    baselines/      # target-tracking, security-blind RL, randomisation
    training/       # train_defender, selfplay loop
    evaluation/     # evaluator, metrics, plotting
    configs/        # YAML experiment configs
    cli.py          # command surface
  tests/            # unit tests (env, billing, detection, baselines)
  results/          # generated run artifacts (gitignored)
  README.md
  pyproject.toml
  LICENSE           # open-source the testbed (recommended)
```

## `pyproject.toml` expectations

- Package name `rl_edos`, `src/` layout.
- Pin exact versions of gymnasium, stable-baselines3, torch, numpy, pandas, matplotlib, seaborn, pyyaml, pyarrow (parquet), and the config-validation lib (pydantic or use stdlib dataclasses).
- Dev deps: black, ruff, pytest.
- Console entry point mapping `rl-edos` → `rl_edos.cli:main`.

## Environment setup (document in README)

```bash
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
rl-edos --help
```

## Tooling gates (must stay green)

- `black .` — formatting clean
- `ruff check .` — lint clean
- `pytest` — all tests pass
- Type hints on all public functions; docstrings on modules and public APIs stating inputs/outputs and **units** (seconds, currency, ms)

## `.gitignore` essentials

`.venv/`, `results/`, `__pycache__/`, `*.zip` checkpoints outside a tracked models dir, TensorBoard logs, `.pytest_cache/`, `*.parquet` under results.

## Commit conventions

- **No `Co-Authored-By: Claude`** and no AI attribution in commit messages or code.
- Small, phase-scoped commits; message states the phase and the component (e.g. `phase1(env): add billing accrual with per-second granularity`).
