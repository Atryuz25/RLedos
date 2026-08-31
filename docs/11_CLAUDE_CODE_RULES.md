# 11 — Standing Rules for Claude Code (pin in every session)

These are non-negotiable constraints for every session of the RL-EDoS build. Read before writing any code.

## Commits
- **No `Co-Authored-By: Claude`** and no AI attribution anywhere in commit messages or code.
- Small, phase-scoped commits. Message names the phase + component, e.g. `phase1(env): warm-up transition with cooldown guard`.

## Dependencies
- The stack is **fixed**: Python 3.11, Gymnasium, Stable-Baselines3, PyTorch, NumPy, Pandas, Matplotlib/Seaborn, PyYAML, pyarrow, config-validation lib, and dev tools (black/ruff/pytest).
- **Ask before installing any new dependency.** Justify it against the fixed stack. Default answer is no.

## Determinism first
- Every new component accepts and respects a **seed**. Never introduce uncontrolled randomness into env, training, or evaluation.
- Reproducibility is research-critical: identical config + seed → identical results within tolerance.

## Fidelity is non-negotiable
- Billing / cooldown / warm-up constants come from **real provider values, cited in comments** (see `05_BILLING_AND_FIDELITY.md`). Do not invent economics.
- Warming instances are billed from launch — this is the modelled attack surface, not an implementation detail to shortcut.

## Phase discipline
- Implement **strictly in phase order**. Do not start a later phase's work early.
- **Do not start self-play (Phase 4) before the Phase 3 defender beats baselines.**
- **Respect STOP points** — pause and hand back for review rather than running ahead. Each phase prompt ends at a STOP; honour it.

## Coding conventions
- PEP 8; type hints on public functions; `black` + `ruff` clean.
- Docstrings on modules and public APIs stating inputs/outputs **and units** (seconds, currency, ms).
- Prefer small, testable functions over monoliths.

## Testing
- Any new env / billing / detection / baseline / reward logic ships with a unit test.
- **No untested changes to reward or billing math.**

## No silent failures
- Validate configs on load; fail fast with clear messages.
- Detect and surface NaN / training divergence; never persist a broken policy as if valid.
- Non-convergent self-play degrades to a reported failure-analysis artifact, not a crash.

## Config discipline
- **Single consolidated config per experiment.** No hard-coded magic numbers in code paths — everything flows through the config schema in `03_DATA_SCHEMAS.md`.

## Scope guards (from the spec — don't drift)
- Not production autoscaler, not sim-to-real, not a web app / SaaS / dashboard product, not a novel RL algorithm, not a production IDS. Detection is a simulation constraint. The "cloud" is simulated — no live cloud API calls, offline only.

## Karthik's working preferences
- Brutal honesty on technical feedback; flag real problems directly.
- Answer the question actually asked; state assumptions in one line when something is ambiguous rather than reframing the task.
- DSA/code explanations, when they come up: brief explanation → code → complexity/essentials.
