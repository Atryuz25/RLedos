# 10 — Testing Strategy & Acceptance Gates

Testing is phase-gated. Each STOP has an acceptance gate that must pass before the next phase unlocks. Rule: **any new env/billing/detection/baseline/reward logic ships with a unit test; no untested changes to reward or billing math.**

## Standing test principles

- **Determinism is testable:** fixed seed → identical outputs. Assert this directly wherever randomness exists.
- **Units are contracts:** test that billing is in the documented currency unit, latency in ms, time in seconds — mismatches are bugs.
- **Fail-fast is a feature:** test that invalid configs raise clear errors and that NaN/divergence is surfaced, not swallowed.
- **Hand-computed oracles:** for metric math (p95, drop rate, overprovision ratio, cost accrual), test against small hand-calculated cases, not just "runs without error".

## Phase 1 gate — Simulation fidelity

Unit tests required:
- **Env dynamics:** warm-up transition (warming→active after `instance_warmup_s`), cooldown enforcement (no scaling within `scale_cooldown_s`), queue/latency response to load, min/max instance bounds respected.
- **Billing accrual:** warming instances billed from launch; per-second granularity; 60-second minimum; deterministic for identical state+config; matches a hand-computed cost for a fixed scenario.
- **Detection scoring:** correct scores on labelled legit vs attack windows for both `rate_threshold` and `variance_threshold`.
- **Config validation:** valid YAML loads; each invalid field fails fast with a clear message.
- **Reproducibility:** `reset(seed)` produces identical episodes across runs.

**Gate to pass Phase 1:** all above green + the Fidelity checklist in `05_BILLING_AND_FIDELITY.md` fully ticked + cited constants present in comments.

## Phase 2 gate — Trustworthy baselines & metrics

Unit tests required:
- **Baselines:** each honours the shared action contract; sensible behaviour under `steady`/`oscillation`/`burst`.
- **Attacks:** scripted patterns are seeded/reproducible and respect `DetectionConfig` bounds.
- **Metric math:** `mean_cost_under_attack`, `p95_latency_ms`, `legit_drop_rate`, `overprovision_ratio` match hand-computed cases.
- **Harness determinism:** identical config+seed → identical `comparison.csv` within tolerance.
- **Artifacts:** `comparison.csv`, `summary.json`, plots, `config.yaml` snapshot written with expected schema; `RunRecord` captures `git_sha` + `seed`.

**Gate to pass Phase 2:** all green + a demo comparison over baselines under ≥1 scripted attack, shown deterministic.

## Phase 3 gate — MVP success milestone

Unit tests required:
- **Reward:** correct sign/scaling for cost and latency components; weight application correct.
- **Training smoke test:** tiny-budget PPO run completes, yields a loadable checkpoint, no NaN.
- **Divergence/NaN guard:** a forced-divergence case is detected and reported; no broken policy saved as valid.
- **Integration:** trained policy evaluates through the same Phase 2 harness.

**Gate to pass Phase 3 (the big one):** RL defender beats target-tracking **and** security-blind RL on **both** cost-under-attack and legit-traffic quality in `comparison.csv` + plots, reproducibly. This is the MVP acceptance.

## Phase 4 gate — Self-play & release readiness

Unit tests required:
- **Self-play round-trip:** tiny-budget loop trains attacker then defender, checkpoints both, logs stability.
- **Non-convergence path:** produces the failure-analysis artifact instead of raising.
- **Release hygiene:** clean-env install works; `black`/`ruff`/`pytest` all green; reproduction path in README verified.

**Gate (final):** result maps cleanly onto the Definition of done in `01_PROJECT_SPEC.md` — converged hardened defender **or** documented failure analysis, full comparison, verified reproduction, complete docs.

## Suggested test tree

```
tests/
  test_config.py          # schema validation, fail-fast
  test_env_dynamics.py     # warm-up, cooldown, queue/latency, bounds
  test_billing.py          # accrual, warming cost, 60s min, determinism
  test_detection.py        # rate/variance scoring on labelled windows
  test_baselines.py        # action contract, behaviour under attacks
  test_metrics.py          # p95, drop rate, overprovision, cost — oracles
  test_harness.py          # determinism, artifact schema, RunRecord
  test_reward.py           # defender/attacker reward sign & scaling
  test_training_smoke.py   # tiny PPO run, loadable ckpt, no NaN
  test_selfplay.py         # round-trip, stability log, non-convergence path
```
