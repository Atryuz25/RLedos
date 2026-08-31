# 07 — PHASE 2 PROMPT: Baselines + Scripted Attacks + Evaluation Harness

> Give this only after the Phase 1 STOP gate has passed. Keep `../11_CLAUDE_CODE_RULES.md` pinned. Prerequisite: a faithful, tested `CloudEnv` + billing + traffic + detection from Phase 1.

## Goal of this phase

Make measurement trustworthy **before** anything learns the defender. Provide reference controllers, a library of scripted attacks, and the evaluation harness that produces the comparison table and plots on the two success axes (cost-under-attack, legit-traffic quality).

## Scope (implement exactly this)

1. **Scripted attack library** (`env/` traffic + an attacks module): parameterised `steady`, `oscillation`, `burst` patterns per `AttackSpec`, injected through the Phase 1 attack-injection interface, bounded by `DetectionConfig`. Seeded and reproducible.
2. **Baseline controllers** (`baselines/`), all exposing the **same action contract** as the future RL defender:
   - `target_tracking` — classic autoscaling to a utilisation/latency target
   - `security_blind_rl` — an RL controller with a cost/latency reward but **no** detection/attack awareness (this is a controller stub whose training is a thin SB3 wrapper; it exists as a comparison point — a security-blind policy, trainable in this phase or trained on demand by the harness)
   - `randomisation` — the SIGMETRICS-style randomisation defence (**optional / cuttable**; two baselines suffice for the core claim — implement if time permits, else leave a clean interface stub)
3. **Evaluation harness** (`evaluation/`):
   - `Evaluator.run(controllers, attack, seeds) -> MetricsSummary` — runs every controller under **identical attack seeds/conditions**
   - computes `MetricsSummary` (`mean_cost_under_attack`, `p95_latency_ms`, `legit_drop_rate`, `overprovision_ratio`) per `../03_DATA_SCHEMAS.md`
   - writes `RunRecord`s and the artifact layout: `results/<run_id>/comparison.csv`, `curves.png`, `attack_trace.png`, `summary.json`, `config.yaml` snapshot
   - captures `git_sha` + `seed` in each `RunRecord`
4. **Plotting** (`evaluation/`): cost-vs-latency comparison, attack-pattern traces, and the learning-curve plotter (curves will be populated once a learner exists; support the interface now). Publication-style theme, colour-blind-safe palette, text `summary.json` alongside every chart.
5. **Wire CLI**: `evaluate --policy <path|none> --baselines <set> --config <path>` runs the harness over the baselines under a chosen scripted attack; `plot --run <run_id>` regenerates artifacts.
6. **Tests**: baseline controllers honour the action contract and behave sensibly under each attack; harness produces deterministic metrics under fixed seeds; metric math (p95, drop rate, overprovision ratio) unit-tested against hand-computed cases; artifact files are written with the expected schema.

## Determinism requirement (critical for a benchmark)

Identical config + seed → identical `comparison.csv` within tolerance. The whole point of Phase 2 is that these numbers are trustworthy, so seed propagation through attacks, controllers, and evaluation must be airtight.

## Tooling gates before STOP

- `black`/`ruff`/`pytest` green
- A demo run on the default config produces a `comparison.csv` + plots for the baselines under at least one scripted attack

## → STOP FOR REVIEW

Stop. Do **not** start Phase 3 (the RL defender). Present:
- the baseline comparison table + plots under the scripted attacks,
- evidence the harness is deterministic under fixed seeds,
- confirmation the metric definitions match `../03_DATA_SCHEMAS.md`.

Karthik verifies baselines and metrics are trustworthy (per `../10_TESTING_STRATEGY.md` Phase 2 gate) before unlocking Phase 3.
