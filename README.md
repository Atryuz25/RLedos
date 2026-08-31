# RL-EDoS

Adversarial reinforcement-learning defence against **Economic Denial of Sustainability (EDoS)** in cloud autoscaling.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
rl-edos --help
```

## Running tests

```bash
black --check .
ruff check .
pytest
```

## Status

**Phase 1 complete**: repo scaffold, config schemas, `CloudEnv` (instance pool, queue/latency,
autoscaler cooldown + warm-up), `BillingModel`, `TrafficGenerator`, `DetectionModel`, CLI skeleton,
unit tests. See `docs/PHASE_1_ENVIRONMENT_CORE.md` for scope and `docs/ASSUMPTIONS.md` for
modelling assumptions and fidelity sources.

**Phase 2 complete**: scripted attack library (steady/oscillation/burst), baseline controllers
(`target_tracking`, `security_blind_rl`; `randomisation` left as a clean stub, cuttable per spec),
the `Evaluator` + `MetricsSummary`/`RunRecord`, plotting (comparison, attack trace, learning-curve
placeholder), and `evaluate`/`plot` wired into the CLI. Run the headline demo:

```bash
rl-edos evaluate --config src/rl_edos/configs/default.yaml --baselines all
rl-edos plot --run <run_id from the line above>
```

`train-defender` (Phase 3) is not yet implemented.

## Build documentation set

The rest of this README indexes the build spec below.

Hand this entire folder to Claude Code in VS Code and drive the build **one phase at a time**. Read the files in order. The build has mandatory STOP points — do not skip ahead.

## What this project is (one line)

A simulated cloud testbed in which an attacker agent and a defender agent learn against each other, so the defender becomes an autoscaling policy hardened against economically-motivated (EDoS) attacks.

## Folder layout

```
rl-edos-docs/
  README.md                     # this file
  00_START_HERE.md              # workflow, phase order, STOP protocol — read first
  01_PROJECT_SPEC.md            # condensed spec (source of truth)
  02_ARCHITECTURE.md            # modules, boundaries, Gym interface contract
  03_DATA_SCHEMAS.md            # every config/state/record schema, typed, with units
  04_TECH_STACK_AND_SETUP.md    # pinned stack, repo layout, tooling
  05_BILLING_AND_FIDELITY.md    # real provider economics + citations (non-negotiable)
  10_TESTING_STRATEGY.md        # per-phase test requirements + acceptance gates
  11_CLAUDE_CODE_RULES.md       # standing constraints — pin in every session
  phases/
    PHASE_1_ENVIRONMENT_CORE.md      # scaffold + simulation core        → STOP
    PHASE_2_BASELINES_AND_EVAL.md    # baselines + attacks + eval harness → STOP
    PHASE_3_RL_DEFENDER.md           # PPO defender (MVP result)          → STOP
    PHASE_4_SELFPLAY_AND_RELEASE.md  # self-play (stretch) + polish       → STOP
```

> The four `phases/PHASE_*.md` files are the same consolidated build prompts referenced in `00_START_HERE.md` (there as `06`–`09`), placed here as individual per-phase files so you can hand Claude Code exactly one phase at a time.

## Reading / build order

1. `00_START_HERE.md` — how the set works and the STOP protocol.
2. `01`–`05` and `10`–`11` — the shared context Claude Code reads once, up front.
3. `phases/PHASE_1_ENVIRONMENT_CORE.md` — build, then STOP for review.
4. `phases/PHASE_2_BASELINES_AND_EVAL.md` — after Phase 1 gate passes.
5. `phases/PHASE_3_RL_DEFENDER.md` — after Phase 2 gate passes (this is the MVP milestone).
6. `phases/PHASE_4_SELFPLAY_AND_RELEASE.md` — after Phase 3 gate passes.

## Session bootstrap (paste at the top of each Claude Code session)

> You are building RL-EDoS. Read `11_CLAUDE_CODE_RULES.md`, `01_PROJECT_SPEC.md`, `02_ARCHITECTURE.md`, and `03_DATA_SCHEMAS.md` before writing code. We are on **Phase N**; implement only what `phases/PHASE_N_*.md` specifies, stop at its STOP point, and do not begin the next phase.

## Non-negotiables (detail in `11_CLAUDE_CODE_RULES.md`)

- No `Co-Authored-By: Claude` in commits.
- Ask before adding any dependency outside the fixed stack.
- Everything seeded and deterministic.
- Billing/cooldown/warm-up constants sourced from real provider docs and cited in comments.
- Build strictly in phase order; respect STOP points.
