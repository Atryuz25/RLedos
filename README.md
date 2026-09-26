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

**Phase 3 implemented**: `Trainer` (`training/train_defender.py`) trains a PPO defender against
scripted attacks with `VecNormalize`-based observation normalization (`training/sb3_utils.py`)
and a NaN/divergence guard; checkpoints are a directory (`policy.zip` + `vecnormalize.pkl`) loaded
back via `agents/defender.py`. Run it:

```bash
rl-edos train-defender --config src/rl_edos/configs/default.yaml --out models/defender
rl-edos evaluate --config src/rl_edos/configs/default.yaml --policy models/defender --baselines all
```

**MVP acceptance gate not yet met.** The spec's success condition is the defender beating both
`target_tracking` and `security_blind_rl` on cost *and* latency simultaneously. Two real runs
against the default config (10k and 150k PPO timesteps) both show `target_tracking` still winning
on `mean_cost_under_attack` — the RL policies converge to an over-conservative near-`max_instances`
policy rather than an efficient one. This looks like a training-budget/reward-tuning gap, not a
plumbing bug (the checkpoint/normalization round-trip is unit- and integration-tested). Next step:
a substantially larger `total_timesteps` run and/or reward-weight retuning.

**Phase 4 implemented, not gated** — results not valid until Phase 3 passes: the co-evolutionary
self-play loop (`training/selfplay.py`, `training/selfplay_envs.py`) alternates freeze/train
between a PPO attacker and defender, warm-starting each round, with convergence detection and a
graceful non-convergence path (`stability.json` + `stability.png` are always written, never a
crash). Demo:

```bash
rl-edos selfplay --config src/rl_edos/configs/selfplay_demo.yaml --out models/selfplay
```

Self-play is a research loop on top of the Phase 3 defender, so a genuinely meaningful (not just
mechanically working) self-play result depends on the Phase 3 MVP gap above being closed first —
this phase was in fact built before that gap was closed (see CLAUDE.md's process note), so treat
any self-play run to date as a mechanical smoke test, not a research finding.

## Build documentation set

The rest of this README indexes the build spec below.

Hand this entire folder to Claude Code in VS Code and drive the build **one phase at a time**. Read the files in order. The build has mandatory STOP points — do not skip ahead.

## What this project is (one line)

A simulated cloud testbed in which an attacker agent and a defender agent learn against each other, so the defender becomes an autoscaling policy hardened against economically-motivated (EDoS) attacks.

## Folder layout

```
rl-edos/                         # this repo's root
  README.md                      # this file
  docs/
    00_START_HERE.md              # workflow, phase order, STOP protocol — read first
    01_PROJECT_SPEC.md            # condensed spec (source of truth)
    02_ARCHITECTURE.md            # modules, boundaries, Gym interface contract
    03_DATA_SCHEMAS.md            # every config/state/record schema, typed, with units
    04_TECH_STACK_AND_SETUP.md    # pinned stack, repo layout, tooling
    05_BILLING_AND_FIDELITY.md    # real provider economics + citations (non-negotiable)
    10_TESTING_STRATEGY.md        # per-phase test requirements + acceptance gates
    11_CLAUDE_CODE_RULES.md       # standing constraints — pin in every session
    PHASE_1_ENVIRONMENT_CORE.md      # scaffold + simulation core        → STOP
    PHASE_2_BASELINES_AND_EVAL.md    # baselines + attacks + eval harness → STOP
    PHASE_3_RL_DEFENDER.md           # PPO defender (MVP result)          → STOP
    PHASE_4_SELFPLAY_AND_RELEASE.md  # self-play (stretch) + polish       → STOP
    ASSUMPTIONS.md                   # modelling assumptions, sim-to-real gap
```

> The four `docs/PHASE_*.md` files are the same consolidated build prompts referenced in `docs/00_START_HERE.md` (there as `06`–`09`), kept as individual per-phase files so you can hand Claude Code exactly one phase at a time.

## Reading / build order

1. `docs/00_START_HERE.md` — how the set works and the STOP protocol.
2. `docs/01`–`05` and `docs/10`–`11` — the shared context Claude Code reads once, up front.
3. `docs/PHASE_1_ENVIRONMENT_CORE.md` — build, then STOP for review.
4. `docs/PHASE_2_BASELINES_AND_EVAL.md` — after Phase 1 gate passes.
5. `docs/PHASE_3_RL_DEFENDER.md` — after Phase 2 gate passes (this is the MVP milestone).
6. `docs/PHASE_4_SELFPLAY_AND_RELEASE.md` — after Phase 3 gate passes. **As of this writing, Phase 3's gate has not passed — see CLAUDE.md's process note — so treat Phase 4 as not yet unlocked in practice even though its code exists.**

## Session bootstrap (paste at the top of each Claude Code session)

> You are building RL-EDoS. Read `docs/11_CLAUDE_CODE_RULES.md`, `docs/01_PROJECT_SPEC.md`, `docs/02_ARCHITECTURE.md`, and `docs/03_DATA_SCHEMAS.md` before writing code. We are on **Phase N**; implement only what `docs/PHASE_N_*.md` specifies, stop at its STOP point, and do not begin the next phase.

## Non-negotiables (detail in `docs/11_CLAUDE_CODE_RULES.md`)

- No `Co-Authored-By: Claude` in commits.
- Ask before adding any dependency outside the fixed stack.
- Everything seeded and deterministic.
- Billing/cooldown/warm-up constants sourced from real provider docs and cited in comments.
- Build strictly in phase order; respect STOP points.
