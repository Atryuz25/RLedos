# RL-EDoS — Documentation Set (START HERE)

This folder is the complete build specification for **RL-EDoS**: an adversarial reinforcement-learning testbed in which an attacker agent and a defender agent learn against each other, producing an autoscaling policy hardened against Economic Denial of Sustainability (EDoS) attacks.

Hand this whole folder to Claude Code in VS Code and drive the build phase by phase.

## What each file is

| File | Purpose | When to use |
|------|---------|-------------|
| `00_START_HERE.md` | This file. How to use the set, phase order, STOP protocol. | Read first. |
| `01_PROJECT_SPEC.md` | Full product spec (the source of truth). Problem, goals, scope guards, users. | Reference throughout. |
| `02_ARCHITECTURE.md` | System architecture, module boundaries, data flow, the Gym interface contract. | Before writing any code. |
| `03_DATA_SCHEMAS.md` | Every config/state/record schema with types and units. | Whenever touching config, env state, or persisted records. |
| `04_TECH_STACK_AND_SETUP.md` | Pinned stack, repo layout, environment setup, tooling (black/ruff/pytest). | Phase 1 scaffolding. |
| `05_BILLING_AND_FIDELITY.md` | The economics model — real provider pricing, cooldown, warm-up, with citations. Fidelity is non-negotiable. | Phase 1 billing model. |
| `06_PHASE_1_PROMPT.md` | Consolidated build prompt: scaffold + environment core. Ends at a STOP. | Give to Claude Code for Phase 1. |
| `07_PHASE_2_PROMPT.md` | Consolidated build prompt: baselines + scripted attacks + eval harness. STOP. | Phase 2. |
| `08_PHASE_3_PROMPT.md` | Consolidated build prompt: the RL defender (MVP success milestone). STOP. | Phase 3. |
| `09_PHASE_4_PROMPT.md` | Consolidated build prompt: self-play (stretch) + polish + release. STOP. | Phase 4. |
| `10_TESTING_STRATEGY.md` | What must be tested at each phase and the acceptance gates. | Every phase. |
| `11_CLAUDE_CODE_RULES.md` | Standing constraints for every Claude Code session (determinism, no AI attribution, fidelity, phase discipline). | Pin in every session. |

## Phase order (strict — do not skip)

```
Phase 1  Scaffold + environment core        → STOP (verify sim fidelity)
Phase 2  Baselines + scripted attacks + eval → STOP (metrics trustworthy)
Phase 3  RL defender (MVP result)            → STOP (success milestone)
Phase 4  Self-play (stretch) + polish + release → STOP (final)
```

Each phase has its own consolidated prompt file, provided **both** as the flat `06`–`09` files and as individual per-phase files in the `phases/` folder (identical content):

| Phase | Per-phase file |
|-------|----------------|
| 1 | `phases/PHASE_1_ENVIRONMENT_CORE.md` |
| 2 | `phases/PHASE_2_BASELINES_AND_EVAL.md` |
| 3 | `phases/PHASE_3_RL_DEFENDER.md` |
| 4 | `phases/PHASE_4_SELFPLAY_AND_RELEASE.md` |

Give Claude Code **one phase file at a time**, plus `11_CLAUDE_CODE_RULES.md` pinned in context.

## The STOP protocol

At every `→ STOP`, Claude Code must **pause and hand back for review** rather than running ahead into the next phase. At each STOP you (Karthik) verify the acceptance gate in `10_TESTING_STRATEGY.md` before unlocking the next phase prompt. This is deliberate: a wrong sim or untrustworthy baseline invalidates everything trained on top of it.

## Session bootstrap (paste at the top of every Claude Code session)

> You are building RL-EDoS. Read `11_CLAUDE_CODE_RULES.md`, `01_PROJECT_SPEC.md`, `02_ARCHITECTURE.md`, and `03_DATA_SCHEMAS.md` before writing code. We are on **Phase N**; implement only what `0X_PHASE_N_PROMPT.md` specifies, stop at its STOP point, and do not begin the next phase.
