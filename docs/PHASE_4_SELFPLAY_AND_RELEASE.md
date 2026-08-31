# 09 — PHASE 4 PROMPT: Self-Play (Stretch) + Polish + Release

> Give this only after the Phase 3 STOP gate has passed — i.e. the defender demonstrably beats baselines against scripted attacks. Keep `../11_CLAUDE_CODE_RULES.md` pinned. **Do not begin this phase early.**

## Goal of this phase

Add the co-evolutionary self-play loop (the advanced stage), then harden the whole project for release: error handling, docs, reproducibility, optional dashboard, open-source packaging.

## Part A — Co-evolutionary self-play (stretch)

1. **Learning attacker** (`agents/`): PPO attacker with action space = traffic-shaping parameters bounded by the detection constraint; reward `cost_gain_w * cost_inflicted - evasion_w * detection_penalty` per `AgentConfig`.
2. **Self-play loop** (`training/selfplay`): alternating freeze/train schedule —
   - freeze defender, train attacker to inflate the bill under the detection constraint
   - freeze attacker, train defender to counter while staying cheap
   - repeat until a convergence criterion or budget
   - checkpoint **both** agents each round; log stability metrics
3. **Graceful non-convergence**: if self-play doesn't converge, degrade to a **documented stability/failure-analysis artifact** rather than crashing the pipeline. A documented failure analysis is an acceptable deliverable per the spec.
4. **Wire CLI**: `selfplay --config <path>` with `attack.mode = learned`; then the same `evaluate` harness reports the hardened defender vs baselines plus the self-play stability analysis.
5. **Tests**: self-play loop runs a tiny-budget round-trip (attacker trains, defender trains, both checkpointed); stability logging present; non-convergence path produces the failure-analysis artifact instead of raising.

## Part B — Polish & release

6. **Error handling pass**: invalid configs fail fast with clear messages; NaN/divergence surfaced in every training path; no silent failures anywhere.
7. **Reproducibility pass**: confirm every run captures config hash + git SHA + seed; identical config → identical metrics within tolerance; pinned deps verified from a clean env.
8. **Docs**: complete `README.md` (setup, run `train-defender` → `evaluate`, reproduce the headline result); `docs/ASSUMPTIONS.md` (threat model, billing assumptions, sim-to-real gap); document the self-play design and any failure analysis.
9. **Optional results dashboard** (post-MVP, cuttable): a lightweight **static HTML + Matplotlib exports** or Streamlit view aggregating a run's artifacts (`report.html`). Minimal, chart-first, colour-blind-safe, keyboard-navigable if built. No SPA framework, no browser storage.
10. **Open-source packaging**: finalise `LICENSE`, ensure `pyproject.toml` installs clean, CI-friendly `black`/`ruff`/`pytest` all green.

## Constraints reminder

- Self-play must not introduce uncontrolled randomness — both agents' training and the loop are seeded.
- No new dependencies without asking and justifying against the fixed stack.
- No AI attribution in commits.

## → STOP (final)

Present the final deliverable state:
- self-play result: either a converged hardened defender **or** a documented stability/failure analysis,
- the full `evaluate` comparison of the hardened defender vs baselines,
- README reproduction path verified from a clean environment,
- docs (assumptions, threat model, sim-to-real gap) complete,
- all tooling gates green,
- (if built) the optional dashboard.

Map the result against the **Definition of done** in `../01_PROJECT_SPEC.md` and confirm each item.
