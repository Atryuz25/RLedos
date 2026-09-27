# 08 — PHASE 3 PROMPT: RL Defender (the MVP Result)

> Give this only after the Phase 2 STOP gate has passed. Keep `../11_CLAUDE_CODE_RULES.md` pinned. This phase is the **success milestone** — lock it before any self-play.

## Goal of this phase

Train a PPO defender whose reward jointly balances cost and latency, and show it beats the baselines on **both** success axes **simultaneously** under scripted attacks. This is the project's headline claim.

## Scope (implement exactly this)

1. **Defender agent** (`agents/`):
   - observation space per `../03_DATA_SCHEMAS.md` (arrival rate, queue, active/warming instances, latency, accrued-cost rate, detection score)
   - action space = the same scaling action contract the baselines emit
   - joint reward `-(cost_w * cost_delta + latency_w * latency_penalty)`, weights from `AgentConfig.reward_weights`
2. **Trainer** (`training/train_defender`): `Trainer.train(agent_cfg, env) -> policy_ckpt` wrapping **SB3 PPO** (no hand-rolled RL). Trains against the Phase 2 scripted attacks. Saves an SB3 `.zip` checkpoint + TensorBoard logs.
   - **Observation normalization required**: `CloudEnv`'s observation space is unbounded above on every dimension (`Box(low=0, high=inf)` — arrival rate, queue length, accrued cost, etc. have no real ceiling). Pre-Phase-3 audit confirmed via `gymnasium.utils.env_checker.check_env` that this trips SB3's own best-practice warning and risks destabilising PPO's value-function learning. Wrap training with `stable_baselines3.common.vec_env.VecNormalize` (`norm_obs=True`, `norm_reward=False` — don't normalize reward, it would change the joint cost+latency semantics the MVP claim is measured on). `VecNormalize`'s running stats are **not** part of the SB3 `.zip` and must be saved/loaded alongside `policy_ckpt` (`VecNormalize.save`/`.load`) so `evaluate --policy <ckpt>` normalizes observations identically at inference time — get this wrong and predictions will be silently miscalibrated (trained-on-normalized-obs, evaluated-on-raw-obs). This was deliberately not retrofitted onto the Phase 2 `security_blind_rl` baseline, since that baseline currently has no companion-file checkpoint format to extend cleanly — design `policy_ckpt`'s shape here with this requirement in mind from the start.
3. **Divergence/NaN guards**: detect training divergence or NaN and **report loudly**; never persist a broken policy as valid.
4. **Wire CLI**: `train-defender --config <path>` runs training and emits checkpoint + logs; `evaluate --policy <ckpt> --baselines <set> --config <path>` runs the Phase 2 harness including the trained defender.
5. **Headline experiment**: on the provided default config, run `train-defender` then `evaluate`, producing `comparison.csv` + learning curves + attack-pattern plots showing the defender vs `target_tracking` and `security_blind_rl` (and `randomisation` if built) on cost-under-attack and legit-traffic quality.
6. **Tests**: reward function unit-tested (correct sign/scaling for cost and latency); training smoke test (a tiny-budget run completes, produces a loadable checkpoint, no NaN); evaluation integrates the trained policy through the same harness.

## MVP acceptance (the gate)

- The RL defender beats **target-tracking** and **security-blind RL** on **both** `mean_cost_under_attack` and legit-traffic quality (`p95_latency_ms` / `legit_drop_rate`) — shown in `comparison.csv` + plots.
- Result is reproducible: same config + seed → same headline numbers within tolerance.
- No broken/NaN policy is ever saved as valid.

## Performance target

A single MVP defender training run completes on CPU within a practical wall-clock budget (overnight or faster). If it can't, tune `total_timesteps` / env step throughput — do **not** switch algorithms or invent new dependencies.

## Tooling gates before STOP

- `black`/`ruff`/`pytest` green
- Headline `comparison.csv` + plots committed as the reproducible result (artifacts themselves gitignored; the config + README instructions to regenerate them are tracked)

## → STOP FOR REVIEW

Stop. Do **not** start Phase 4 (self-play). Present:
- the headline comparison proving the defender wins on both axes,
- the learning curves and attack-pattern plots,
- reproduction instructions (config + commands),
- confirmation of the divergence/NaN guard behaviour.

Karthik verifies the MVP success milestone (per `../10_TESTING_STRATEGY.md` Phase 3 gate). Only after this passes is self-play unlocked.

---

## Current status (as of 2026-09-27): gate NOT met — documented failure analysis

**Tried:** three real end-to-end runs of `train-defender` + `evaluate` on `configs/default.yaml` against the scripted `oscillation` attack: 10k timesteps, 150k timesteps, and a fresh 10k-timestep smoke run (identical config) re-run during a plumbing-fix pass covering Phases 0–5 of an audit (see `CLAUDE.md`'s process note for that pass's context).

**Result:** all three show the same qualitative pattern. `target_tracking` beats the RL defender on `mean_cost_under_attack` every time; the smoke run's real numbers: `target_tracking` cost `0.003988`, `rl_defender` cost `0.008784`, `security_blind_rl` cost `0.010288` (5-decimal currency units per interval, mean over the episode). `rl_defender`'s `overprovision_ratio` (4.53) is more than double `target_tracking`'s (2.03) — both RL-trained controllers (the joint-reward defender *and* the security-blind baseline) independently converge toward over-provisioning rather than toward `target_tracking`'s leaner policy, which suggests a systematic bias rather than training noise.

**Likely cause (identified, not yet re-tested):** `defender_reward = -(cost_w * cost_delta + latency_w * latency_ms)` combines two terms on very different numeric scales given `default.yaml`'s real values. At baseline (near-empty queue): `cost_w * cost_delta ≈ 1.0 * 0.004 ≈ 0.004`, while `latency_w * latency_ms ≈ 0.01 * 20.0 ≈ 0.2` — the latency term dominates the reward by roughly **50x even before any queueing penalty applies**. The policy gradient has almost no incentive to trade latency for cost savings, because essentially any latency increase (however small) outweighs the cost saved by running fewer instances. This is a config-level reward-weight scaling problem, not a reward-formula or plumbing bug — `agent.reward_weights` is exactly the tunable surface `docs/03_DATA_SCHEMAS.md` and `docs/PHASE_3_RL_DEFENDER.md` intend for this.

**Not yet tried:** retuning `latency_w` down (recommendation: `0.01` → roughly `0.0005`, so the baseline latency contribution lands in the same order of magnitude as cost's) combined with a much larger training budget (500k–2M timesteps, per this phase's own performance target). Both changes are recommended together — a longer run under the current unbalanced reward is not expected to change the qualitative outcome, since the imbalance affects every gradient step, not just early ones.

**Phase 4 status:** remains **implemented, not gated** (see `CLAUDE.md`'s process note) until this gate passes for real and self-play is re-run against a defender that has actually closed this gap.
