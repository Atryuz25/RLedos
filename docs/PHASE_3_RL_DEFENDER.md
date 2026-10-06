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

**Not yet tried (as of 2026-09-27):** retuning `latency_w` down (recommendation: `0.01` → roughly `0.0005`, so the baseline latency contribution lands in the same order of magnitude as cost's) combined with a much larger training budget (500k–2M timesteps, per this phase's own performance target). Both changes are recommended together — a longer run under the current unbalanced reward is not expected to change the qualitative outcome, since the imbalance affects every gradient step, not just early ones.

---

## Update (2026-10-06): the retune was tried — non-monotonic, then a reward-formula fix

**Tried:** three real 1,000,000-timestep `train-defender` + `evaluate --seeds 0 1 2 3 4` runs on `configs/default.yaml`, each at a smaller `latency_w` (`cost_w=1.0` fixed, `sim.seed=42` fixed across all three, so these are not random-seed noise):

| `latency_w` | `rl_defender` cost | p95 latency | vs. `target_tracking` (cost `0.003977`) |
|---|---|---|---|
| 0.0005 | 0.007526 | 20.0 ms (floor) | +89% |
| 0.0001 | 0.005010 | 20.0 ms (floor) | **+26% (best found)** |
| 0.00002 | 0.013641 | 20.0 ms (floor) | +243% (worse than the first attempt) |

**Result:** `0.0001` beat `security_blind_rl` outright on both axes (58% cheaper, latency tied) — that baseline comparison is closed. Against `target_tracking`, `rl_defender` won decisively on latency every time (pinned at the `base_latency_ms` floor throughout, i.e. the queue never built at all) but never beat it on cost, and cutting `latency_w` further **stopped helping and then reversed** at `0.00002`. Since the seed was fixed, this is a real non-monotonic response to the weight, not noise — simple linear down-scaling of `latency_w` was not a reliable lever past a point.

**Revised cause:** a purely linear `-(cost_w * cost_delta + latency_w * latency_ms)` combination means *any* latency above zero costs something. Because the latency formula is steep (`base_latency_ms + queue_len/capacity_rps * 1000` — a small backlog spikes latency by hundreds to thousands of ms, exactly what `security_blind_rl` hit at `6686.7ms` in one run), the *marginal* risk of ever letting the queue build outweighs the cost saved, regardless of how small `latency_w`'s baseline contribution is. This is a structural property of the reward shape, not just a scale problem — which is why shrinking the weight further eventually stopped working and became unstable instead.

**Fix applied:** `agents/reward.py::defender_reward` now takes an optional `latency_threshold_ms` (default `0.0`, which exactly reproduces the original always-penalize formula — fully backward compatible) and only penalizes `max(0, latency_ms - latency_threshold_ms)`. A new `AgentConfig.latency_threshold_ms` field (`docs/03_DATA_SCHEMAS.md`) wires it through `CloudEnv.step()`. This gives the defender an SLA-style band (e.g. latency up to `2x base_latency_ms`) in which it pays nothing for latency at all, directly removing the "always keep the queue perfectly empty" pressure instead of hoping a smaller weight threads the needle. Tested in `tests/test_reward.py` (pure-function threshold math, and a default-matches-original-formula regression) and `tests/test_env_dynamics.py::test_latency_threshold_raises_reward_under_overload` (confirms `CloudEnv` actually reads the new field, per Rule 7).

**Not yet tried:** training with a real `latency_threshold_ms` value (e.g. `40.0`, twice `default.yaml`'s `base_latency_ms`) combined with `latency_w` restored to something less extreme than the values above (e.g. back to `0.01`, since the threshold now absorbs ordinary operation for free and only the genuine-degradation case needs a strong penalty) and the same `1,000,000`-timestep budget.

**Phase 4 status:** remains **implemented, not gated** (see `CLAUDE.md`'s process note) until this gate passes for real and self-play is re-run against a defender that has actually closed this gap.
