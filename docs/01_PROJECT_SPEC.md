# 01 — Project Spec (Source of Truth)

## One-line pitch

A simulated cloud testbed in which an attacker agent and a defender agent learn against each other, so the defender becomes an autoscaling policy hardened against economically-motivated (EDoS) attacks. Built for cloud-security researchers and platform/SRE engineers who design scaling controllers.

## Problem

Cloud autoscaling trades money for availability: it provisions more compute as observed load rises. An adversary weaponises this by sending traffic that looks legitimate at the request level but is collectively shaped to keep the autoscaler over-provisioning. The service never crashes, so uptime and error-rate monitoring show nothing wrong, yet the cloud bill inflates. This is **Economic Denial of Sustainability (EDoS)**. Existing defences are static (fixed thresholds, rule filters, randomisation) and present a fixed decision boundary an adaptive attacker learns to evade.

## Primary goal (defines success)

Produce a scaling controller (defender policy) that, under attack, keeps **both** cloud cost low **and** legitimate-traffic service quality high, and demonstrably outperforms three baselines — target-tracking autoscaling, security-blind RL, and a randomisation defence — on these two axes **simultaneously**.

## Non-goals (scope guards — enforce these)

- **Not** a production-deployable autoscaler. Output is a validated policy + testbed, not a drop-in controller for a live cluster.
- **Not** a sim-to-real transfer project. Real-cluster deployment, domain randomisation, and shadow-mode evaluation are future work, not delivered.
- **Not** a web application, dashboard product, or SaaS. No end-user UI, no auth, no multi-tenancy.
- **Not** a novel RL algorithm. Standard PPO is used; the contribution is the environment, the adversarial formulation, and the hardened policy — not a new optimiser.
- **Not** a real-traffic intrusion-detection system. Detection is a constraint inside the simulation, not a production IDS.

## Target users

- **Primary — cloud-security / autoscaling researchers.** High technical level. Run experiments from a terminal, read result plots and metrics tables. Consume the testbed as a reproducible benchmark and the defender as a comparison method.
- **Secondary — platform / SRE engineers.** High technical level. Evaluating whether a security-aware controller is worth pursuing for their own stack; read the evaluation output and documented threat/billing assumptions.
- **Tertiary — project evaluators (guide, review panel).** Mixed level. Read the consolidated report, run the code from a README, view result plots.

No non-technical persona. All interaction is via CLI, config files, and generated artifacts.

## Core features (MVP) — ranked, must-haves only

1. **Simulated cloud environment (Gym interface)** — instantiate a discrete-event cloud (instance pool, request queue, latency, autoscaler) as a standard `gymnasium.Env` and step it with a scaling action.
2. **Economically-faithful billing model** — configure per-second instance-hour pricing, scaling cooldown, warm-up delay; read an accrued-cost signal each step reflecting real provider economics.
3. **Traffic + detection layer** — generate legitimate (diurnal + Poisson) traffic and parameterised attack traffic, passed through an explicit statistical detection model constraining the attacker's evasion.
4. **RL defender training against scripted attacks** — train a PPO defender (Stable-Baselines3) whose reward jointly balances cost and latency; obtain a saved policy.
5. **Baseline suite + evaluation harness** — run target-tracking autoscaling and security-blind RL (optionally randomisation) under identical attack conditions; get a comparison table + learning curves + attack-pattern plots on the two success axes.

Features 1–5 constitute the MVP and satisfy the success definition without requiring self-play.

## Nice-to-have (post-MVP)

- **Co-evolutionary self-play loop** — a learning *attacker* co-trained against the defender in an alternating freeze/train schedule (the "advanced stage"). Defer until MVP defender reliably beats baselines against scripted attacks.
- **SIGMETRICS randomisation baseline** — faithful reimplementation for a third, stronger comparison. Cuttable; two baselines suffice for the core claim.
- **Real-trace grounding** — replay a public request trace to validate the synthetic traffic generator.
- **Results dashboard** — lightweight static HTML/Streamlit view of runs and comparisons (research convenience, not a product).
- **Domain-randomisation harness** — first concrete step toward sim-to-real, for future work.
- **Experiment config sweeps** — Hydra/Optuna-style hyperparameter and scenario sweeps.

## Key user flows

**Flow A — Train & evaluate the defender (core research loop):**
User selects a config (SimConfig + BillingConfig + TrafficSpec + AttackSpec + AgentConfig) → runs `train-defender` → PPO trains against scripted attacks in the Gym env → system saves a policy checkpoint + TensorBoard logs → user runs `evaluate` pointing at the checkpoint and the baseline set → harness runs all controllers under identical attack seeds → system emits a comparison table (cost vs latency), learning curves, and attack-pattern plots → user reads the result and decides whether the defender wins on both axes.

**Flow B — Co-evolutionary self-play (advanced stage):**
User sets `attack.mode = learned` and runs `selfplay` → loop freezes defender, trains attacker to inflate bill under the detection constraint → freezes attacker, trains defender to counter while staying cheap → repeats until convergence criterion or budget → system checkpoints both agents each round and logs stability metrics → on completion (or non-convergence), user runs the same `evaluate` harness → system reports the hardened defender vs baselines, plus a self-play stability/failure analysis.

## Non-functional requirements

- **Performance:** A single MVP defender training run completes on CPU within a practical wall-clock budget (target: overnight or faster). Evaluation of all controllers over a fixed attack suite completes in minutes. Env stepping must not be the training bottleneck.
- **Reproducibility (research-critical):** Every run fully seeded; config + git SHA + seed captured in `RunRecord`; identical config → identical metrics within tolerance. Pinned dependency versions.
- **Correctness / fidelity:** Billing, cooldown, warm-up values traceable to real provider documentation; all assumptions documented; detection model validated in isolation before wiring into the loop.
- **Security (of the tool):** No secrets, no network calls, no external data ingestion at runtime; configs validated and schema-checked; safe artifact file paths.
- **Error handling:** Invalid configs fail fast with clear messages; training divergence/NaN detected and reported (never silently saved); non-convergent self-play degrades gracefully to a reported failure-analysis artifact rather than crashing.
- **Responsive:** N/A (no UI). Optional dashboard need only render on a desktop browser.

## Definition of done

- Runs locally from a README: create env, install pinned deps, run `train-defender` then `evaluate` on the default config, reproduce the headline comparison.
- MVP acceptance: RL defender beats target-tracking and security-blind RL on **both** cost-under-attack and legitimate-traffic quality, shown in `comparison.csv` + plots.
- Tests pass for env dynamics, billing accrual, detection scoring, and baseline controllers.
- Documented threat model, billing assumptions, and sim-to-real gap.
- (Stretch) Self-play run with either a converged hardened defender **or** a documented stability/failure analysis.
