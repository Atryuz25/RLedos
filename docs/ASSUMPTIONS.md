# Modelling Assumptions

Simplifications made in the simulation core, disclosed per the Phase 1 fidelity checklist
(`05_BILLING_AND_FIDELITY.md`). Billing/cooldown/warm-up **constants** are cited real provider
values; the items below are the modelling choices layered on top of them.

## Billing: 60-second minimum applied per control-interval, not per-instance-lifetime

AWS bills with a 60-second minimum duration per instance. Modelling that exactly requires
tracking each instance's own launch time and settling the floor at termination. Phase 1 instead
applies `min_service_charge` (currency per instance) as a floor on the *per-interval* charge:
`max(price_per_instance_second * elapsed_s, min_service_charge)` per existing instance. This is
simpler, deterministic, and conservative (never under-charges relative to the real floor) as long
as `control_interval_s` isn't far below 60s. `src/rl_edos/env/billing.py` carries a `ponytail:`
comment marking this; upgrade to per-instance lifecycle billing if configs start using very short
control intervals where the approximation would matter.

## Instance cancellation order: newest warming instances cancelled first

When scaling down, warming instances are cancelled before active ones are terminated (they aren't
serving traffic yet, so cancelling them is strictly cheaper). Among warming instances, the
most-recently-launched are cancelled first — they've accrued the least billed time so far.

## Queue/latency model: single-interval backlog, not full queueing theory

Each control interval: `incoming = arrival_rate * control_interval_s + carried_queue`,
`served = min(incoming, active_instances * capacity_rps_per_instance * control_interval_s)`,
backlog carries to the next interval. Latency is `base_latency_ms + (queue_len / capacity_rps) *
1000`. This is a deterministic, discrete-interval approximation (per `04_TECH_STACK_AND_SETUP.md`:
"discrete-event scheduling via a plain step-loop"), not an M/M/c queueing model. `capacity_rps_per_instance`
and `base_latency_ms` are engineering knobs (not billing economics) added to `SimConfig` beyond the
`03_DATA_SCHEMAS.md` table, needed to make the queue/latency dynamics config-driven rather than
hard-coded.

## Detection score: normalised ratio, not a calibrated classifier

`DetectionModel.score` returns `stat(window) / threshold` (mean for `rate_threshold`, std for
`variance_threshold`). A score >= 1.0 means the window trips the threshold. This is a simulation
constraint bounding attacker evasion, not a production intrusion-detection system.

## Action space: continuous target instance count

The defender (and baselines) emit a target active-instance count in `[min_instances,
max_instances]`, rounded to the nearest integer and clipped to bounds. This matches the same
action contract baselines will use in Phase 2.

## Queue capacity and drop attribution (Phase 2)

Phase 1's queue was an unbounded backlog. Phase 2 adds `SimConfig.max_queue_len` (default
infinite, so Phase 1 configs/behaviour are unaffected) as a backlog ceiling: overflow beyond it is
dropped rather than queued, which is what makes `legit_drop_rate` measurable. Because the queue is
a single shared backlog (not per-origin), overflow in a given interval is apportioned between
legit and attack traffic by each stream's share of *that interval's new arrivals* — carried-over
backlog composition isn't tracked. This is a fair-sharing approximation, not literal per-request
attribution.

## Sim-to-real gap

This testbed is a simplified discrete-interval simulation of a single-region, single-instance-type
autoscaling group. It does not model multi-AZ placement, spot pricing, network topology, or
request-level heterogeneity. Results characterise the adversarial RL formulation and relative
controller performance under the simulated economics — they are not a deployment-ready policy.
