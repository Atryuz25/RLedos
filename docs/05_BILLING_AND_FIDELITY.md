# 05 — Billing & Fidelity

**Fidelity is non-negotiable.** Billing, cooldown, and warm-up constants must be sourced from real provider values and cited in code comments. Do not invent economics. The EDoS claim only holds if the cost model reflects how providers actually bill.

## Reference provider values (cite these in code comments)

Use AWS EC2 On-Demand, Linux, `us-east-1` as the reference economics. Values below are for the config defaults; put the source URL and access date in a comment next to each constant.

- **Per-second billing is real.** <cite index="6-1">Amazon EC2 usage is billed in one-second increments, with a minimum of 60 seconds, and applies to all purchase options.</cite> This directly motivates `billing_granularity: per_second` and a `min_service_charge` reflecting the 60-second minimum.
- **On-demand is pay-as-you-go by the hour or second.** <cite index="6-1">On-Demand Instances offer pay-as-you-go compute capacity by the hour or second, with no upfront payment or long-term commitment, and per-second billing removes the cost of unused compute time from the bill.</cite>
- **Reference instance price.** A `t3.medium` (2 vCPU, 4 GiB) <cite index="8-1">is priced from $0.0416 per hour in the us-east-1 region.</cite> Derive `price_per_instance_second = 0.0416 / 3600 ≈ 0.00001156` currency/instance-second.
- **Billing starts at launch, not at ready.** <cite index="4-1">Pricing is per instance-hour consumed for each instance, from the time an instance is launched until it is terminated or stopped.</cite> This is the crux of the EDoS economics: **you pay for warming instances even before they serve traffic** — so `instance_warmup_s` costs money without adding capacity, which is exactly what an attacker exploits.

Source URLs to cite in comments:
- `https://aws.amazon.com/ec2/pricing/` (per-second billing, 60s minimum)
- `https://aws.amazon.com/ec2/pricing/on-demand/` (billed from launch to termination)
- t3.medium hourly rate — cite the AWS pricing page for us-east-1 at your build date; the aggregator figure above is a cross-check, not the primary source.

> When you build, re-verify the t3.medium us-east-1 on-demand rate against the official AWS pricing page at build time and update the constant + comment if it has moved. Prices drift.

## Billing model requirements

- `BillingModel.accrue(state) -> cost_delta` computes cost for one control interval.
- Cost accrues for **every instance that exists**, including `warming_instances`, from launch — not just serving instances. This is the modelled attack surface.
- Honour `billing_granularity`: under `per_second`, accrue per simulated second; enforce the 60-second minimum via `min_service_charge` semantics.
- All costs in a single currency unit, documented in the docstring.
- Deterministic: identical state + config → identical `cost_delta`.

## Cooldown & warm-up (the autoscaler physics)

- `scale_cooldown_s` — after a scaling action, no further scaling for this many seconds. Models real autoscaler cooldown so the controller can't oscillate instantly.
- `instance_warmup_s` — a newly launched instance spends this long in `warming_instances` (billed, not serving) before moving to `active_instances`. This delay is what makes over-provisioning under a shaped attack economically damaging.

Pick default cooldown/warm-up values consistent with common autoscaling-group defaults and document the choice; note that exact numbers are configurable and the fidelity requirement is that they be realistic and cited, not that one specific number is used.

## Detection model (evasion constraint)

- `DetectionModel.score(traffic_window) -> detection_score` is a **statistical** detector (`rate_threshold` or `variance_threshold`) over a sliding `window`.
- It is a constraint inside the simulation, **not a production IDS**. Its job is to bound how aggressively the attacker can shape traffic before being penalised (`penalty` on the attacker reward).
- **Validate the detector in isolation before wiring it into the loop** (Phase 1 STOP requirement): feed it known-legit and known-attack windows and confirm scores/threshold behave as intended.

## Fidelity checklist (Phase 1 STOP gate)

- [ ] Every billing/cooldown/warm-up constant has a real source cited in a comment.
- [ ] Warming instances are billed from launch.
- [ ] Per-second granularity and 60-second minimum modelled.
- [ ] Detection model validated standalone against labelled windows.
- [ ] All modelling assumptions written up (in README or a `docs/ASSUMPTIONS.md`).
- [ ] Billing accrual is deterministic and unit-tested.
