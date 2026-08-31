"""BillingModel: per-second cloud instance billing.

Fidelity sources (docs/05_BILLING_AND_FIDELITY.md):
- Per-second billing, 60-second minimum: https://aws.amazon.com/ec2/pricing/
- Billed from launch to termination (warming instances included):
  https://aws.amazon.com/ec2/pricing/on-demand/
- Reference rate: AWS EC2 t3.medium on-demand, us-east-1, $0.0416/hour
  (re-verify against the live AWS pricing page at build time; prices drift).
"""

from __future__ import annotations

import math

from rl_edos.config import BillingConfig


class BillingModel:
    """Accrues cost for one control interval given the instances that existed during it."""

    def __init__(self, config: BillingConfig) -> None:
        self.config = config

    def accrue(self, active_instances: int, warming_instances: int, elapsed_s: float) -> float:
        """Cost, in currency, for `elapsed_s` seconds with the given instance counts.

        Warming instances are billed identically to active instances — billing
        starts at launch, not at ready (the core EDoS attack surface).

        # ponytail: the 60s minimum is modelled per control-interval via
        # min_service_charge rather than per-instance-lifetime tracking (which
        # would need launch-time bookkeeping per instance). Upgrade to
        # per-instance lifecycle billing if configs start using
        # control_interval_s << 60s in ways where this approximation matters.
        """
        n = active_instances + warming_instances
        if n <= 0:
            return 0.0
        if self.config.billing_granularity == "per_second":
            per_instance = self.config.price_per_instance_second * elapsed_s
        else:  # per_hour
            per_instance = (
                self.config.price_per_instance_second * 3600 * math.ceil(elapsed_s / 3600)
            )
        per_instance = max(per_instance, self.config.min_service_charge)
        return n * per_instance
