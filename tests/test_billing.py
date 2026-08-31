"""BillingModel: accrual math, warming-instance billing, 60s minimum, determinism."""

from __future__ import annotations

import pytest
from rl_edos.config import BillingConfig
from rl_edos.env.billing import BillingModel


def test_per_second_accrual_matches_hand_computation():
    cfg = BillingConfig(
        price_per_instance_second=0.00001, billing_granularity="per_second", min_service_charge=0.0
    )
    model = BillingModel(cfg)
    # 3 instances, 30 seconds: 3 * 0.00001 * 30 = 0.0009
    assert model.accrue(active_instances=3, warming_instances=0, elapsed_s=30.0) == pytest.approx(
        0.0009
    )


def test_warming_instances_billed_same_as_active():
    cfg = BillingConfig(
        price_per_instance_second=0.00001, billing_granularity="per_second", min_service_charge=0.0
    )
    model = BillingModel(cfg)
    cost_active = model.accrue(active_instances=2, warming_instances=0, elapsed_s=10.0)
    cost_warming = model.accrue(active_instances=0, warming_instances=2, elapsed_s=10.0)
    assert cost_active == cost_warming


def test_min_service_charge_floor_applies():
    cfg = BillingConfig(
        price_per_instance_second=0.00001,
        billing_granularity="per_second",
        min_service_charge=0.0005,
    )
    model = BillingModel(cfg)
    # raw per-instance = 0.00001 * 1 = 0.00001, below the 0.0005 floor
    cost = model.accrue(active_instances=2, warming_instances=0, elapsed_s=1.0)
    assert cost == pytest.approx(2 * 0.0005)


def test_no_instances_costs_nothing():
    cfg = BillingConfig(
        price_per_instance_second=0.00001,
        billing_granularity="per_second",
        min_service_charge=0.001,
    )
    model = BillingModel(cfg)
    assert model.accrue(active_instances=0, warming_instances=0, elapsed_s=100.0) == 0.0


def test_per_hour_granularity_rounds_up():
    cfg = BillingConfig(
        price_per_instance_second=0.00001, billing_granularity="per_hour", min_service_charge=0.0
    )
    model = BillingModel(cfg)
    # 1 instance, 1 second elapsed but per_hour bills a full ceil(1/3600)=1 hour block
    expected = 1 * (0.00001 * 3600)
    assert model.accrue(active_instances=1, warming_instances=0, elapsed_s=1.0) == pytest.approx(
        expected
    )


def test_accrual_is_deterministic():
    cfg = BillingConfig(
        price_per_instance_second=0.00001, billing_granularity="per_second", min_service_charge=0.0
    )
    model = BillingModel(cfg)
    a = model.accrue(active_instances=4, warming_instances=1, elapsed_s=15.0)
    b = model.accrue(active_instances=4, warming_instances=1, elapsed_s=15.0)
    assert a == b
