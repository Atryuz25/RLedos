"""Scripted attack patterns: seeded/reproducible (deterministic), respect intensity scaling."""

from __future__ import annotations

from rl_edos.config import AttackSpec
from rl_edos.env.attacks import ScriptedAttack


def test_steady_attack_is_constant(make_config):
    cfg = make_config()
    spec = AttackSpec(mode="scripted", pattern="steady", intensity=0.5, evasion_budget=0.0)
    attack = ScriptedAttack(spec, cfg.sim, base_rate=cfg.traffic.base_rate)
    rate = attack.rate(0.0)
    assert rate == cfg.traffic.base_rate * 0.5
    assert attack.rate(500.0) == rate


def test_oscillation_attack_varies_and_stays_bounded(make_config):
    cfg = make_config()
    spec = AttackSpec(mode="scripted", pattern="oscillation", intensity=1.0, evasion_budget=0.0)
    attack = ScriptedAttack(spec, cfg.sim, base_rate=cfg.traffic.base_rate)
    peak = cfg.traffic.base_rate
    samples = [attack.rate(t) for t in range(0, 2000, 50)]
    assert min(samples) >= 0.0
    assert max(samples) <= peak + 1e-9
    assert len(set(samples)) > 1  # actually varies


def test_burst_attack_is_zero_outside_duty_cycle(make_config):
    cfg = make_config()
    spec = AttackSpec(mode="scripted", pattern="burst", intensity=1.0, evasion_budget=0.0)
    attack = ScriptedAttack(spec, cfg.sim, base_rate=cfg.traffic.base_rate)
    # just past the start of a cycle: inside the 20% duty-cycle window -> "on"
    on_sample = attack.rate(1.0)
    # deep into the cycle: past the duty cycle -> "off"
    off_sample = attack.rate(attack._period_s * 0.9)
    assert on_sample > 0
    assert off_sample == 0.0


def test_attack_is_deterministic(make_config):
    cfg = make_config()
    spec = AttackSpec(mode="scripted", pattern="oscillation", intensity=0.8, evasion_budget=0.0)
    a1 = ScriptedAttack(spec, cfg.sim, base_rate=cfg.traffic.base_rate)
    a2 = ScriptedAttack(spec, cfg.sim, base_rate=cfg.traffic.base_rate)
    for t in [0.0, 10.0, 123.4]:
        assert a1.rate(t) == a2.rate(t)
