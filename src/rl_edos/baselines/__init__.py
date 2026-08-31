"""Reference controllers exposing the same action contract as the RL defender."""

from rl_edos.baselines.randomisation import RandomisationController
from rl_edos.baselines.security_blind_rl import SecurityBlindController, train_security_blind
from rl_edos.baselines.target_tracking import TargetTrackingController

__all__ = [
    "RandomisationController",
    "SecurityBlindController",
    "TargetTrackingController",
    "train_security_blind",
]
