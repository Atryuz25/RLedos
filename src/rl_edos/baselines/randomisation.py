"""randomisation: SIGMETRICS-style randomisation defence.

Optional / cuttable per docs/PHASE_2_BASELINES_AND_EVAL.md ("two baselines
suffice for the core claim... implement if time permits, else leave a clean
interface stub"). Left as a stub: two baselines (target_tracking,
security_blind_rl) carry the core comparison.
"""

from __future__ import annotations

import numpy as np

from rl_edos.env.state import EnvState


class RandomisationController:
    def __call__(self, state: EnvState) -> np.ndarray:
        raise NotImplementedError(
            "randomisation baseline is not implemented (cuttable per Phase 2 spec)"
        )
