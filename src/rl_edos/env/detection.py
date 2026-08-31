"""DetectionModel: statistical evasion constraint, not a production IDS.

Validatable in isolation: feed labelled legit/attack windows and check the
score behaves sensibly relative to `threshold` (docs/05, Phase 1 STOP gate).
"""

from __future__ import annotations

import numpy as np

from rl_edos.config import DetectionConfig


class DetectionModel:
    """Scores a sliding window of arrival rates for anomalous shape."""

    def __init__(self, config: DetectionConfig) -> None:
        self.config = config

    def score(self, traffic_window: np.ndarray) -> float:
        """Dimensionless detection score for `traffic_window` (requests/second samples).

        A score >= 1.0 means the window trips the configured `threshold`.
        Empty windows score 0 (nothing observed yet, e.g. right after reset).
        """
        window = np.asarray(traffic_window, dtype=np.float64)
        if window.size == 0:
            return 0.0
        if self.config.type == "rate_threshold":
            stat = float(np.mean(window))
        else:  # variance_threshold
            stat = float(np.std(window))
        return stat / self.config.threshold
