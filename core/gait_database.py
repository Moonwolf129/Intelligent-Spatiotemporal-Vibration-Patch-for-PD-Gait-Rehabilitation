import numpy as np
from dataclasses import dataclass, field
from typing import List


@dataclass
class GaitCycleFeature:
    """Resampled gait cycle as a feature vector."""
    feature: np.ndarray
    timestamp: float  # e.g. end time of the cycle


@dataclass
class GaitDatabase:
    """
    Personalized gait database with dynamic DBSCAN-inspired updating and
    time-weighted averaging.
    """
    cycles: List[GaitCycleFeature] = field(default_factory=list)
    rejected_count: int = 0

    def _compute_mean_vector(self):
        if not self.cycles:
            return None
        feats = np.stack([c.feature for c in self.cycles], axis=0)
        return np.mean(feats, axis=0)

    def _dynamic_radius(self, mean_vec):
        if not self.cycles:
            return np.inf
        feats = np.stack([c.feature for c in self.cycles], axis=0)
        dists = np.linalg.norm(feats - mean_vec, axis=1)
        return np.mean(dists)

    def _min_neighbors(self):
        return max(2, len(self.cycles))

    def _is_core_point(self, new_feat, radius, min_neighbors):
        if not self.cycles:
            return True
        feats = np.stack([c.feature for c in self.cycles], axis=0)
        dists = np.linalg.norm(feats - new_feat, axis=1)
        return np.sum(dists <= radius) >= min_neighbors

    def add_cycle(self, theta_cycle: np.ndarray, time_cycle: np.ndarray,
                  resample_points: int = 100):
        """
        Add a new gait cycle to the database if it passes dynamic DBSCAN criteria.

        Parameters
        ----------
        theta_cycle : (M,) array
            Thigh angle samples for one gait cycle.
        time_cycle : (M,) array
            Time stamps for that gait cycle.
        resample_points : int
            Number of points for uniform resampling to a fixed length feature.
        """
        if len(theta_cycle) < 5:
            return False

        # Normalize to [0, 1] in phase by linear interpolation
        phase = (time_cycle - time_cycle[0]) / (time_cycle[-1] - time_cycle[0] + 1e-8)
        phase_uniform = np.linspace(0, 1, resample_points)
        feat = np.interp(phase_uniform, phase, theta_cycle)

        mean_vec = self._compute_mean_vector()
        if mean_vec is None:
            # First sample directly accepted
            self.cycles.append(GaitCycleFeature(feature=feat,
                                                timestamp=float(time_cycle[-1])))
            return True

        radius = self._dynamic_radius(mean_vec)
        min_neighbors = self._min_neighbors()
        if self._is_core_point(feat, radius, min_neighbors):
            self.cycles.append(GaitCycleFeature(feature=feat,
                                                timestamp=float(time_cycle[-1])))
            return True
        else:
            self.rejected_count += 1
            return False

    @property
    def reliability_index(self):
        """
        Sample reliability index:
        R = d_min / (1 + N_out / N_total)
        """
        if len(self.cycles) < 2:
            return 0.0
        feats = np.stack([c.feature for c in self.cycles], axis=0)
        N = feats.shape[0]
        d_min = np.inf
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(feats[i] - feats[j])
                d_min = min(d_min, d)
        N_out = self.rejected_count
        return float(d_min / (1.0 + N_out / max(1, N)))

    def time_weighted_template(self, alpha: float = 0.03):
        """
        Compute time-weighted average gait curve.

        w_i = exp(alpha * (i - N)), i = 1..N
        """
        if not self.cycles:
            return None
        feats = np.stack([c.feature for c in self.cycles], axis=0)
        N = feats.shape[0]
        idx = np.arange(1, N + 1)
        w = np.exp(alpha * (idx - N))
        w = w / np.sum(w)
        template = np.sum(feats * w[:, None], axis=0)
        return template
