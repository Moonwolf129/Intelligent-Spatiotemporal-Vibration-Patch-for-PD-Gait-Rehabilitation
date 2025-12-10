import numpy as np
from dataclasses import dataclass
from .gait_database import GaitDatabase


@dataclass
class TemplateFusionParams:
    w_patient_start: float = 0.9
    w_patient_min: float = 0.5
    reliability_mid: float = 0.5  # reliability where weights are ~0.7/0.3


class TemplateFusion:
    """
    Fuse patient-specific template with normative template:

        T_fused = w_p * T_patient + w_n * T_norm

    The patient weight w_p is gradually reduced as reliability increases.
    """

    def __init__(self, norm_template: np.ndarray,
                 params: TemplateFusionParams = None):
        self.norm_template = norm_template
        self.params = params or TemplateFusionParams()

    def _compute_weights(self, reliability: float):
        """
        Map reliability (0..1 or more) to fusion weights.
        Simple heuristic: exponential decay of patient weight.
        """
        r = max(0.0, reliability)
        # Map reliability to a decay factor in [0, 1]
        k = np.clip(r / (self.params.reliability_mid + 1e-8), 0.0, 2.0)
        w_p = self.params.w_patient_min + \
              (self.params.w_patient_start - self.params.w_patient_min) * np.exp(-k)
        w_p = float(np.clip(w_p, self.params.w_patient_min,
                            self.params.w_patient_start))
        w_n = 1.0 - w_p
        return w_p, w_n

    def get_fused_template(self, db: GaitDatabase):
        """
        Compute fused template curve.

        Parameters
        ----------
        db : GaitDatabase

        Returns
        -------
        fused : (L,) array or None
        weights : (w_patient, w_norm)
        """
        patient_template = db.time_weighted_template()
        if patient_template is None:
            return self.norm_template, (0.0, 1.0)

        # Ensure same length via interpolation
        L = len(self.norm_template)
        x_norm = np.linspace(0, 1, L)
        x_pat = np.linspace(0, 1, len(patient_template))
        patient_resampled = np.interp(x_norm, x_pat, patient_template)

        reliability = db.reliability_index
        w_p, w_n = self._compute_weights(reliability)
        fused = w_p * patient_resampled + w_n * self.norm_template
        return fused, (w_p, w_n)

    @staticmethod
    def gait_normality(current_cycle: np.ndarray,
                       target_template: np.ndarray):
        """
        Simple Gait Normality index based on normalized L2 distance
        between the current cycle and the fused target template.

        Returns a score in (0, 1], where 1 means identical.
        """
        x = current_cycle
        y = target_template
        L = len(y)
        x_phase = np.linspace(0, 1, len(x))
        y_phase = np.linspace(0, 1, L)
        x_res = np.interp(y_phase, x_phase, x)

        diff = x_res - y
        d = np.linalg.norm(diff) / np.sqrt(L)
        # Convert to similarity: 1 / (1 + d / c)
        c = 10.0
        score = 1.0 / (1.0 + d / c)
        return float(score)
