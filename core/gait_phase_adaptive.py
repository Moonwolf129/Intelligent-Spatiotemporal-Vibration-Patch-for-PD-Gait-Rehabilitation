import numpy as np
from dataclasses import dataclass
from .signal_utils import lowpass_filter


@dataclass
class GaitPhaseAdaptiveParams:
    fs: float = 50.0           # IMU sampling frequency (Hz)
    stance_alpha: float = 0.1  # filter weight in stance (more accel)
    swing_alpha: float = 0.8   # filter weight in swing (more gyro)
    trans_alpha: float = 0.4   # in transition
    var_window: float = 0.4    # window length (s) for variance
    acc_var_thresh: float = 0.5  # threshold separating stance/swing
    lp_cutoff: float = 5.0       # low-pass cutoff for final angle


class GaitPhaseAdaptiveFilter:
    """
    Single-IMU Gait Phase Adaptive complementary filter.

    Coordinate convention:
    - acc = [ax, ay, az]
    - gyro = [gx, gy, gz]
    - We assume sagittal-plane flexion-extension is dominated by rotation
      about the mediolateral axis => pitch angle.
    """

    def __init__(self, params: GaitPhaseAdaptiveParams = None):
        self.params = params or GaitPhaseAdaptiveParams()
        self.prev_angle_gyro = 0.0
        self.initialized = False

    @staticmethod
    def _acc_to_pitch(acc):
        """
        Compute pitch angle (deg) from accelerometer.
        pitch = atan2(-ax, sqrt(ay^2 + az^2)) or similar depending on axes.
        """
        ax, ay, az = acc
        pitch = np.arctan2(-ax, np.sqrt(ay ** 2 + az ** 2))
        return np.degrees(pitch)

    def _update_phase_labels(self, acc_z_hist, theta_hist, n_window):
        """
        Simple phase classifier using vertical acceleration variance and angle trend.
        Returns one of: "stance", "swing", "transition".
        """
        if len(acc_z_hist) < n_window:
            return "transition"

        window = acc_z_hist[-n_window:]
        var_z = np.var(window)

        # Trend from last few samples
        if len(theta_hist) >= 3:
            trend = theta_hist[-1] - theta_hist[-3]
        else:
            trend = 0.0

        if var_z < self.params.acc_var_thresh and abs(trend) < 1.0:
            return "stance"
        elif var_z > self.params.acc_var_thresh and abs(trend) > 1.0:
            return "swing"
        else:
            return "transition"

    def update(self, acc, gyro, dt, acc_z_hist, theta_hist):
        """
        Update filter with one IMU sample.

        Parameters
        ----------
        acc : (3,) array
        gyro : (3,) array
            Angular velocity (deg/s) or (rad/s). Here we assume deg/s.
        dt : float
        acc_z_hist : list
            History of vertical acceleration az for phase detection.
        theta_hist : list
            History of fused thigh angles.

        Returns
        -------
        theta_fused : float
            Fused thigh angle (deg).
        phase : str
            "stance", "swing" or "transition".
        """
        # 1. Compute accelerometer-based pitch
        theta_acc = self._acc_to_pitch(acc)

        # 2. Gyro integration around mediolateral axis (assuming gy = pitch rate)
        gx, gy, gz = gyro
        theta_gyro = self.prev_angle_gyro + gy * dt

        if not self.initialized:
            theta_gyro = theta_acc
            self.initialized = True

        # 3. Determine gait phase
        fs = self.params.fs
        n_window = max(3, int(self.params.var_window * fs))
        phase = self._update_phase_labels(acc_z_hist, theta_hist, n_window)

        if phase == "stance":
            alpha = self.params.stance_alpha
        elif phase == "swing":
            alpha = self.params.swing_alpha
        else:
            alpha = self.params.trans_alpha

        # 4. Complementary fusion
        theta_fused = alpha * theta_gyro + (1.0 - alpha) * theta_acc
        self.prev_angle_gyro = theta_gyro

        return theta_fused, phase

    def run_batch(self, acc, gyro, t):
        """
        Run the filter on full IMU sequence.

        Parameters
        ----------
        acc : (N, 3)
        gyro : (N, 3)
        t : (N,)

        Returns
        -------
        theta : (N,)
            Smoothed fused thigh angle (deg).
        phases : list of str
            Detected phase labels.
        """
        N = acc.shape[0]
        theta = np.zeros(N)
        phases = []
        acc_z_hist, theta_hist = [], []

        for i in range(N):
            if i == 0:
                dt = 1.0 / self.params.fs
            else:
                dt = t[i] - t[i - 1]
            a = acc[i]
            g = gyro[i]

            theta_i, phase = self.update(a, g, dt, acc_z_hist, theta_hist)
            theta[i] = theta_i
            phases.append(phase)

            acc_z_hist.append(a[2])
            theta_hist.append(theta_i)

        # Final low-pass smoothing
        theta_smooth = lowpass_filter(theta,
                                      cutoff=self.params.lp_cutoff,
                                      fs=self.params.fs,
                                      order=4)
        return theta_smooth, phases
