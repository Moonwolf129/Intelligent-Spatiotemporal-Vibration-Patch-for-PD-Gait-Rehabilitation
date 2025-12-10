import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from dataclasses import dataclass

def butter_lowpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    Wn = cutoff / nyq
    b, a = butter(order, Wn, btype="low")
    return b, a


def lowpass_filter(signal, cutoff, fs, order=4):
    b, a = butter_lowpass(cutoff, fs, order)
    return filtfilt(b, a, signal)


@dataclass
class GaitCycle:
    theta: np.ndarray
    time: np.ndarray
    heel_strike_idx: int
    toe_off_idx: int


def detect_gait_cycles(theta: np.ndarray,
                       t: np.ndarray,
                       prominence: float = 5.0,
                       distance: int = 30):
    """
    Detect gait cycles based on thigh-angle local minima (e.g. heel strikes).
    This is a simple reference implementation; thresholds can be tuned.

    Parameters
    ----------
    theta : (N,) array
        Thigh flexion/extension angle (deg).
    t : (N,) array
        Time stamps (s).
    prominence : float
        Prominence for minima detection.
    distance : int
        Minimal number of samples between heel strikes.

    Returns
    -------
    indices : list of int
        Indices of detected heel strikes.
    """
    # Heel strike often corresponds to local minimum in flexion-extension
    inv_theta = -theta
    peaks, _ = find_peaks(inv_theta, prominence=prominence, distance=distance)
    return peaks.tolist()


def segment_gait_cycles(theta: np.ndarray,
                        t: np.ndarray,
                        heel_strikes: np.ndarray = None):
    """
    Segment continuous thigh-angle trajectory into gait cycles.

    Parameters
    ----------
    theta : (N,) array
    t : (N,) array
    heel_strikes : list or array, optional
        Pre-detected heel-strike indices. If None, will be detected.

    Returns
    -------
    cycles : list of GaitCycle
    """
    if heel_strikes is None:
        heel_strikes = detect_gait_cycles(theta, t)

    cycles = []
    for i in range(len(heel_strikes) - 1):
        start = heel_strikes[i]
        end = heel_strikes[i + 1]
        if end <= start + 5:
            continue
        seg_theta = theta[start:end]
        seg_t = t[start:end]
        # For simplicity we do not explicitly detect toe-off here
        cycle = GaitCycle(theta=seg_theta,
                          time=seg_t,
                          heel_strike_idx=0,
                          toe_off_idx=len(seg_theta) // 2)
        cycles.append(cycle)
    return cycles
