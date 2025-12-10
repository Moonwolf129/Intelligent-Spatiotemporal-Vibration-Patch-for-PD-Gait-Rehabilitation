import numpy as np
from scipy.signal import butter, filtfilt


def bandpass_filter(signal, fs, low=10.0, high=500.0, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal)


def notch_filter_50hz(signal, fs, q=30.0):
    """
    Simple 50 Hz notch using IIR (biquad). For clarity, here we approximate
    with a narrow band-stop Butterworth around 50 Hz.
    """
    nyq = 0.5 * fs
    low = 49.0 / nyq
    high = 51.0 / nyq
    b, a = butter(2, [low, high], btype="bandstop")
    return filtfilt(b, a, signal)


def full_wave_rectify(signal):
    return np.abs(signal)


def rms_envelope(signal, fs, win_ms=200, overlap=0.5):
    """
    Sliding-window RMS envelope.

    Parameters
    ----------
    signal : (N,) array
    fs : float
    win_ms : float
    overlap : float

    Returns
    -------
    rms : (M,) array
    t_rms : (M,) array
    """
    N = len(signal)
    win_len = int(win_ms / 1000.0 * fs)
    step = int(win_len * (1.0 - overlap))
    if step <= 0:
        step = 1
    rms_list = []
    t_list = []
    for start in range(0, N - win_len + 1, step):
        seg = signal[start:start + win_len]
        rms_val = np.sqrt(np.mean(seg ** 2))
        rms_list.append(rms_val)
        t_list.append((start + win_len / 2) / fs)
    return np.array(rms_list), np.array(t_list)


def iemg(rectified_signal, fs):
    """
    Integrated EMG over full trial.

    Parameters
    ----------
    rectified_signal : (N,) array
    fs : float

    Returns
    -------
    val : float
    """
    dt = 1.0 / fs
    return float(np.sum(rectified_signal) * dt)


def percent_reduction_iemg(iemg_control, iemg_condition):
    """
    % reduction = (I_control - I_condition) / I_control * 100
    """
    return float((iemg_control - iemg_condition) / (iemg_control + 1e-8) * 100.0)
