import numpy as np

def dtw_distance(x: np.ndarray, y: np.ndarray):
    """
    Classic dynamic time warping distance between two sequences.

    Parameters
    ----------
    x, y : 1D arrays

    Returns
    -------
    dist : float
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n, m = len(x), len(y)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = (x[i - 1] - y[j - 1]) ** 2
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(np.sqrt(D[n, m]))


def normalized_dtw_distance(patient_cycle, normal_template):
    """
    Resample both to the same length (e.g., 101 points) and compute DTW.
    """
    L = 101
    phase = np.linspace(0, 1, L)
    x_phase = np.linspace(0, 1, len(patient_cycle))
    y_phase = np.linspace(0, 1, len(normal_template))
    x_res = np.interp(phase, x_phase, patient_cycle)
    y_res = np.interp(phase, y_phase, normal_template)
    return dtw_distance(x_res, y_res)
