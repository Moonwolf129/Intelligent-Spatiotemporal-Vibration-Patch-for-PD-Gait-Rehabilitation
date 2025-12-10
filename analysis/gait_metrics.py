import numpy as np
import pandas as pd

def compute_stride_length(foot_positions):
    """
    Compute stride length from footprint data.

    Parameters
    ----------
    foot_positions : DataFrame
        Columns: ["time", "foot", "x", "y"] where "foot" in {"L", "R"}

    Returns
    -------
    strides : DataFrame
        Columns: ["foot", "time_start", "time_end", "stride_length"]
    """
    rows = []
    for foot in ["L", "R"]:
        df = foot_positions[foot_positions["foot"] == foot].sort_values("time")
        for i in range(len(df) - 1):
            x1 = df.iloc[i]["x"]
            x2 = df.iloc[i + 1]["x"]
            t1 = df.iloc[i]["time"]
            t2 = df.iloc[i + 1]["time"]
            stride_len = x2 - x1
            rows.append({
                "foot": foot,
                "time_start": t1,
                "time_end": t2,
                "stride_length": stride_len
            })
    return pd.DataFrame(rows)


def compute_gait_cycle_duration(strides: pd.DataFrame):
    """
    Use stride start/end times to compute gait-cycle duration.
    """
    durations = strides.copy()
    durations["gait_cycle_duration"] = durations["time_end"] - durations["time_start"]
    return durations


def compute_stance_phase_ratio(stances: pd.DataFrame,
                               cycles: pd.DataFrame):
    """
    Compute stance phase ratio = stance_time / gait_cycle_duration.

    Parameters
    ----------
    stances : DataFrame
        Columns: ["foot", "cycle_id", "stance_time"]
    cycles : DataFrame
        Columns: ["foot", "cycle_id", "gait_cycle_duration"]

    Returns
    -------
    df : DataFrame
        With stance-phase ratio per gait cycle.
    """
    df = pd.merge(cycles, stances, on=["foot", "cycle_id"])
    df["stance_ratio"] = df["stance_time"] / df["gait_cycle_duration"]
    return df


def summarize_variances(values: np.ndarray):
    """
    Return variance (and SD) of a 1D array for reporting.

    Parameters
    ----------
    values : array-like

    Returns
    -------
    var, sd
    """
    values = np.asarray(values)
    return float(np.var(values, ddof=1)), float(np.std(values, ddof=1))
