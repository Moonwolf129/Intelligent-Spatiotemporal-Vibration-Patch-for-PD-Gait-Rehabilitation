import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.anova import AnovaRM
from itertools import combinations


def repeated_measures_anova(df: pd.DataFrame,
                            subject_col: str,
                            condition_col: str,
                            value_col: str):
    """
    One-way repeated-measures ANOVA.

    Parameters
    ----------
    df : DataFrame
        Long-format data with columns [subject_col, condition_col, value_col].
    """
    aov = AnovaRM(df, depvar=value_col,
                  subject=subject_col,
                  within=[condition_col])
    res = aov.fit()
    return res


def bonferroni_posthoc(df: pd.DataFrame,
                       subject_col: str,
                       condition_col: str,
                       value_col: str):
    """
    Simple Bonferroni-corrected paired t-tests between conditions.
    """
    from scipy.stats import ttest_rel
    conds = df[condition_col].unique()
    pairs = list(combinations(conds, 2))
    results = []
    m = len(pairs)
    for c1, c2 in pairs:
        d1 = df[df[condition_col] == c1].sort_values(subject_col)[value_col].values
        d2 = df[df[condition_col] == c2].sort_values(subject_col)[value_col].values
        t_stat, p_val = ttest_rel(d1, d2)
        p_corrected = min(1.0, p_val * m)
        results.append({
            "cond1": c1,
            "cond2": c2,
            "t": t_stat,
            "p_raw": p_val,
            "p_bonf": p_corrected
        })
    return pd.DataFrame(results)
