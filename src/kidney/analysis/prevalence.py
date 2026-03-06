"""Core weighted prevalence and CI calculations."""

import numpy as np
import pandas as pd


def weighted_prevalence(df: pd.DataFrame, mask: pd.Series) -> float:
    """Compute weighted prevalence percentage (0-100).

    Falls back to unweighted mean if no survey_weight column.
    """
    if "survey_weight" not in df.columns:
        return float(mask.mean() * 100)
    weights = df["survey_weight"][mask.index]
    total = weights.sum()
    if total == 0:
        return 0.0
    return float((mask.astype(float) * weights).sum() / total * 100)


def weighted_stats(df: pd.DataFrame, mask: pd.Series) -> tuple[float, float, float]:
    """Weighted prevalence with 95% CI using Kish's effective sample size.

    Returns (mean_pct, lower_ci, upper_ci).
    """
    if "survey_weight" not in df.columns:
        n = len(df)
        prev = mask.mean() * 100
        se = np.sqrt(prev * (100 - prev) / n) if n > 0 else 0.0
        return float(prev), float(prev - 1.96 * se), float(prev + 1.96 * se)

    weights = df["survey_weight"][mask.index]
    total = weights.sum()
    if total == 0:
        return 0.0, 0.0, 0.0

    mean_pct = (mask.astype(float) * weights).sum() / total * 100
    neff = total**2 / (weights**2).sum()
    se = np.sqrt(mean_pct * (100 - mean_pct) / neff)
    return float(mean_pct), float(mean_pct - 1.96 * se), float(mean_pct + 1.96 * se)
