"""Unified eligibility and condition masks for KDIGO kidney donation criteria."""

import pandas as pd

from kidney.analysis.prevalence import weighted_prevalence
from kidney.config import ELIGIBILITY_BMI_THRESHOLD, OBESITY_THRESHOLD


def eligible_mask(
    df: pd.DataFrame,
    *,
    bmi_threshold: float = ELIGIBILITY_BMI_THRESHOLD,
    exclude_prediabetes: bool = True,
) -> pd.Series:
    """KDIGO eligibility mask.

    Args:
        df: Preprocessed DataFrame.
        bmi_threshold: Upper BMI bound (default 35).
        exclude_prediabetes: When False, only check diabetes (used in
            manuscript "impact of prediabetes" analysis).
    """
    mask = (df["diabetes"] != 1) & (df["hypertension"] != 1) & (df["BMI"] < bmi_threshold)
    if exclude_prediabetes:
        mask = mask & (df["prediabetes"] != 1)
    return mask


def any_condition_mask(
    df: pd.DataFrame,
    *,
    obesity_threshold: float = OBESITY_THRESHOLD,
    include_historic_htn: bool = True,
) -> pd.Series:
    """Union of health conditions.

    Args:
        df: Preprocessed DataFrame.
        obesity_threshold: BMI cutoff for obesity (default 30).
        include_historic_htn: Set False for insurance analysis which
            excludes historic_hypertension.
    """
    mask = (
        (df["diabetes"] == 1)
        | (df["prediabetes"] == 1)
        | (df["hypertension"] == 1)
        | (df["BMI"] > obesity_threshold)
    )
    if include_historic_htn:
        mask = mask | (df["historic_hypertension"] == 1)
    return mask


def eligibility_metrics(
    dataframes: dict[str, pd.DataFrame],
    bmi_threshold: float = ELIGIBILITY_BMI_THRESHOLD,
) -> tuple[list[float], list[float]]:
    """National eligibility percentages and absolute weighted counts per year."""
    years = sorted(dataframes)
    pcts: list[float] = []
    counts: list[float] = []
    for year in years:
        df = dataframes[year]
        mask = eligible_mask(df, bmi_threshold=bmi_threshold)
        pcts.append(weighted_prevalence(df, mask))
        if "survey_weight" in df.columns:
            counts.append(float(df.loc[mask, "survey_weight"].sum()))
        else:
            counts.append(float(mask.sum()))
    return pcts, counts
